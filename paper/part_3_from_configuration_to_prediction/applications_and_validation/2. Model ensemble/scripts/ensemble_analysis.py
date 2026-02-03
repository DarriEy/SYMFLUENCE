#!/usr/bin/env python3
"""
Multi-model ensemble analysis for SYMFLUENCE paper Section 4.2.
Generates three publication-quality figures for the Bow River at Banff case study.

Models included (calibration KGE > 0.5):
  SUMMA, FUSE, GR4J, HBV, HYPE, VIC, LSTM, RHESSys
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/"
    "domain_Bow_at_Banff_lumped_era5"
)
PROJECT_ROOT = Path(
    "/Users/darrieythorsson/compHydro/papers/article_2_symfluence/"
    "applications_and_validation /2. Model ensemble"
)
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OBS_FILE = (
    DATA_ROOT / "observations" / "streamflow" / "preprocessed" /
    "Bow_at_Banff_lumped_era5_streamflow_processed.csv"
)
OPT_DIR = DATA_ROOT / "optimization"

# Basin area (m²) — from SUMMA attributes.nc / shapefiles
BASIN_AREA_M2 = 2_209_951_307.64
BASIN_AREA_KM2 = BASIN_AREA_M2 / 1e6  # 2209.95 km²

# Model specification
# Units key:
#   "m_per_s"  — SUMMA/FUSE native output (m/s), multiply by BASIN_AREA_M2
#   "mm_per_d" — GR4J output (mm/day), multiply by BASIN_AREA_KM2 / 86.4
#   "cms"      — already m³/s
MODEL_SPEC = {
    "SUMMA": {
        "file": OPT_DIR / "SUMMA/dds_run_1/final_evaluation/run_1_timestep.nc",
        "fmt": "netcdf",
        "var": "averageRoutedRunoff",
        "units": "m_per_s",
    },
    "FUSE": {
        "file": OPT_DIR / "FUSE/dds_run_1/final_evaluation/run_1_timestep.nc",
        "fmt": "netcdf",
        "var": "averageRoutedRunoff",
        "units": "m_per_s",
    },
    "GR4J": {
        "file": OPT_DIR / "GR/dds_run_1/final_evaluation/GR_results.csv",
        "fmt": "csv",
        "var": "q_sim",
        "units": "mm_per_d",
    },
    "HBV": {
        "file": OPT_DIR / "HBV/dds_run_1/final_evaluation/Bow_at_Banff_lumped_era5_hbv_output.csv",
        "fmt": "csv",
        "var": "streamflow_cms",
        "units": "cms",
    },
    "HYPE": {
        "file": OPT_DIR / "HYPE/dds_run_1/final_evaluation/timeCOUT.txt",
        "fmt": "tsv",
        "var": "1",
        "units": "cms",
    },
    "VIC": {
        "file": OPT_DIR / "VIC/dds_run_1/final_evaluation/vic_output.2002-01-01.nc",
        "fmt": "netcdf_vic",
        "var": "OUT_RUNOFF+OUT_BASEFLOW",
        "units": "mm_per_d",
    },
    "LSTM": {
        "file": DATA_ROOT / "results/run_2_results.csv",
        "fmt": "csv_hourly",
        "var": "LSTM_discharge_cms",
        "units": "cms",
    },
    "RHESSys": {
        "file": OPT_DIR / "RHESSys/dds_run_7/final_evaluation/rhessys_basin.daily",
        "fmt": "rhessys",
        "var": "routedstreamflow",
        "units": "mm_per_d",
    },
}

# JSON metric files per model
METRIC_FILES = {
    "SUMMA":   OPT_DIR / "SUMMA/dds_run_1/run_1_dds_final_evaluation.json",
    "FUSE":    OPT_DIR / "FUSE/dds_run_1/run_1_dds_final_evaluation.json",
    "GR4J":    OPT_DIR / "GR/dds_run_1/run_1_dds_final_evaluation.json",
    "HBV":     OPT_DIR / "HBV/dds_run_1/run_1_dds_final_evaluation.json",
    "HYPE":    OPT_DIR / "HYPE/dds_run_1/run_1_dds_final_evaluation.json",
    "VIC":     OPT_DIR / "VIC/dds_run_1/run_1_dds_final_evaluation.json",
    "LSTM":    OPT_DIR / "LSTM/dds_run_1/run_1_dds_final_evaluation.json",
    "RHESSys": OPT_DIR / "RHESSys/dds_run_7/run_7_dds_final_evaluation.json",
}

KGE_THRESHOLD = 0.5

# Common analysis period
PERIOD_START = "2003-01-01"
PERIOD_END   = "2009-12-31"

# Calibration / evaluation split
CALIB_START = "2003-01-01"
CALIB_END   = "2005-12-31"
EVAL_START  = "2006-01-01"
EVAL_END    = "2009-12-31"

# Plot period (skip first spin-up year)
PLOT_START  = "2004-01-01"
PLOT_END    = "2009-12-31"

# Zoom water year
ZOOM_START = "2005-04-01"
ZOOM_END   = "2005-10-31"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "SUMMA":   "#1f77b4",
    "FUSE":    "#ff7f0e",
    "GR4J":    "#2ca02c",
    "HBV":     "#d62728",
    "HYPE":    "#9467bd",
    "VIC":     "#e377c2",
    "LSTM":    "#8c564b",
    "RHESSys": "#17becf",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_obs(path: Path) -> pd.Series:
    """Load observed streamflow, resample hourly -> daily mean."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    daily = df["discharge_cms"].resample("D").mean()
    return daily.loc[PERIOD_START:PERIOD_END]


def load_netcdf(path: Path, var: str) -> pd.Series:
    """Load a streamflow variable from a NetCDF file, resample to daily."""
    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required for NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )
    ds = xr.open_dataset(path)
    da = ds[var]
    # Squeeze extra dimensions
    while da.ndim > 1:
        for dim in da.dims:
            if dim != "time":
                da = da.isel({dim: 0})
    s = da.to_series()
    s.index = pd.DatetimeIndex(s.index)
    # Convert to daily if sub-daily
    if len(s) > 1 and (s.index[1] - s.index[0]) < pd.Timedelta("1D"):
        s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_csv_daily(path: Path, var: str) -> pd.Series:
    """Load daily CSV (GR4J, HBV)."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    s = df[var]
    s.index = pd.DatetimeIndex(s.index)
    return s.loc[PERIOD_START:PERIOD_END]


def load_csv_hourly(path: Path, var: str) -> pd.Series:
    """Load hourly CSV (LSTM), resample to daily mean."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df[var].dropna()
    s = s.resample("D").mean()
    return s.loc[PERIOD_START:PERIOD_END]


def load_netcdf_vic(path: Path, var: str) -> pd.Series:
    """Load VIC NetCDF output, summing OUT_RUNOFF + OUT_BASEFLOW (mm/day)."""
    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required for NetCDF files. "
            "Install with: pip install xarray netCDF4"
        )
    ds = xr.open_dataset(path)
    runoff = ds["OUT_RUNOFF"].squeeze()
    baseflow = ds["OUT_BASEFLOW"].squeeze()
    total = (runoff + baseflow).to_series()
    total.index = pd.DatetimeIndex(total.index)
    return total.loc[PERIOD_START:PERIOD_END]


def load_tsv_hype(path: Path, var: str) -> pd.Series:
    """Load HYPE timeCOUT.txt (tab-separated, 1-line header comment)."""
    df = pd.read_csv(
        path, sep="\t", skiprows=1, index_col="DATE", parse_dates=True
    )
    s = df[var].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s.loc[PERIOD_START:PERIOD_END]


def load_rhessys(path: Path, var: str) -> pd.Series:
    """Load RHESSys basin.daily output (whitespace-separated)."""
    df = pd.read_csv(path, sep=r"\s+")
    # Build datetime index from day, month, year columns
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day"]].rename(
            columns={"year": "year", "month": "month", "day": "day"}
        )
    )
    df = df.set_index("datetime")
    s = df[var].astype(float)
    return s.loc[PERIOD_START:PERIOD_END]


LOADERS = {
    "netcdf":     load_netcdf,
    "netcdf_vic": load_netcdf_vic,
    "csv":        load_csv_daily,
    "csv_hourly": load_csv_hourly,
    "tsv":        load_tsv_hype,
    "rhessys":    load_rhessys,
}


def load_model(spec: dict) -> pd.Series:
    """Dispatch to the right loader and apply unit conversion to m³/s."""
    loader = LOADERS[spec["fmt"]]
    s = loader(spec["file"], spec["var"])
    units = spec.get("units", "cms")
    if units == "m_per_s":
        s = s * BASIN_AREA_M2
    elif units == "mm_per_d":
        s = s * BASIN_AREA_KM2 / 86.4
    return s


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def load_metrics(path: Path) -> dict:
    """Load JSON metrics, normalising HBV-style prefixed keys."""
    with open(path) as f:
        raw = json.load(f)

    out = {"calibration": {}, "evaluation": {}}
    for period, key_prefix in [
        ("calibration", "calibration_metrics"),
        ("evaluation", "evaluation_metrics"),
    ]:
        block = raw[key_prefix]
        for metric in ("KGE", "r", "alpha", "beta", "NSE", "RMSE", "PBIAS"):
            val = block.get(metric)
            if val is None:
                prefix = "Calib_" if period == "calibration" else "Eval_"
                val = block.get(prefix + metric)
            if val is not None:
                out[period][metric] = val
    return out


def kge_from_series(sim: pd.Series, obs: pd.Series) -> dict:
    """Compute KGE and its components from aligned series."""
    common = sim.dropna().index.intersection(obs.dropna().index)
    s, o = sim.loc[common].values, obs.loc[common].values
    r = np.corrcoef(s, o)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"KGE": kge, "r": r, "alpha": alpha, "beta": beta}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading observed streamflow...")
    obs = load_obs(OBS_FILE)
    print(
        f"  Obs period: {obs.index[0].date()} to {obs.index[-1].date()}, "
        f"n={len(obs)}"
    )

    # Load all models
    simulations = {}
    for name, spec in MODEL_SPEC.items():
        try:
            s = load_model(spec)
            print(
                f"  {name}: loaded {len(s)} daily values "
                f"({s.index[0].date()} – {s.index[-1].date()})"
            )
            simulations[name] = s
        except Exception as e:
            print(f"  {name}: FAILED to load – {e}")

    # Load JSON metrics
    all_metrics = {}
    for name, path in METRIC_FILES.items():
        if name in simulations:
            try:
                all_metrics[name] = load_metrics(path)
            except Exception as e:
                print(f"  {name}: metrics load failed – {e}")

    # Apply KGE filter
    included = {}
    for name, sim in simulations.items():
        m = all_metrics.get(name, {})
        calib_kge = m.get("calibration", {}).get("KGE", None)
        if calib_kge is not None and calib_kge > KGE_THRESHOLD:
            included[name] = sim
            print(f"  {name}: INCLUDED (Calib KGE = {calib_kge:.3f})")
        else:
            print(f"  {name}: EXCLUDED (Calib KGE = {calib_kge})")

    if not included:
        print("No models passed the KGE filter. Exiting.")
        return

    # Align to common daily index
    common_idx = obs.index
    for s in included.values():
        common_idx = common_idx.intersection(s.index)
    print(
        f"\nCommon period: {common_idx[0].date()} to {common_idx[-1].date()}, "
        f"n={len(common_idx)}"
    )

    obs_aligned = obs.loc[common_idx]
    sim_aligned = {k: v.loc[common_idx] for k, v in included.items()}

    # Build ensemble
    ensemble_df = pd.DataFrame(sim_aligned, index=common_idx)
    ens_mean = ensemble_df.mean(axis=1)
    ens_median = ensemble_df.median(axis=1)
    ens_min = ensemble_df.min(axis=1)
    ens_max = ensemble_df.max(axis=1)

    # Ensemble metrics
    ens_mean_metrics = kge_from_series(ens_mean, obs_aligned)
    ens_median_metrics = kge_from_series(ens_median, obs_aligned)
    print(f"\nEnsemble mean  KGE: {ens_mean_metrics['KGE']:.3f}")
    print(f"Ensemble median KGE: {ens_median_metrics['KGE']:.3f}")

    # ==================================================================
    # FIGURE A: Multi-Model Hydrograph
    # ==================================================================
    print("\nGenerating Figure A: Multi-Model Hydrograph...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [2, 1.2]}
    )

    # Top panel: full period
    plot_mask = (obs_aligned.index >= PLOT_START) & (obs_aligned.index <= PLOT_END)
    plot_idx = obs_aligned.index[plot_mask]
    ax1.plot(
        plot_idx, obs_aligned.loc[plot_idx],
        color="black", linestyle="--", linewidth=1.2, label="Observed", zorder=5,
    )
    for name in sorted(sim_aligned.keys()):
        s = sim_aligned[name]
        ax1.plot(
            plot_idx, s.loc[plot_idx],
            color=MODEL_COLORS[name], linewidth=0.8, alpha=0.85, label=name,
        )

    # Calibration / evaluation boundary
    ax1.axvline(pd.Timestamp(EVAL_START), color="grey", linestyle=":", linewidth=0.8)
    ax1.text(
        pd.Timestamp("2004-10-01"), 285,
        "Calibration", fontsize=8.5, color="grey", ha="center",
        fontstyle="italic",
    )
    ax1.text(
        pd.Timestamp("2007-10-01"), 285,
        "Evaluation", fontsize=8.5, color="grey", ha="center",
        fontstyle="italic",
    )
    ax1.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax1.set_title("(a) Multi-model hydrograph — Bow River at Banff (2004–2009)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    ax1.set_ylim(bottom=0, top=310)

    # Bottom panel: zoomed water year
    zoom_mask = (obs_aligned.index >= ZOOM_START) & (obs_aligned.index <= ZOOM_END)
    zoom_idx = obs_aligned.index[zoom_mask]
    ax2.plot(
        zoom_idx, obs_aligned.loc[zoom_idx],
        color="black", linestyle="--", linewidth=1.3, label="Observed", zorder=5,
    )
    for name in sorted(sim_aligned.keys()):
        s = sim_aligned[name]
        ax2.plot(
            zoom_idx, s.loc[zoom_idx],
            color=MODEL_COLORS[name], linewidth=1.0, alpha=0.9, label=name,
        )
    ax2.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax2.set_title("(b) Snowmelt season detail (Apr–Oct 2005)")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.set_xlim(pd.Timestamp(ZOOM_START), pd.Timestamp(ZOOM_END))

    # Shared legend below both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=8,
        framealpha=0.95, fontsize=8, borderaxespad=0.3,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(FIG_DIR / "fig_ensemble_hydrograph.png")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_ensemble_hydrograph.png'}")

    # ==================================================================
    # FIGURE B: KGE Decomposition
    # ==================================================================
    print("Generating Figure B: KGE Decomposition...")
    models_ordered = sorted(all_metrics.keys())
    components = ["r", "alpha", "beta"]
    comp_labels = [
        "$r$ (correlation)",
        r"$\alpha$ (variability)",
        r"$\beta$ (bias)",
    ]
    periods = ["calibration", "evaluation"]
    period_labels = ["Calibration", "Evaluation"]

    n_models = len(models_ordered)
    x = np.arange(n_models)
    bar_w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)

    for ci, (comp, clabel) in enumerate(zip(components, comp_labels)):
        ax = axes[ci]
        for pi, (period, plabel) in enumerate(zip(periods, period_labels)):
            vals = [
                all_metrics[m].get(period, {}).get(comp, np.nan)
                for m in models_ordered
            ]
            offset = (pi - 0.5) * bar_w
            ax.bar(
                x + offset, vals, bar_w, label=plabel,
                color=["#4c72b0", "#dd8452"][pi], alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )
            for xi, v in zip(x + offset, vals):
                if not np.isnan(v):
                    # Stagger label position: calibration left, evaluation right
                    ha = "right" if pi == 0 else "left"
                    ax.text(
                        xi, v + 0.008, f"{v:.2f}", ha=ha, va="bottom",
                        fontsize=6, rotation=50,
                    )
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(models_ordered, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(clabel)
        ax.set_ylim(0.6, 1.35)
        if ci == 0:
            ax.legend(framealpha=0.9, fontsize=8, loc="lower left")

    fig.suptitle(
        "KGE decomposition — calibration vs. evaluation", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_kge_decomposition.png")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_kge_decomposition.png'}")

    # ==================================================================
    # FIGURE C: Ensemble Spread / Envelope + FDC
    # ==================================================================
    print("Generating Figure C: Ensemble Envelope + FDC...")
    fig, (ax_env, ax_fdc) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.30},
    )

    # Left: envelope time series (evaluation period)
    env_mask = (common_idx >= EVAL_START) & (common_idx <= EVAL_END)
    idx_env = common_idx[env_mask]

    ax_env.fill_between(
        idx_env, ens_min.loc[idx_env], ens_max.loc[idx_env],
        color="#a6cee3", alpha=0.45, label="Ensemble range",
    )
    ax_env.plot(
        idx_env, obs_aligned.loc[idx_env],
        color="black", linewidth=1.0, linestyle="--", label="Observed",
    )
    ax_env.plot(
        idx_env, ens_mean.loc[idx_env],
        color="#1f78b4", linewidth=0.9,
        label=f"Ensemble mean (KGE={ens_mean_metrics['KGE']:.2f})",
    )
    ax_env.plot(
        idx_env, ens_median.loc[idx_env],
        color="#e31a1c", linewidth=0.9,
        label=f"Ensemble median (KGE={ens_median_metrics['KGE']:.2f})",
    )

    ax_env.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_env.set_title("(a) Ensemble envelope — evaluation period")
    ax_env.legend(
        loc="upper left", fontsize=7.5, framealpha=0.95,
        borderaxespad=0.5,
    )
    ax_env.xaxis.set_major_locator(mdates.YearLocator())
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Right: Flow Duration Curve
    def fdc(series):
        sorted_vals = np.sort(series.dropna().values)[::-1]
        exceedance = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        return exceedance, sorted_vals

    exc_obs, val_obs = fdc(obs_aligned)
    ax_fdc.plot(
        exc_obs, val_obs, color="black", linewidth=1.2,
        linestyle="--", label="Observed",
    )
    exc_mean, val_mean = fdc(ens_mean)
    ax_fdc.plot(exc_mean, val_mean, color="#1f78b4", linewidth=1.0, label="Ens. mean")
    exc_med, val_med = fdc(ens_median)
    ax_fdc.plot(exc_med, val_med, color="#e31a1c", linewidth=1.0, label="Ens. median")

    for name in sorted(sim_aligned.keys()):
        exc_m, val_m = fdc(sim_aligned[name])
        ax_fdc.plot(
            exc_m, val_m, color=MODEL_COLORS[name],
            linewidth=0.6, alpha=0.6, label=name,
        )

    ax_fdc.set_xlabel("Exceedance probability (%)")
    ax_fdc.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_fdc.set_title("(b) Flow duration curve")
    ax_fdc.set_yscale("log")
    ax_fdc.legend(
        loc="lower left", fontsize=6.5, framealpha=0.95, ncol=1,
        borderaxespad=0.5,
    )
    ax_fdc.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ensemble_envelope.png")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_ensemble_envelope.png'}")

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = (
        f"{'Model':<8} {'Cal KGE':>8} {'Cal r':>7} {'Cal α':>7} {'Cal β':>7} "
        f"{'Eval KGE':>9} {'Eval r':>7} {'Eval α':>8} {'Eval β':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in models_ordered:
        cal = all_metrics[m].get("calibration", {})
        evl = all_metrics[m].get("evaluation", {})
        print(
            f"{m:<8} {cal.get('KGE', 0):.3f}    {cal.get('r', 0):.3f}  "
            f"{cal.get('alpha', 0):.3f}  {cal.get('beta', 0):.3f}  "
            f"  {evl.get('KGE', 0):.3f}   {evl.get('r', 0):.3f}   "
            f"{evl.get('alpha', 0):.3f}   {evl.get('beta', 0):.3f}"
        )
    print("-" * len(header))
    print(
        f"{'Ens.mean':<8} {ens_mean_metrics['KGE']:.3f}    "
        f"{ens_mean_metrics['r']:.3f}  {ens_mean_metrics['alpha']:.3f}  "
        f"{ens_mean_metrics['beta']:.3f}"
    )
    print(
        f"{'Ens.med':<8} {ens_median_metrics['KGE']:.3f}    "
        f"{ens_median_metrics['r']:.3f}  {ens_median_metrics['alpha']:.3f}  "
        f"{ens_median_metrics['beta']:.3f}"
    )
    print("=" * 70)

    print("\nDone. All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
