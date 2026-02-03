#!/usr/bin/env python3
"""
Data-Centric Pipeline Visualisation for SYMFLUENCE Paper Section 4.12

Unlike the existing visualize_pipeline.py (which shows metadata: DAGs, volumes,
timing), this script shows the ACTUAL DATA flowing through the pipeline stages.

  Figure 1 -- Forcing transformation panel: raw ERA5 grid → remapped HRUs → SUMMA-ready
              for temperature and precipitation (Bow at Banff)
  Figure 2 -- Spatial remapping geometry: ERA5 grid cells overlaid on HRU polygons,
              with intersection weights and a remapped field example
  Figure 3 -- GRACE TWS remote sensing: time series of 3 Mascon solutions over
              Iceland basin with data gap visualisation
  Figure 4 -- Lapse-rate effect: temperature across elevation bands showing the
              correction applied during forcing preprocessing

Usage:
    python visualize_pipeline_data.py [--format png|pdf|svg]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_data_viz")

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DOMAIN_COLORS = {
    "paradise": "#2ca02c",
    "bow":      "#1f77b4",
    "iceland":  "#d62728",
}


# ============================================================
# Figure 1: Forcing Transformation Panel
# ============================================================
def fig_forcing_transformation(save_path: Path, fmt: str = "png"):
    """
    3-column × 2-row panel showing temperature and precipitation through
    the pipeline: raw ERA5 grid → remapped to HRUs → SUMMA-ready.

    Row 1: Air temperature (airtemp)
    Row 2: Precipitation rate (pptrate)
    Col 1: Raw ERA5 (lat×lon heatmap, single timestep)
    Col 2: Basin-averaged (HRU bar chart, same timestep)
    Col 3: Time series comparison (raw spatial mean vs basin-avg mean, 1 month)
    """
    bow_dir = DATA_DIR / "domain_Bow_at_Banff_semi_distributed"
    raw_dir = bow_dir / "forcing" / "raw_data"
    ba_dir = bow_dir / "forcing" / "basin_averaged_data"
    summa_dir = bow_dir / "forcing" / "SUMMA_input"

    raw_files = sorted(raw_dir.glob("*.nc"))
    ba_files = sorted(ba_dir.glob("*.nc"))
    summa_files = sorted(summa_dir.glob("*.nc"))

    if not raw_files or not ba_files:
        logger.warning("Missing forcing data for Bow domain; skipping Figure 1")
        return

    # Load January 2004 data
    ds_raw = xr.open_dataset(raw_files[0])
    ds_ba = xr.open_dataset(ba_files[0])
    ds_summa = xr.open_dataset(summa_files[0]) if summa_files else None

    variables = [
        ("airtemp", "Air temperature", "K", "RdYlBu_r"),
        ("pptrate", "Precipitation rate", "kg m$^{-2}$ s$^{-1}$", "YlGnBu"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, (vname, vlabel, vunit, cmap) in enumerate(variables):
        if vname not in ds_raw.data_vars or vname not in ds_ba.data_vars:
            logger.warning(f"Variable {vname} not found in data; skipping row")
            continue

        # Pick a representative timestep (noon on day 15)
        t_idx = min(14 * 24 + 12, len(ds_raw.time) - 1)

        # --- Col 0: Raw ERA5 grid (lat×lon heatmap) ---
        ax = axes[row, 0]
        raw_slice = ds_raw[vname].isel(time=t_idx)
        lats = ds_raw.latitude.values
        lons = ds_raw.longitude.values

        im = ax.pcolormesh(
            lons, lats, raw_slice.values,
            cmap=cmap, shading="nearest"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(vunit, fontsize=9)

        time_str = str(ds_raw.time.values[t_idx])[:16]
        ax.set_title(f"Raw ERA5 grid (6×7)\n{time_str}", fontsize=10)

        # Grid cell count annotation
        ax.text(0.02, 0.98, f"{len(lats)}×{len(lons)} = {len(lats)*len(lons)} cells",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        if row == 0:
            ax.set_title(f"(a) Raw ERA5 grid (6×7)\n{time_str}", fontsize=10, fontweight="bold")

        # --- Col 1: Basin-averaged (HRU dimension) ---
        ax = axes[row, 1]
        ba_slice = ds_ba[vname].isel(time=t_idx)
        hru_ids = np.arange(len(ba_slice.hru))
        values = ba_slice.values

        # Color bars by value
        norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        colors = sm.to_rgba(values)

        ax.bar(hru_ids, values, color=colors, edgecolor="white", linewidth=0.3, width=0.9)
        ax.set_xlabel("HRU index")
        ax.set_ylabel(vunit)
        ax.text(0.02, 0.98, "49 HRUs",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        if row == 0:
            ax.set_title("(b) Remapped to HRUs (49)\nSame timestep", fontsize=10, fontweight="bold")
        else:
            ax.set_title("Remapped to HRUs\nSame timestep", fontsize=10)

        # Show value range
        ax.axhline(np.nanmean(values), color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(0.98, 0.98, f"range: {np.nanmin(values):.2f}–{np.nanmax(values):.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # --- Col 2: Time series (1 month: raw spatial mean vs basin-avg mean) ---
        ax = axes[row, 2]

        # Raw: compute spatial mean per timestep
        raw_ts = ds_raw[vname].mean(dim=["latitude", "longitude"])
        # Basin-avg: compute mean across HRUs per timestep
        ba_ts = ds_ba[vname].mean(dim="hru")

        # Plot first 2 weeks for clarity
        n_show = min(14 * 24, len(raw_ts))
        time_hours = np.arange(n_show)

        ax.plot(time_hours, raw_ts.values[:n_show], color="#888888",
                linewidth=0.8, alpha=0.7, label="Raw (spatial mean)")
        ax.plot(time_hours, ba_ts.values[:n_show], color=DOMAIN_COLORS["bow"],
                linewidth=1.2, label="Basin-averaged (HRU mean)")

        if ds_summa is not None and vname in ds_summa.data_vars:
            summa_ts = ds_summa[vname].mean(dim="hru")
            ax.plot(time_hours, summa_ts.values[:n_show], color="#d62728",
                    linewidth=1.0, linestyle="--", alpha=0.8, label="SUMMA-ready")

        ax.set_xlabel("Hour of month")
        ax.set_ylabel(vunit)
        ax.legend(fontsize=7, loc="best", frameon=True)

        # Compute and annotate difference
        diff = np.nanmean(np.abs(raw_ts.values[:n_show] - ba_ts.values[:n_show]))
        if vname == "airtemp":
            ax.text(0.02, 0.02, f"MAD: {diff:.2f} K",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            ax.text(0.02, 0.02, f"MAD: {diff:.2e}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        if row == 0:
            ax.set_title("(c) Time series comparison\nFirst 2 weeks of Jan 2004", fontsize=10, fontweight="bold")
        else:
            ax.set_title("Time series comparison\nFirst 2 weeks of Jan 2004", fontsize=10)

    # Row labels
    for row, (_, vlabel, _, _) in enumerate(variables):
        axes[row, 0].annotate(
            vlabel, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90,
            ha="center", va="center", color=DOMAIN_COLORS["bow"])

    fig.suptitle("Forcing data transformation: Raw ERA5 → Remapped → Model-ready (Bow at Banff)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = save_path / f"fig_forcing_transformation.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)
    ds_raw.close()
    ds_ba.close()
    if ds_summa is not None:
        ds_summa.close()


# ============================================================
# Figure 2: Spatial Remapping Geometry
# ============================================================
def fig_spatial_remapping(save_path: Path, fmt: str = "png"):
    """
    3-panel figure showing the spatial remapping for Bow at Banff:
    (a) ERA5 grid cells overlaid on HRU polygons
    (b) Remapping weight matrix as a heatmap (actual weights from intersection)
    (c) A forcing variable mapped: raw grid vs remapped HRUs side-by-side
    """
    if not HAS_GEOPANDAS:
        logger.warning("geopandas not available; skipping Figure 2")
        return

    bow_dir = DATA_DIR / "domain_Bow_at_Banff_semi_distributed"
    hru_shp = bow_dir / "shapefiles" / "catchment" / "Bow_at_Banff_semi_distributed_HRUs_GRUs.shp"
    grid_shp = bow_dir / "shapefiles" / "forcing" / "forcing_ERA5.shp"
    intersect_csv = bow_dir / "shapefiles" / "catchment_intersection" / "with_forcing" / "Bow_at_Banff_semi_distributed_ERA5_intersected_shapefile.csv"

    if not hru_shp.exists() or not grid_shp.exists():
        logger.warning("Missing shapefiles for Bow domain; skipping Figure 2")
        return

    hrus = gpd.read_file(hru_shp)
    grid = gpd.read_file(grid_shp)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel (a): ERA5 grid cells overlaid on HRU polygons ---
    ax = axes[0]
    hrus.plot(ax=ax, facecolor=DOMAIN_COLORS["bow"], alpha=0.3,
              edgecolor=DOMAIN_COLORS["bow"], linewidth=0.8)
    grid.plot(ax=ax, facecolor="none", edgecolor="#DD8452", linewidth=1.5, linestyle="--")

    # Label grid cell centroids
    for _, row in grid.iterrows():
        centroid = row.geometry.centroid
        ax.plot(centroid.x, centroid.y, "s", color="#DD8452", markersize=4, alpha=0.7)

    # Label HRU centroids
    for _, row in hrus.iterrows():
        centroid = row.geometry.centroid
        ax.plot(centroid.x, centroid.y, ".", color=DOMAIN_COLORS["bow"], markersize=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("(a) ERA5 grid cells (orange)\noverlaid on 49 HRUs (blue)", fontsize=10, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=DOMAIN_COLORS["bow"], alpha=0.3,
                       edgecolor=DOMAIN_COLORS["bow"], label=f"HRUs (n={len(hrus)})"),
        mpatches.Patch(facecolor="none", edgecolor="#DD8452",
                       linestyle="--", linewidth=1.5, label=f"ERA5 grid (n={len(grid)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=True)

    # --- Panel (b): Remapping weight matrix ---
    ax = axes[1]
    if intersect_csv.exists():
        idf = pd.read_csv(intersect_csv)
        # Build weight matrix from intersection areas
        # AP1N = normalised area of intersection relative to HRU area
        if "S_1_HRU_ID" in idf.columns and "S_2_ID" in idf.columns and "AP1N" in idf.columns:
            hru_ids = sorted(idf["S_1_HRU_ID"].unique())
            grid_ids = sorted(idf["S_2_ID"].unique())
            hru_map = {h: i for i, h in enumerate(hru_ids)}
            grid_map = {g: i for i, g in enumerate(grid_ids)}

            W = np.zeros((len(grid_ids), len(hru_ids)))
            for _, r in idf.iterrows():
                hi = hru_map.get(r["S_1_HRU_ID"])
                gi = grid_map.get(r["S_2_ID"])
                if hi is not None and gi is not None:
                    W[gi, hi] = r["AP1N"]

            im = ax.imshow(W, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                           vmin=0, vmax=1)
            ax.set_xlabel(f"Target HRU index (n={len(hru_ids)})")
            ax.set_ylabel(f"Source ERA5 cell index (n={len(grid_ids)})")
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("Weight (normalised intersection area)", fontsize=8)

            n_nonzero = np.count_nonzero(W)
            n_total = W.size
            sparsity = 1.0 - n_nonzero / n_total
            ax.text(0.02, 0.98,
                    f"Non-zero: {n_nonzero}/{n_total}\nSparsity: {sparsity:.1%}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "Weight columns not found", transform=ax.transAxes,
                    ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No intersection CSV", transform=ax.transAxes,
                ha="center", va="center")

    ax.set_title("(b) Remapping weight matrix\n(EASYMORE intersection weights)", fontsize=10, fontweight="bold")

    # --- Panel (c): Temperature mapped onto HRUs ---
    ax = axes[2]
    raw_dir = bow_dir / "forcing" / "raw_data"
    ba_dir = bow_dir / "forcing" / "basin_averaged_data"
    raw_files = sorted(raw_dir.glob("*.nc"))
    ba_files = sorted(ba_dir.glob("*.nc"))

    if raw_files and ba_files:
        ds_ba = xr.open_dataset(ba_files[0])
        # Pick noon on day 15
        t_idx = min(14 * 24 + 12, len(ds_ba.time) - 1)
        temp_vals = ds_ba["airtemp"].isel(time=t_idx).values

        # Map temperature to HRU polygons
        hrus_plot = hrus.copy()
        if len(temp_vals) == len(hrus_plot):
            hrus_plot["airtemp"] = temp_vals
            hrus_plot.plot(column="airtemp", ax=ax, cmap="RdYlBu_r",
                           edgecolor="white", linewidth=0.5,
                           legend=True,
                           legend_kwds={"label": "Air temperature (K)",
                                        "shrink": 0.8})
            # Overlay grid outlines
            grid.plot(ax=ax, facecolor="none", edgecolor="#DD8452",
                      linewidth=0.8, linestyle=":", alpha=0.5)
            time_str = str(ds_ba.time.values[t_idx])[:16]
            ax.set_title(f"(c) Temperature remapped to HRUs\n{time_str}", fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Shape mismatch: {len(temp_vals)} vs {len(hrus_plot)}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title("(c) Remapped temperature", fontsize=10)
        ds_ba.close()

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle("Spatial remapping: ERA5 grid → HRU polygons (Bow at Banff, 42 cells → 49 HRUs)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = save_path / f"fig_spatial_remapping.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 3: GRACE Remote Sensing over Basin
# ============================================================
def fig_grace_remote_sensing(save_path: Path, fmt: str = "png"):
    """
    3-panel figure:
    (a) GRACE TWS anomaly time series (3 Mascon solutions) over Iceland
    (b) Seasonal decomposition / annual cycle
    (c) Data availability & gap analysis across solutions
    """
    # Try Iceland GRACE first, fall back to other domains
    grace_paths = [
        ("Iceland (Ellidaar)", DATA_DIR / "domain_ellioaar_iceland" / "observations" / "grace" / "preprocessed" / "ellioaar_iceland_grace_tws_processed.csv"),
        ("Gulkana", DATA_DIR / "domain_Gulkana" / "observations" / "grace" / "preprocessed" / "Gulkana_grace_tws_processed.csv"),
    ]

    grace_csv = None
    domain_label = None
    for label, path in grace_paths:
        if path.exists():
            grace_csv = path
            domain_label = label
            break

    if grace_csv is None:
        logger.warning("No GRACE processed data found; skipping Figure 3")
        return

    df = pd.read_csv(grace_csv, parse_dates=True, index_col=0)
    logger.info(f"Loaded GRACE data for {domain_label}: {df.shape}")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # --- Panel (a): Full TWS anomaly time series ---
    ax_a = fig.add_subplot(gs[0, :])

    solution_cols = {
        "grace_jpl_anomaly": ("JPL Mascon", "#1f77b4"),
        "grace_csr_anomaly": ("CSR Mascon", "#ff7f0e"),
        "grace_gsfc_anomaly": ("GSFC Mascon", "#2ca02c"),
    }

    for col, (label, color) in solution_cols.items():
        if col in df.columns:
            ax_a.plot(df.index, df[col], color=color, linewidth=1.2,
                      alpha=0.85, label=label)
            # Highlight data gaps
            gap_mask = df[col].isna()
            if gap_mask.any():
                gap_starts = df.index[gap_mask & ~gap_mask.shift(1, fill_value=False)]
                gap_ends = df.index[gap_mask & ~gap_mask.shift(-1, fill_value=False)]
                for gs_t, ge_t in zip(gap_starts, gap_ends):
                    ax_a.axvspan(gs_t, ge_t, alpha=0.08, color=color, zorder=0)

    # Mark GRACE/GRACE-FO transition
    grace_fo_start = pd.Timestamp("2018-06-01")
    if df.index[-1] > grace_fo_start:
        ax_a.axvline(grace_fo_start, color="#888888", linewidth=1, linestyle="--", alpha=0.6)
        ax_a.text(grace_fo_start, ax_a.get_ylim()[1] * 0.95, " GRACE-FO",
                  fontsize=8, color="#888888", va="top")
        ax_a.axvline(pd.Timestamp("2017-06-01"), color="#888888", linewidth=1,
                     linestyle="--", alpha=0.6)
        ax_a.text(pd.Timestamp("2017-06-01"), ax_a.get_ylim()[1] * 0.95, "Gap ",
                  fontsize=8, color="#888888", va="top", ha="right")

    # Experiment period overlay
    for period_name, dates, pcolor in [
        ("Calibration", ("2004-01-01", "2007-12-31"), "#e6f2ff"),
        ("Evaluation", ("2008-01-01", "2009-12-31"), "#fff2e6"),
    ]:
        s = pd.Timestamp(dates[0])
        e = pd.Timestamp(dates[1])
        if s >= df.index[0]:
            ax_a.axvspan(s, e, alpha=0.15, color=pcolor, label=period_name, zorder=0)

    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("TWS anomaly (mm w.e.)")
    ax_a.set_title(f"(a) GRACE/GRACE-FO terrestrial water storage anomalies — {domain_label}",
                   fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    ax_a.grid(True, alpha=0.2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_a.xaxis.set_major_locator(mdates.YearLocator(2))

    # --- Panel (b): Annual cycle (monthly climatology) ---
    ax_b = fig.add_subplot(gs[1, 0])

    for col, (label, color) in solution_cols.items():
        if col in df.columns:
            monthly = df[col].groupby(df.index.month)
            means = monthly.mean()
            stds = monthly.std()
            months = means.index
            ax_b.plot(months, means.values, "o-", color=color, linewidth=1.5,
                      markersize=5, label=label)
            ax_b.fill_between(months, (means - stds).values, (means + stds).values,
                              color=color, alpha=0.15)

    ax_b.set_xlabel("Month")
    ax_b.set_ylabel("TWS anomaly (mm w.e.)")
    ax_b.set_title("(b) Mean annual cycle (±1σ)", fontsize=11, fontweight="bold")
    ax_b.set_xticks(range(1, 13))
    ax_b.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax_b.legend(fontsize=7, frameon=True)
    ax_b.grid(True, alpha=0.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # --- Panel (c): Data availability heatmap ---
    ax_c = fig.add_subplot(gs[1, 1])

    avail_cols = [c for c in solution_cols.keys() if c in df.columns]
    if avail_cols:
        # Create year × month availability matrix per solution
        years = sorted(df.index.year.unique())
        n_solutions = len(avail_cols)

        # Stacked availability: each solution gets a row band
        avail_matrix = np.full((n_solutions, len(years) * 12), np.nan)
        for si, col in enumerate(avail_cols):
            for yi, year in enumerate(years):
                for month in range(1, 13):
                    idx = yi * 12 + (month - 1)
                    mask = (df.index.year == year) & (df.index.month == month)
                    if mask.any():
                        val = df.loc[mask, col].iloc[0]
                        avail_matrix[si, idx] = 0 if pd.isna(val) else 1

        # Simple bar-style availability
        for si, col in enumerate(avail_cols):
            label, color = solution_cols[col]
            available = avail_matrix[si] == 1
            missing = avail_matrix[si] == 0
            x_avail = np.where(available)[0]
            x_miss = np.where(missing)[0]
            y_pos = si

            if len(x_avail) > 0:
                ax_c.barh([y_pos] * len(x_avail), [1] * len(x_avail),
                          left=x_avail, height=0.7, color=color, alpha=0.7)
            if len(x_miss) > 0:
                ax_c.barh([y_pos] * len(x_miss), [1] * len(x_miss),
                          left=x_miss, height=0.7, color="#cccccc", alpha=0.5)

        ax_c.set_yticks(range(n_solutions))
        ax_c.set_yticklabels([solution_cols[c][0] for c in avail_cols], fontsize=9)

        # X-axis: show years
        year_ticks = [i * 12 for i in range(0, len(years), 3)]
        ax_c.set_xticks(year_ticks)
        ax_c.set_xticklabels([str(years[i]) for i in range(0, len(years), 3)], fontsize=8)

        # Gap statistics
        for si, col in enumerate(avail_cols):
            total = np.sum(~np.isnan(avail_matrix[si]))
            available = np.sum(avail_matrix[si] == 1)
            gap_pct = (1 - available / max(total, 1)) * 100
            label, color = solution_cols[col]
            ax_c.text(len(years) * 12 + 2, si,
                      f"{available}/{int(total)} ({gap_pct:.0f}% gap)",
                      va="center", fontsize=8, color=color)

    ax_c.set_xlabel("Time (months)")
    ax_c.set_title("(c) Data availability by solution", fontsize=11, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.suptitle(f"GRACE remote sensing: terrestrial water storage observations — {domain_label}",
                 fontsize=14, fontweight="bold", y=1.01)
    out = save_path / f"fig_grace_remote_sensing.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Figure 4: Lapse-Rate Effect Across Elevation Bands
# ============================================================
def fig_lapse_rate_effect(save_path: Path, fmt: str = "png"):
    """
    2-panel figure showing the effect of elevation-based lapse rate correction:
    (a) Temperature vs HRU elevation (scatter + regression) at a single timestep,
        showing the raw basin-average vs lapse-corrected values
    (b) Time series for the lowest and highest elevation HRUs showing the
        divergence introduced by lapse-rate correction
    """
    bow_dir = DATA_DIR / "domain_Bow_at_Banff_semi_distributed"
    ba_dir = bow_dir / "forcing" / "basin_averaged_data"
    summa_dir = bow_dir / "forcing" / "SUMMA_input"
    hru_shp = bow_dir / "shapefiles" / "catchment" / "Bow_at_Banff_semi_distributed_HRUs_GRUs.shp"

    ba_files = sorted(ba_dir.glob("*.nc"))
    summa_files = sorted(summa_dir.glob("*.nc"))

    if not ba_files or not HAS_GEOPANDAS:
        logger.warning("Missing data for lapse rate figure; skipping Figure 4")
        return

    hrus = gpd.read_file(hru_shp)
    elevations = hrus["elev_mean"].values

    ds_ba = xr.open_dataset(ba_files[0])
    ds_summa = xr.open_dataset(summa_files[0]) if summa_files else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel (a): Temperature vs elevation at single timestep ---
    ax = axes[0]
    t_idx = min(14 * 24 + 12, len(ds_ba.time) - 1)  # noon day 15

    ba_temp = ds_ba["airtemp"].isel(time=t_idx).values
    summa_temp = ds_summa["airtemp"].isel(time=t_idx).values if ds_summa is not None else None

    ax.scatter(elevations, ba_temp, color="#888888", s=30, alpha=0.7,
               edgecolors="white", linewidth=0.5, label="Before lapse correction", zorder=3)

    if summa_temp is not None:
        ax.scatter(elevations, summa_temp, color=DOMAIN_COLORS["bow"], s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label="After lapse correction", zorder=4)
        # Draw arrows from before to after
        for i in range(len(elevations)):
            if not np.isnan(ba_temp[i]) and not np.isnan(summa_temp[i]):
                ax.annotate("", xy=(elevations[i], summa_temp[i]),
                            xytext=(elevations[i], ba_temp[i]),
                            arrowprops=dict(arrowstyle="->", color=DOMAIN_COLORS["bow"],
                                            alpha=0.3, lw=0.8))

    # Theoretical lapse rate line
    elev_range = np.linspace(elevations.min(), elevations.max(), 100)
    lapse_rate = 0.0065  # K/m
    ref_temp = np.nanmean(ba_temp)
    ref_elev = np.nanmean(elevations)
    theoretical = ref_temp - lapse_rate * (elev_range - ref_elev)
    ax.plot(elev_range, theoretical, "--", color="#d62728", linewidth=1.5,
            alpha=0.6, label="Theoretical (−6.5 K/km)")

    ax.set_xlabel("HRU mean elevation (m)")
    ax.set_ylabel("Air temperature (K)")
    time_str = str(ds_ba.time.values[t_idx])[:16]
    ax.set_title(f"(a) Temperature vs elevation\n{time_str}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (b): Time series for lowest and highest HRUs ---
    ax = axes[1]
    low_idx = np.argmin(elevations)
    high_idx = np.argmax(elevations)
    low_elev = elevations[low_idx]
    high_elev = elevations[high_idx]

    n_show = min(7 * 24, len(ds_ba.time))  # 1 week
    hours = np.arange(n_show)

    ba_low = ds_ba["airtemp"].isel(hru=low_idx).values[:n_show]
    ba_high = ds_ba["airtemp"].isel(hru=high_idx).values[:n_show]

    ax.plot(hours, ba_low, color="#ff7f0e", linewidth=1.0, alpha=0.6,
            label=f"Basin-avg, lowest ({low_elev:.0f} m)")
    ax.plot(hours, ba_high, color="#9467bd", linewidth=1.0, alpha=0.6,
            label=f"Basin-avg, highest ({high_elev:.0f} m)")

    if ds_summa is not None:
        summa_low = ds_summa["airtemp"].isel(hru=low_idx).values[:n_show]
        summa_high = ds_summa["airtemp"].isel(hru=high_idx).values[:n_show]
        ax.plot(hours, summa_low, color="#ff7f0e", linewidth=1.5, linestyle="--",
                label=f"Lapse-corrected, lowest ({low_elev:.0f} m)")
        ax.plot(hours, summa_high, color="#9467bd", linewidth=1.5, linestyle="--",
                label=f"Lapse-corrected, highest ({high_elev:.0f} m)")

        # Annotate elevation difference effect
        elev_diff = high_elev - low_elev
        temp_diff = lapse_rate * elev_diff
        ax.text(0.02, 0.02,
                f"Δelev = {elev_diff:.0f} m → ΔT ≈ {temp_diff:.1f} K",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Hour of month")
    ax.set_ylabel("Air temperature (K)")
    ax.set_title("(b) Lowest vs highest elevation HRU\nFirst week of Jan 2004",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (c): Spatial map of lapse correction magnitude ---
    ax = axes[2]
    if ds_summa is not None and len(ba_temp) == len(hrus):
        correction = summa_temp - ba_temp
        hrus_plot = hrus.copy()
        hrus_plot["correction"] = correction

        # Diverging colormap centered on 0
        vmax = max(abs(np.nanmin(correction)), abs(np.nanmax(correction)))
        hrus_plot.plot(column="correction", ax=ax, cmap="RdBu_r",
                       edgecolor="white", linewidth=0.5,
                       vmin=-vmax, vmax=vmax,
                       legend=True,
                       legend_kwds={"label": "Temperature correction (K)",
                                    "shrink": 0.8})

        # Overlay grid
        grid_shp = bow_dir / "shapefiles" / "forcing" / "forcing_ERA5.shp"
        if grid_shp.exists():
            grid = gpd.read_file(grid_shp)
            grid.plot(ax=ax, facecolor="none", edgecolor="#888888",
                      linewidth=0.5, linestyle=":", alpha=0.4)

        ax.set_title(f"(c) Lapse-rate correction magnitude\n{time_str}",
                     fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No SUMMA data for correction map",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("(c) Lapse-rate correction", fontsize=10)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle("Elevation lapse-rate correction: −6.5 K/km applied per HRU (Bow at Banff)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = save_path / f"fig_lapse_rate_effect.{fmt}"
    fig.savefig(out)
    logger.info(f"Saved: {out}")
    plt.close(fig)
    ds_ba.close()
    if ds_summa is not None:
        ds_summa.close()


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Create data-centric figures for Section 4.12 (actual data through pipeline stages)"
    )
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Creating data-centric pipeline visualisations")
    logger.info("=" * 60)

    logger.info("\n[1/4] Forcing transformation panel ...")
    fig_forcing_transformation(FIGURES_DIR, args.format)

    logger.info("\n[2/4] Spatial remapping geometry ...")
    fig_spatial_remapping(FIGURES_DIR, args.format)

    logger.info("\n[3/4] GRACE remote sensing ...")
    fig_grace_remote_sensing(FIGURES_DIR, args.format)

    logger.info("\n[4/4] Lapse-rate effect ...")
    fig_lapse_rate_effect(FIGURES_DIR, args.format)

    logger.info(f"\nAll data-centric figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
