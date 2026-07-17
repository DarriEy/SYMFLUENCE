#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
25-model ensemble hydrograph for SYMFLUENCE paper Section 4.2.

Generates a four-panel figure with metrics table:
  (a) Full-period hydrograph (2004-2009) with calibration/evaluation split
  (b) Calibration zoom: Apr-Oct 2005
  (c) Evaluation zoom: Apr-Oct 2008
  (d) Flow duration curve (log-scale)

Individual models shown as thin grey lines with ensemble percentile shading
(10-90th and 25-75th).  Ensemble mean (blue) and median (orange dashed) are
highlighted.  A ranked metrics table is placed beside the hydrograph panel.

Imports model specifications and loaders from ensemble_analysis.py
to avoid duplicating the 25-model loading infrastructure.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Reuse the full model infrastructure from ensemble_analysis
from ensemble_analysis import (
    MODEL_SPEC, MODEL_COLORS, LOADERS, METRIC_FILES,
    OBS_FILE, EVAL_START, EVAL_END,
    CALIB_START, CALIB_END,
    PERIOD_START, PERIOD_END,
    ZOOM_CAL_START, ZOOM_CAL_END,
    ZOOM_EVAL_START, ZOOM_EVAL_END,
    KGE_THRESHOLD,
    BASIN_AREA_M2, BASIN_AREA_KM2,
    PARAM_COUNTS,
    load_obs, load_model, load_metrics,
    load_crash_rates, load_calib_runtimes,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Figures go to plotting/output/ next to the other staged paper figures.
FIG_DIR = Path(__file__).resolve().parents[1] / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Models with asterisk: SYMFLUENCE native JAX re-implementations
JAX_MODELS = {"TOPMODEL", "HBV", "SACSMA", "XAJ", "XAJ+Snow17", "HECHMS"}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_kge(sim: pd.Series, obs: pd.Series) -> dict:
    """Gupta et al. (2009) KGE and components from paired daily series."""
    common = sim.dropna().index.intersection(obs.dropna().index)
    if len(common) < 10:
        return {"KGE": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
    s = sim.loc[common].values.astype(float)
    o = obs.loc[common].values.astype(float)
    r = np.corrcoef(s, o)[0, 1]
    alpha = np.std(s, ddof=0) / np.std(o, ddof=0)
    beta = np.mean(s) / np.mean(o)
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"KGE": kge, "r": r, "alpha": alpha, "beta": beta}


def compute_nse(sim: pd.Series, obs: pd.Series) -> float:
    common = sim.dropna().index.intersection(obs.dropna().index)
    if len(common) < 10:
        return np.nan
    s = sim.loc[common].values.astype(float)
    o = obs.loc[common].values.astype(float)
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2)


def compute_pbias(sim: pd.Series, obs: pd.Series) -> float:
    common = sim.dropna().index.intersection(obs.dropna().index)
    if len(common) < 10:
        return np.nan
    s = sim.loc[common].values.astype(float)
    o = obs.loc[common].values.astype(float)
    return 100.0 * np.sum(s - o) / np.sum(o)


def fdc(series):
    """Flow duration curve: returns (exceedance %, values)."""
    vals = np.sort(series.dropna().values)[::-1]
    exc = np.arange(1, len(vals) + 1) / len(vals) * 100
    return exc, vals


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 7,
        "figure.dpi": 300,
    })

    # ── Load observed ─────────────────────────────────────────────────
    print("Loading observations...")
    obs_full = load_obs(OBS_FILE)
    obs = obs_full.loc[PERIOD_START:PERIOD_END].dropna()
    if obs.empty:
        print("ERROR: No observation data found.")
        return

    # ── Load all models ───────────────────────────────────────────────
    print("Loading model simulations...")
    simulations = {}
    for name, spec in MODEL_SPEC.items():
        try:
            s = load_model(spec)
            if s is not None and len(s) > 30:
                simulations[name] = s
                print(f"  [{name}] OK ({len(s)} days)")
        except Exception as e:
            print(f"  [{name}] SKIP: {e}")

    # ── Compute metrics (calibration + evaluation) ────────────────────
    print("\nComputing metrics...")
    metrics = {}
    for name, sim in simulations.items():
        cal_mask = (sim.index >= CALIB_START) & (sim.index <= CALIB_END)
        eval_mask = (sim.index >= EVAL_START) & (sim.index <= EVAL_END)
        obs_cal = obs.reindex(sim.index[cal_mask]).dropna()
        obs_eval = obs.reindex(sim.index[eval_mask]).dropna()
        sim_cal = sim.reindex(obs_cal.index).dropna()
        sim_eval = sim.reindex(obs_eval.index).dropna()

        cal_kge = compute_kge(sim_cal, obs_cal)
        eval_kge = compute_kge(sim_eval, obs_eval)
        cal_nse = compute_nse(sim_cal, obs_cal)
        eval_nse = compute_nse(sim_eval, obs_eval)
        cal_pbias = compute_pbias(sim_cal, obs_cal)
        eval_pbias = compute_pbias(sim_eval, obs_eval)

        metrics[name] = {
            "Cal_KGE": cal_kge["KGE"], "Cal_r": cal_kge["r"],
            "Cal_alpha": cal_kge["alpha"], "Cal_beta": cal_kge["beta"],
            "Cal_NSE": cal_nse, "Cal_PBIAS": cal_pbias,
            "Eval_KGE": eval_kge["KGE"], "Eval_r": eval_kge["r"],
            "Eval_alpha": eval_kge["alpha"], "Eval_beta": eval_kge["beta"],
            "Eval_NSE": eval_nse, "Eval_PBIAS": eval_pbias,
        }
        print(f"  {name:20s}: Cal KGE = {cal_kge['KGE']:.3f}, "
              f"Eval KGE = {eval_kge['KGE']:.3f}")

    # ── Filter by calibration KGE ─────────────────────────────────────
    print(f"\nFiltering models (Cal KGE > {KGE_THRESHOLD})...")
    included = {}
    for name, sim in simulations.items():
        cal_kge = metrics[name]["Cal_KGE"]
        if not np.isnan(cal_kge) and cal_kge > KGE_THRESHOLD:
            included[name] = sim
            print(f"  INCLUDE: {name}")
        else:
            print(f"  EXCLUDE: {name} (Cal KGE = {cal_kge:.3f})")

    n_included = len(included)
    print(f"\n{n_included} models included in ensemble")

    if not included:
        print("WARNING: No models passed filter. Nothing to plot.")
        return

    # ── Align to common index ─────────────────────────────────────────
    common_idx = obs.index
    for s in included.values():
        common_idx = common_idx.intersection(s.index)

    obs_aligned = obs.loc[common_idx]
    sim_aligned = {k: v.loc[common_idx] for k, v in included.items()}

    # ── Build ensemble statistics ─────────────────────────────────────
    ens_df = pd.DataFrame(sim_aligned, index=common_idx)
    ens_mean = ens_df.mean(axis=1)
    ens_median = ens_df.median(axis=1)
    ens_q10 = ens_df.quantile(0.10, axis=1)
    ens_q25 = ens_df.quantile(0.25, axis=1)
    ens_q75 = ens_df.quantile(0.75, axis=1)
    ens_q90 = ens_df.quantile(0.90, axis=1)

    # Ensemble summary metrics
    cal_mask = (ens_mean.index >= CALIB_START) & (ens_mean.index <= CALIB_END)
    eval_mask = (ens_mean.index >= EVAL_START) & (ens_mean.index <= EVAL_END)
    ens_cal_kge = compute_kge(
        ens_mean.loc[cal_mask], obs_aligned.loc[cal_mask])["KGE"]
    ens_eval_kge = compute_kge(
        ens_mean.loc[eval_mask], obs_aligned.loc[eval_mask])["KGE"]
    ens_med_cal_kge = compute_kge(
        ens_median.loc[cal_mask], obs_aligned.loc[cal_mask])["KGE"]
    ens_med_eval_kge = compute_kge(
        ens_median.loc[eval_mask], obs_aligned.loc[eval_mask])["KGE"]

    print(f"\nEnsemble mean  KGE: Cal = {ens_cal_kge:.3f}, "
          f"Eval = {ens_eval_kge:.3f}")
    print(f"Ensemble median KGE: Cal = {ens_med_cal_kge:.3f}, "
          f"Eval = {ens_med_eval_kge:.3f}")

    # ── Helper: plot ensemble on an axis for a date range ─────────────
    def plot_ensemble_panel(ax, idx):
        ax.fill_between(idx, ens_q10.loc[idx], ens_q90.loc[idx],
                         color="#d1e5f0", alpha=0.7, zorder=1, edgecolor="none")
        ax.fill_between(idx, ens_q25.loc[idx], ens_q75.loc[idx],
                         color="#92c5de", alpha=0.7, zorder=1.5, edgecolor="none")
        for name in sim_aligned:
            ax.plot(idx, sim_aligned[name].loc[idx], color="#aaaaaa",
                    linewidth=0.4, alpha=0.5, zorder=2)
        ax.plot(idx, ens_median.loc[idx], color="#e66101", linewidth=1.2,
                linestyle="--", zorder=3.5)
        ax.plot(idx, ens_mean.loc[idx], color="#1f78b4", linewidth=1.5, zorder=3)
        ax.plot(idx, obs_aligned.loc[idx], color="black", linewidth=1.3, zorder=4)

    # ══════════════════════════════════════════════════════════════════
    # FIGURE — matches ensemble_analysis.py Figure A layout exactly
    # ══════════════════════════════════════════════════════════════════
    print("\nGenerating figure...")

    fig = plt.figure(figsize=(24, 13))

    # Top-left: hydrograph
    ax_full = fig.add_axes([0.04, 0.40, 0.42, 0.56])
    # Top-right: table
    ax_tab = fig.add_axes([0.48, 0.40, 0.51, 0.56])
    # Bottom row: three panels
    ax_cal  = fig.add_axes([0.05, 0.05, 0.27, 0.28])
    ax_eval = fig.add_axes([0.38, 0.05, 0.27, 0.28])
    ax_fdc  = fig.add_axes([0.71, 0.05, 0.27, 0.28])

    # ── Panel (a): Full period hydrograph ─────────────────────────────
    plot_mask = ((obs_aligned.index >= PERIOD_START) &
                 (obs_aligned.index <= PERIOD_END))
    plot_idx = obs_aligned.index[plot_mask]
    plot_ensemble_panel(ax_full, plot_idx)

    # Background shading for calibration vs evaluation periods
    ax_full.axvspan(pd.Timestamp(PERIOD_START), pd.Timestamp(CALIB_END),
                    color="#e8f4fd", alpha=0.35, zorder=0)
    ax_full.axvspan(pd.Timestamp(EVAL_START), pd.Timestamp(PERIOD_END),
                    color="#fde8e8", alpha=0.35, zorder=0)
    ax_full.axvline(pd.Timestamp(EVAL_START), color="#555555", linestyle="--",
                    linewidth=1.2, zorder=5)
    ax_full.text(pd.Timestamp("2005-12-01"), 305, "Calibration (2004\u20132007)",
                 fontsize=10, color="#2166ac", ha="center", fontweight="semibold")
    ax_full.text(pd.Timestamp("2009-01-01"), 305, "Evaluation (2008\u20132009)",
                 fontsize=10, color="#b2182b", ha="center", fontweight="semibold")
    ax_full.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_full.set_title(
        "(a) Multi-model ensemble hydrograph \u2014 "
        "Bow River at Banff (2004\u20132009)",
        fontsize=13, fontweight="semibold", pad=8,
    )
    ax_full.xaxis.set_major_locator(mdates.YearLocator())
    ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_full.set_xlim(pd.Timestamp(PERIOD_START), pd.Timestamp(PERIOD_END))
    ax_full.set_ylim(bottom=0, top=320)

    # Legend underneath hydrograph
    leg_handles = [
        Line2D([], [], color="black", lw=1.3, label="Observed"),
        Line2D([], [], color="#1f78b4", lw=1.5, label="Ens. mean"),
        Line2D([], [], color="#e66101", lw=1.2, ls="--", label="Ens. median"),
        Patch(facecolor="#92c5de", alpha=0.7, label="IQR (25\u201375th)"),
        Patch(facecolor="#d1e5f0", alpha=0.7, label="10\u201390th pctl"),
        Line2D([], [], color="#aaaaaa", lw=0.6, alpha=0.6,
               label="Individual models"),
    ]
    ax_full.legend(handles=leg_handles, loc="lower center", fontsize=9,
                   framealpha=0.95, ncol=6, handlelength=1.5,
                   columnspacing=1.0,
                   bbox_to_anchor=(0.5, -0.08), borderaxespad=0)

    # ── Table panel (b): model metrics ranked by eval KGE ─────────────
    ax_tab.set_axis_off()
    ax_tab.set_title("(b) Model performance summary (ranked by evaluation KGE)",
                      fontsize=13, fontweight="semibold", pad=8, loc="left")
    ranked = sorted(
        [n for n in metrics if n in included],
        key=lambda m: -metrics[m].get("Eval_KGE", -999),
    )

    # Load crash rates and calibration runtimes
    crash_rates = load_crash_rates()
    calib_runtimes = load_calib_runtimes()

    SHORT_NAMES = {
        "SUMMA+MODFLOW": "SUMMA+MOD",
        "ParFlow+Snow17": "PF+Snow17",
        "XAJ+Snow17": "XAJ+S17*",
        "CLM+ParFlow": "CLM+PF",
    }

    def _fmt_rt(hrs):
        if hrs is None:
            return "    --"
        if hrs < 0.05:
            return f"{hrs*3600:5.0f}s"
        if hrs < 1.0:
            return f"{hrs*60:5.1f}m"
        return f"{hrs:5.1f}h"

    def _fmt_np(count):
        if count is None:
            return "   --"
        if count >= 1000:
            return f"{count/1000:4.0f}k"
        return f"{count:5d}"

    # Styled performance table (paper Fig 7b): dark header row, alternating row
    # shading, highlighted ensemble-summary rows. Columns:
    #   #, Model, nP, Cal KGE, Eval KGE, r, alpha, beta, NSE, PB%, FI%, Time
    from matplotlib.patches import Rectangle

    HEADER_BG = "#2f3b52"
    ROW_ALT = "#eef2f7"
    ENS_BG = "#dbe5f0"

    # (label, x position in axes fraction, horizontal alignment)
    columns = [
        ("#",       0.028, "center"),
        ("Model",   0.065, "left"),
        ("nP",      0.360, "right"),
        ("Cal\nKGE", 0.440, "right"),
        ("Eval\nKGE", 0.530, "right"),
        ("r",       0.595, "right"),
        (chr(945),  0.655, "right"),   # alpha
        (chr(946),  0.715, "right"),   # beta
        ("NSE",     0.785, "right"),
        ("PB%",     0.860, "right"),
        ("FI%",     0.925, "right"),
        ("Time",    0.998, "right"),
    ]

    body = []
    for i, name in enumerate(ranked, 1):
        m = metrics[name]
        raw_disp = SHORT_NAMES.get(name, name)
        disp = raw_disp + "*" if name in JAX_MODELS and "*" not in raw_disp else raw_disp
        cr = crash_rates.get(name)
        body.append([
            str(i), disp, _fmt_np(PARAM_COUNTS.get(name)).strip(),
            f"{m['Cal_KGE']:.2f}", f"{m['Eval_KGE']:.2f}", f"{m['Eval_r']:.2f}",
            f"{m['Eval_alpha']:.2f}", f"{m['Eval_beta']:.2f}", f"{m['Eval_NSE']:.2f}",
            f"{m['Eval_PBIAS']:.1f}",
            (f"{cr:.0f}" if cr is not None else "\u2013"),
            _fmt_rt(calib_runtimes.get(name)).strip(),
        ])
    n_models = len(body)
    body.append(["", "Ens. mean", "", f"{ens_cal_kge:.2f}", f"{ens_eval_kge:.2f}",
                 "", "", "", "", "", "", ""])
    body.append(["", "Ens. median", "", f"{ens_med_cal_kge:.2f}",
                 f"{ens_med_eval_kge:.2f}", "", "", "", "", "", "", ""])
    n_rows = len(body)

    header_top, header_bot = 1.0, 0.925
    body_top = header_bot
    row_h = body_top / n_rows

    # Header band
    ax_tab.add_patch(Rectangle((0, header_bot), 1.0, header_top - header_bot,
                               transform=ax_tab.transAxes, facecolor=HEADER_BG,
                               edgecolor="none", zorder=1, clip_on=False))
    for label, xc, align in columns:
        ax_tab.text(xc, (header_top + header_bot) / 2, label,
                    transform=ax_tab.transAxes, ha=align, va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=3,
                    linespacing=0.9)

    for r, rowdata in enumerate(body):
        y_hi = body_top - r * row_h
        y_ctr = y_hi - row_h / 2
        is_ens = r >= n_models
        bg = ENS_BG if is_ens else (ROW_ALT if r % 2 == 1 else None)
        if bg is not None:
            ax_tab.add_patch(Rectangle((0, y_hi - row_h), 1.0, row_h,
                             transform=ax_tab.transAxes, facecolor=bg,
                             edgecolor="none", zorder=0, clip_on=False))
        fw = "bold" if is_ens else "normal"
        for (label, xc, align), val in zip(columns, rowdata):
            ax_tab.text(xc, y_ctr, val, transform=ax_tab.transAxes,
                        ha=align, va="center", fontsize=9.5, fontweight=fw,
                        color="#222222", zorder=3)

    ax_tab.text(0.0, -0.015,
                "* JAX re-implementation    nP: # calibrated params    "
                "PB%: percent bias    FI%: DDS crash rate    Time: DDS calibration wall-clock",
                transform=ax_tab.transAxes, fontsize=8.5, color="#666666",
                va="top", ha="left")

    # ── Panel (c): Calibration zoom ───────────────────────────────────
    cal_zoom = ((obs_aligned.index >= ZOOM_CAL_START) &
                (obs_aligned.index <= ZOOM_CAL_END))
    cal_idx = obs_aligned.index[cal_zoom]
    plot_ensemble_panel(ax_cal, cal_idx)
    ax_cal.set_ylabel("Streamflow (m$^3$ s$^{-1}$)")
    ax_cal.set_title("(c) Calibration: Apr\u2013Oct 2005", fontsize=12,
                      fontweight="semibold", pad=6)
    ax_cal.xaxis.set_major_locator(mdates.MonthLocator())
    ax_cal.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_cal.set_xlim(pd.Timestamp(ZOOM_CAL_START),
                    pd.Timestamp(ZOOM_CAL_END))

    # ── Panel (d): Evaluation zoom ────────────────────────────────────
    eval_zoom = ((obs_aligned.index >= ZOOM_EVAL_START) &
                 (obs_aligned.index <= ZOOM_EVAL_END))
    eval_idx = obs_aligned.index[eval_zoom]
    plot_ensemble_panel(ax_eval, eval_idx)
    ax_eval.set_ylabel("")
    ax_eval.set_title("(d) Evaluation: Apr\u2013Oct 2008", fontsize=12,
                       fontweight="semibold", pad=6)
    ax_eval.xaxis.set_major_locator(mdates.MonthLocator())
    ax_eval.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_eval.set_xlim(pd.Timestamp(ZOOM_EVAL_START),
                     pd.Timestamp(ZOOM_EVAL_END))

    # ── Panel (e): Flow Duration Curve ────────────────────────────────
    exc_obs, val_obs = fdc(obs_aligned)
    ax_fdc.plot(exc_obs, val_obs, color="black", lw=1.5,
                label="Observed", zorder=4)
    exc_mean, val_mean = fdc(ens_mean)
    ax_fdc.plot(exc_mean, val_mean, color="#1f78b4", lw=1.5,
                label="Ens. mean", zorder=3)
    exc_med, val_med = fdc(ens_median)
    ax_fdc.plot(exc_med, val_med, color="#e66101", lw=1.2, ls="--",
                label="Ens. median", zorder=3.5)

    _, v_q10 = fdc(ens_q10)
    _, v_q25 = fdc(ens_q25)
    _, v_q75 = fdc(ens_q75)
    _, v_q90 = fdc(ens_q90)
    ax_fdc.fill_between(exc_mean, v_q10, v_q90, color="#d1e5f0", alpha=0.7,
                         zorder=1, edgecolor="none")
    ax_fdc.fill_between(exc_mean, v_q25, v_q75, color="#92c5de", alpha=0.7,
                         zorder=1.5, edgecolor="none")
    for name in sim_aligned:
        exc_m, val_m = fdc(sim_aligned[name])
        ax_fdc.plot(exc_m, val_m, color="#aaaaaa", lw=0.4, alpha=0.5, zorder=2)

    ax_fdc.set_xlabel("Exceedance probability (%)")
    ax_fdc.set_ylabel("")
    ax_fdc.set_title("(e) Flow duration curve", fontsize=12,
                      fontweight="semibold", pad=6)
    ax_fdc.set_yscale("log")
    ax_fdc.set_xlim(0, 100)
    ax_fdc.set_ylim(bottom=1)
    ax_fdc.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = FIG_DIR / "figure_07_model_ensemble.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
