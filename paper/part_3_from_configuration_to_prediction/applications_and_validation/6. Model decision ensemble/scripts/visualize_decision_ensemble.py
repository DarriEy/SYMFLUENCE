#!/usr/bin/env python3
"""
Decision Ensemble Visualisation — SYMFLUENCE Paper Section 4.6

Publication-quality figures for the 64-member FUSE structural decision
ensemble.  Designed to match the visual style of Section 4.2 (model
ensemble) and to present the analysis concisely in 3 focused figures:

  Fig 1  (2-panel):  (a) KGE distribution  (b) All structures ranked
  Fig 2  (main):     Decision sensitivity — ordered boxplots with sig. stars
  Fig 3  (2-panel):  (a) ANOVA variance decomposition  (b) Interaction heatmap

Three figures only — no summary panel.

Usage:
    python visualize_decision_ensemble.py [--results-csv PATH]
                                          [--analysis-dir DIR]
                                          [--output-dir DIR]
                                          [--format png|pdf|svg]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
ANALYSIS_DIR = BASE_DIR / "analysis"

SYMFLUENCE_DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DEFAULT_RESULTS_CSV = (
    SYMFLUENCE_DATA_DIR / "domain_Bow_at_Banff_lumped_era5" / "optimization"
    / "decision_ensemble_fuse_decisions_comparison.csv"
)

# ── constants ──────────────────────────────────────────────────────
VARIED_DECISIONS = ["ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF"]

DECISION_LABELS = {
    "ARCH1": "Upper-layer\narchitecture",
    "ARCH2": "Lower-layer\narchitecture",
    "QSURF": "Surface\nrunoff",
    "QPERC": "Percolation",
    "ESOIL": "Evaporation",
    "QINTF": "Interflow",
}

OPTION_LABELS = {
    "tension1_1": "Tension\n(2-state)",
    "onestate_1": "Single\nbucket",
    "tens2pll_2": "Tension\nparallel",
    "unlimfrc_2": "Unlimited\nfraction",
    "arno_x_vic": "VIC",
    "prms_varnt": "PRMS",
    "perc_f2sat": "Frac-to-sat",
    "perc_lower": "Lower zone",
    "sequential": "Sequential",
    "rootweight": "Root\nweighting",
    "intflwnone": "None",
    "intflwsome": "Active",
}

DECISION_LABELS_SHORT = {
    "ARCH1": "ARCH1",
    "ARCH2": "ARCH2",
    "QSURF": "QSURF",
    "QPERC": "QPERC",
    "ESOIL": "ESOIL",
    "QINTF": "QINTF",
}

# ── style ──────────────────────────────────────────────────────────
C_BLUE = "#2c7bb6"
C_RED = "#d7191c"
C_GREEN = "#1a9641"
C_GREY = "#808080"
C_FILL = "#abd9e9"
C_BG = "#f7f7f7"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("decision_viz")

# Match Section 4.2 style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ══════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════
def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for m in ["kge", "kgep", "nse", "mae", "rmse"]:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.dropna(subset=["kge"], how="all")


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ══════════════════════════════════════════════════════════════════
# Figure 1: Performance overview (2-panel)
# ══════════════════════════════════════════════════════════════════
def fig_performance_overview(df: pd.DataFrame, path: Path) -> None:
    """(a) KGE violin + strip  (b) all 64 structures ranked by KGE."""
    fig, (ax_v, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2),
                                      gridspec_kw={"width_ratios": [1, 2.2]})

    kge = df["kge"].dropna().values

    # ── panel (a): violin ──
    parts = ax_v.violinplot(kge, positions=[0], showmeans=False,
                            showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_FILL)
        pc.set_edgecolor(C_BLUE)
        pc.set_alpha(0.65)

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.025, len(kge))
    ax_v.scatter(jitter, kge, s=14, alpha=0.55, color=C_BLUE, edgecolors="none", zorder=5)

    med = np.median(kge)
    ax_v.axhline(med, color="k", lw=1, ls="--", alpha=0.5)
    ax_v.text(0.38, med, f"median = {med:.2f}", fontsize=8, va="bottom")

    q25, q75 = np.percentile(kge, [25, 75])
    ax_v.axhspan(q25, q75, color=C_FILL, alpha=0.25, zorder=0)

    ax_v.set_xticks([])
    ax_v.set_ylabel("KGE")
    ax_v.set_title("(a) KGE distribution")
    ax_v.grid(axis="y", alpha=0.25, lw=0.5)

    # ── panel (b): ranked bars ──
    sorted_kge = np.sort(kge)
    norm = plt.Normalize(sorted_kge.min(), sorted_kge.max())
    cmap = plt.cm.RdYlGn
    colours = cmap(norm(sorted_kge))

    ax_b.bar(range(len(sorted_kge)), sorted_kge, color=colours, width=1.0, edgecolor="none")
    ax_b.axhline(0, color="k", lw=0.6)
    ax_b.axhline(np.mean(kge), color="k", lw=0.8, ls="--", alpha=0.5)
    ax_b.text(len(kge) - 1, np.mean(kge) + 0.02, f"mean = {np.mean(kge):.2f}",
              fontsize=8, ha="right", va="bottom")

    ax_b.set_xlabel("Structure (sorted by KGE)")
    ax_b.set_ylabel("KGE")
    ax_b.set_title(f"(b) All {len(kge)} structures ranked")
    ax_b.grid(axis="y", alpha=0.25, lw=0.5)

    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Decision sensitivity (main figure)
# ══════════════════════════════════════════════════════════════════
def fig_decision_sensitivity(df: pd.DataFrame, sensitivity_df: pd.DataFrame,
                             path: Path) -> None:
    """
    Paired boxplots per decision, ordered by |Δ KGE| (most sensitive first).
    Significance stars from Welch t-test; Δ and η² annotations.
    """
    # Order decisions by sensitivity
    if sensitivity_df.empty:
        order = [d for d in VARIED_DECISIONS if d in df.columns]
    else:
        order = sensitivity_df["decision"].tolist()

    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, dec in zip(axes, order):
        opts = sorted(df[dec].unique())
        data = [df.loc[df[dec] == o, "kge"].dropna().values for o in opts]
        labels = [OPTION_LABELS.get(o, o) for o in opts]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color="k", lw=1.5),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        flierprops=dict(ms=4, alpha=0.4))

        colours = [C_BLUE, C_RED]
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.50)

        # Strip (jittered points)
        for i, d in enumerate(data):
            x = rng_jitter(i + 1, len(d))
            ax.scatter(x, d, s=12, alpha=0.4, color="k", edgecolors="none", zorder=5)

        # Annotations
        if not sensitivity_df.empty:
            row = sensitivity_df[sensitivity_df["decision"] == dec]
            if not row.empty:
                r = row.iloc[0]
                delta = r["abs_delta"]
                eta = r["eta_squared"]
                p = r["p_value"]
                sig = _sig_label(p)

                # Sig stars at top
                ymax = ax.get_ylim()[1]
                ax.text(1.5, ymax - 0.05 * (ymax - ax.get_ylim()[0]),
                        sig, ha="center", va="top", fontsize=11, fontweight="bold",
                        color=C_RED if p < 0.05 else C_GREY)

                # Delta + eta at bottom
                ax.text(1.5, ax.get_ylim()[0] + 0.03 * (ymax - ax.get_ylim()[0]),
                        f"|Δ|={delta:.2f}\nη²={eta:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color="#444444")

        ax.set_title(DECISION_LABELS.get(dec, dec), fontsize=9.5, pad=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    axes[0].set_ylabel("KGE")
    fig.suptitle("Marginal KGE sensitivity per structural decision", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def rng_jitter(centre: float, n: int, spread: float = 0.06) -> np.ndarray:
    return np.random.default_rng(42).normal(centre, spread, n)


# ══════════════════════════════════════════════════════════════════
# Figure 3: Variance decomposition + interaction heatmap (2-panel)
# ══════════════════════════════════════════════════════════════════
def fig_variance_and_interactions(anova_df: pd.DataFrame, interaction_df: pd.DataFrame,
                                  path: Path) -> None:
    """
    (a) Horizontal bar chart of variance explained (main + top interactions).
    (b) 6×6 interaction heatmap.
    """
    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(12, 5),
                                      gridspec_kw={"width_ratios": [1, 1]})

    # ── panel (a): variance decomposition bar chart ──
    if not anova_df.empty:
        # Show main effects + top-5 interactions + residual
        main = anova_df[anova_df["type"] == "main"].copy()
        ints = anova_df[anova_df["type"] == "interaction"].nlargest(5, "pct_variance").copy()
        resid = anova_df[anova_df["type"] == "residual"].copy()

        # Aggregate remaining interactions into "Other interactions"
        all_int_pct = anova_df.loc[anova_df["type"] == "interaction", "pct_variance"].sum()
        top5_int_pct = ints["pct_variance"].sum()
        other_int_pct = all_int_pct - top5_int_pct

        show = pd.concat([main, ints], ignore_index=True)
        if other_int_pct > 0.1:
            show = pd.concat([show, pd.DataFrame([{
                "source_name": "Other interactions", "pct_variance": other_int_pct,
                "type": "interaction", "p_value": np.nan
            }])], ignore_index=True)
        show = pd.concat([show, resid], ignore_index=True)

        # Reverse for horizontal bar (top → bottom = most → least)
        show = show.iloc[::-1].reset_index(drop=True)

        colors = []
        for _, r in show.iterrows():
            if r["type"] == "main":
                colors.append(C_BLUE)
            elif r["type"] == "interaction":
                colors.append(C_RED if r.get("pct_variance", 0) > 2 else "#e8a0a0")
            else:
                colors.append(C_GREY)

        bars = ax_v.barh(range(len(show)), show["pct_variance"].values,
                         color=colors, alpha=0.7, edgecolor="none")

        ax_v.set_yticks(range(len(show)))
        ylabels = []
        for _, r in show.iterrows():
            name = r["source_name"]
            # Shorten long interaction names
            if " × " in str(name):
                parts = str(name).split(" × ")
                name = " × ".join(p.split()[-1] if len(p.split()) > 1 else p for p in parts)
            ylabels.append(name)
        ax_v.set_yticklabels(ylabels, fontsize=8.5)
        ax_v.set_xlabel("% of total KGE variance")
        ax_v.set_title("(a) Variance decomposition")

        # Value labels
        for bar, val in zip(bars, show["pct_variance"].values):
            if val > 0.5:
                ax_v.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=8)

        # Significance markers on main effects
        for i, (_, r) in enumerate(show.iterrows()):
            p = r.get("p_value", np.nan)
            if not np.isnan(p) and p < 0.05:
                idx = len(show) - 1 - i  # reversed
                ax_v.text(r["pct_variance"] + 3.5, i,
                          _sig_label(p), va="center", fontsize=8, color=C_RED)

        ax_v.grid(axis="x", alpha=0.25, lw=0.5)

    # ── panel (b): interaction heatmap ──
    decs = [d for d in VARIED_DECISIONS]
    n_dec = len(decs)
    matrix = np.full((n_dec, n_dec), np.nan)

    if not interaction_df.empty:
        for _, r in interaction_df.iterrows():
            d1, d2 = r["decision_1"], r["decision_2"]
            if d1 in decs and d2 in decs:
                i, j = decs.index(d1), decs.index(d2)
                matrix[i, j] = r["abs_interaction"]
                matrix[j, i] = r["abs_interaction"]

    # Fill diagonal with main-effect |Δ| if available
    sens_path = ANALYSIS_DIR / "decision_sensitivity.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        for _, r in sens.iterrows():
            d = r["decision"]
            if d in decs:
                matrix[decs.index(d), decs.index(d)] = r["abs_delta"]

    mask = np.isnan(matrix)
    masked = np.ma.masked_where(mask, matrix)
    vmax = np.nanmax(matrix) if not np.all(mask) else 1.0

    im = ax_h.imshow(masked, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    ax_h.set_xticks(range(n_dec))
    ax_h.set_yticks(range(n_dec))
    labels = [DECISION_LABELS_SHORT.get(d, d) for d in decs]
    ax_h.set_xticklabels(labels, fontsize=9)
    ax_h.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(n_dec):
        for j in range(n_dec):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                colour = "white" if val > vmax * 0.6 else "black"
                label = f"{val:.2f}"
                if i == j:
                    label = f"|Δ|={val:.2f}"
                ax_h.text(j, i, label, ha="center", va="center",
                          fontsize=7.5, color=colour, fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax_h, shrink=0.8, pad=0.02)
    cbar.set_label("|Δ KGE| or |interaction|", fontsize=9)
    ax_h.set_title("(b) Interaction matrix")

    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# Figure 4: Summary panel (compact 4-panel for paper)
# ══════════════════════════════════════════════════════════════════
def fig_summary_panel(df: pd.DataFrame, sensitivity_df: pd.DataFrame,
                      anova_df: pd.DataFrame, path: Path) -> None:
    """
    4-panel summary: (a) KGE dist  (b) sensitivity bars  (c) ranked  (d) variance pie
    """
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    kge = df["kge"].dropna().values

    # ── (a) KGE distribution ──
    ax_a = fig.add_subplot(gs[0, 0])
    parts = ax_a.violinplot(kge, positions=[0], showmeans=False, showmedians=False,
                            showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_FILL)
        pc.set_edgecolor(C_BLUE)
        pc.set_alpha(0.6)
    jit = np.random.default_rng(42).normal(0, 0.025, len(kge))
    ax_a.scatter(jit, kge, s=12, alpha=0.5, color=C_BLUE, edgecolors="none", zorder=5)
    ax_a.axhline(np.median(kge), color="k", lw=0.8, ls="--", alpha=0.5)
    ax_a.set_xticks([])
    ax_a.set_ylabel("KGE")
    ax_a.set_title(f"(a) KGE distribution (n={len(kge)})")
    ax_a.grid(axis="y", alpha=0.25, lw=0.5)

    # ── (b) sensitivity bars ──
    ax_b = fig.add_subplot(gs[0, 1])
    if not sensitivity_df.empty:
        sens = sensitivity_df.sort_values("abs_delta", ascending=True)
        y_pos = range(len(sens))
        bars = ax_b.barh(list(y_pos), sens["abs_delta"].values, color=C_BLUE, alpha=0.7)
        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels([DECISION_LABELS.get(d, d).replace("\n", " ")
                               for d in sens["decision"]], fontsize=9)
        ax_b.set_xlabel("|Δ KGE|")
        ax_b.set_title("(b) Decision sensitivity")
        for bar, (_, r) in zip(bars, sens.iterrows()):
            sig = _sig_label(r["p_value"])
            ax_b.text(r["abs_delta"] + 0.005, bar.get_y() + bar.get_height() / 2,
                      f'{r["abs_delta"]:.3f} {sig}', va="center", fontsize=8)
    ax_b.grid(axis="x", alpha=0.25, lw=0.5)

    # ── (c) ranked structures ──
    ax_c = fig.add_subplot(gs[1, 0])
    sorted_kge = np.sort(kge)
    norm = plt.Normalize(sorted_kge.min(), sorted_kge.max())
    colours = plt.cm.RdYlGn(norm(sorted_kge))
    ax_c.bar(range(len(sorted_kge)), sorted_kge, color=colours, width=1.0, edgecolor="none")
    ax_c.axhline(0, color="k", lw=0.6)
    ax_c.axhline(np.mean(kge), color="k", lw=0.8, ls="--", alpha=0.5)
    ax_c.set_xlabel("Structure (sorted)")
    ax_c.set_ylabel("KGE")
    ax_c.set_title(f"(c) {len(kge)} structures ranked by KGE")
    ax_c.grid(axis="y", alpha=0.25, lw=0.5)

    # ── (d) variance pie ──
    ax_d = fig.add_subplot(gs[1, 1])
    if not anova_df.empty:
        main_pct = anova_df.loc[anova_df["type"] == "main", "pct_variance"].sum()
        int_pct = anova_df.loc[anova_df["type"] == "interaction", "pct_variance"].sum()
        res_pct = anova_df.loc[anova_df["type"] == "residual", "pct_variance"].sum()

        # Individual main effects for the pie
        main_rows = anova_df[anova_df["type"] == "main"].sort_values("pct_variance", ascending=False)
        labels_pie = [DECISION_LABELS_SHORT.get(r["source"], r["source"]) for _, r in main_rows.iterrows()]
        sizes_pie = main_rows["pct_variance"].values.tolist()
        colors_pie = [plt.cm.Blues(0.3 + 0.6 * i / max(len(labels_pie) - 1, 1)) for i in range(len(labels_pie))]

        # Add interactions and residual
        labels_pie.extend(["Interactions", "Residual"])
        sizes_pie.extend([int_pct, res_pct])
        colors_pie.extend([C_RED, C_GREY])

        wedges, texts, autotexts = ax_d.pie(
            sizes_pie, labels=labels_pie, autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            colors=colors_pie, startangle=90, pctdistance=0.78,
            textprops={"fontsize": 8.5}
        )
        for at in autotexts:
            at.set_fontsize(7.5)
        ax_d.set_title("(d) KGE variance decomposition")
    else:
        ax_d.text(0.5, 0.5, "ANOVA not available", ha="center", va="center", transform=ax_d.transAxes)
        ax_d.set_title("(d) Variance decomposition")

    fig.suptitle("FUSE structural decision ensemble — Bow at Banff (ERA5)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════
def run_visualisation(results_csv: Path, analysis_dir: Path,
                      output_dir: Path, fmt: str = "pdf") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_csv(results_csv)
    logger.info(f"Loaded {len(df)} combinations")

    # Load analysis outputs
    sens_path = analysis_dir / "decision_sensitivity.csv"
    sens_df = pd.read_csv(sens_path) if sens_path.exists() else pd.DataFrame()

    anova_path = analysis_dir / "variance_decomposition.csv"
    anova_df = pd.read_csv(anova_path) if anova_path.exists() else pd.DataFrame()

    int_path = analysis_dir / "interaction_effects.csv"
    int_df = pd.read_csv(int_path) if int_path.exists() else pd.DataFrame()

    # Generate figures
    fig_performance_overview(df, output_dir / f"fig1_performance_overview.{fmt}")
    fig_decision_sensitivity(df, sens_df, output_dir / f"fig2_decision_sensitivity.{fmt}")
    fig_variance_and_interactions(anova_df, int_df, output_dir / f"fig3_variance_interactions.{fmt}")

    logger.info(f"\nAll figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise FUSE decision ensemble (Section 4.6)")
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"], default="pdf")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR

    if not results_csv.exists():
        logger.error(f"Results CSV not found: {results_csv}")
        sys.exit(1)

    run_visualisation(results_csv, analysis_dir, output_dir, args.format)


if __name__ == "__main__":
    main()
