#!/usr/bin/env python3
"""
Decision Ensemble Analysis — SYMFLUENCE Paper Section 4.6

Quantifies structural uncertainty in FUSE by analysing the 64-member
decision ensemble (6 binary decisions).  Produces:

  1. Per-decision sensitivity (mean difference, Welch t-test, Cohen d, η²)
  2. ANOVA-style variance decomposition (main effects + interactions)
  3. Pairwise interaction effects with actual option names
  4. Failure-mode characterisation of catastrophic structures
  5. Combination rankings and best/worst identification
  6. Compact, publication-ready analysis report

Usage:
    python analyze_decision_ensemble.py [--results-csv PATH] [--output-dir DIR]
"""

import argparse
import logging
import sys
from datetime import datetime
from itertools import combinations as combo_pairs
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"

SYMFLUENCE_DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DEFAULT_RESULTS_CSV = (
    SYMFLUENCE_DATA_DIR / "domain_Bow_at_Banff_lumped_era5" / "optimization"
    / "decision_ensemble_fuse_decisions_comparison.csv"
)

VARIED_DECISIONS = ["ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF"]
ALL_METRICS = ["kge", "kgep", "nse", "mae", "rmse"]
HIGHER_BETTER = {"kge", "kgep", "nse"}
LOWER_BETTER = {"mae", "rmse"}

# Human-readable decision / option names
DECISION_NAMES = {
    "ARCH1": "Upper-layer architecture",
    "ARCH2": "Lower-layer architecture",
    "QSURF": "Surface runoff",
    "QPERC": "Percolation",
    "ESOIL": "Evaporation",
    "QINTF": "Interflow",
}
OPTION_NAMES = {
    "tension1_1": "Tension (2-state)",
    "onestate_1": "Single bucket",
    "tens2pll_2": "Tension parallel",
    "unlimfrc_2": "Unlimited fraction",
    "arno_x_vic": "VIC-style",
    "prms_varnt": "PRMS-style",
    "perc_f2sat": "Fraction-to-saturation",
    "perc_lower": "Lower-zone control",
    "sequential": "Sequential",
    "rootweight": "Root weighting",
    "intflwnone": "None",
    "intflwsome": "Active",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("decision_analysis")


# ===================================================================
# Data loading
# ===================================================================
def load_results(path: Path) -> pd.DataFrame:
    """Load master CSV and coerce metric columns to float."""
    logger.info(f"Loading results from: {path}")
    df = pd.read_csv(path)
    for col in ALL_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    metric_cols = [c for c in ALL_METRICS if c in df.columns]
    df = df.dropna(subset=metric_cols, how="all")
    logger.info(f"Loaded {len(df)} valid combinations")
    return df


# ===================================================================
# 1. Per-decision sensitivity with statistical tests
# ===================================================================
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled standard deviation)."""
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def compute_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each varied decision compute:
      - mean & std per option
      - Welch t-test (two-sided) → t-statistic, p-value
      - Cohen's d effect size
      - η² (eta-squared) from one-way ANOVA
    """
    rows = []
    for dec in VARIED_DECISIONS:
        if dec not in df.columns:
            continue
        opts = sorted(df[dec].unique())
        if len(opts) != 2:
            continue
        ga = df.loc[df[dec] == opts[0], "kge"].dropna().values
        gb = df.loc[df[dec] == opts[1], "kge"].dropna().values
        if len(ga) < 2 or len(gb) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(ga, gb, equal_var=False)
        f_stat, p_anova = stats.f_oneway(ga, gb)
        d = _cohens_d(ga, gb)

        # η² = SS_between / SS_total
        grand_mean = np.concatenate([ga, gb]).mean()
        ss_between = len(ga) * (ga.mean() - grand_mean) ** 2 + len(gb) * (gb.mean() - grand_mean) ** 2
        ss_total = np.sum((np.concatenate([ga, gb]) - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        rows.append({
            "decision": dec,
            "decision_name": DECISION_NAMES.get(dec, dec),
            "option_a": opts[0],
            "option_a_name": OPTION_NAMES.get(opts[0], opts[0]),
            "option_b": opts[1],
            "option_b_name": OPTION_NAMES.get(opts[1], opts[1]),
            "n_a": len(ga),
            "n_b": len(gb),
            "mean_a": float(ga.mean()),
            "mean_b": float(gb.mean()),
            "std_a": float(ga.std(ddof=1)),
            "std_b": float(gb.std(ddof=1)),
            "delta": float(ga.mean() - gb.mean()),
            "abs_delta": float(abs(ga.mean() - gb.mean())),
            "cohens_d": d,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "eta_squared": float(eta_sq),
        })

    out = pd.DataFrame(rows).sort_values("abs_delta", ascending=False)
    return out


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ===================================================================
# 2. ANOVA variance decomposition (full factorial, Type-I SS)
# ===================================================================
def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-level full-factorial ANOVA for KGE:
      - Main effects for each of the 6 decisions
      - All 15 two-way interactions
      - Residual
    Reports SS, fraction of total variance, F-statistic, p-value.
    """
    metric = "kge"
    if metric not in df.columns:
        return pd.DataFrame()

    y = df[metric].values
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    n = len(y)

    # Encode decisions as -1 / +1
    codes = {}
    for dec in VARIED_DECISIONS:
        if dec not in df.columns:
            continue
        opts = sorted(df[dec].unique())
        if len(opts) == 2:
            codes[dec] = np.where(df[dec] == opts[0], -1.0, 1.0)

    if not codes:
        return pd.DataFrame()

    decs = list(codes.keys())

    # Main effects SS
    rows = []
    ss_explained = 0.0
    for dec in decs:
        x = codes[dec]
        contrast = x @ y / n
        ss = n * contrast ** 2
        ss_explained += ss
        rows.append({"source": dec, "source_name": DECISION_NAMES.get(dec, dec),
                      "SS": ss, "df": 1, "type": "main"})

    # Two-way interaction SS
    for d1, d2 in combo_pairs(decs, 2):
        x_int = codes[d1] * codes[d2]
        contrast = x_int @ y / n
        ss = n * contrast ** 2
        ss_explained += ss
        rows.append({"source": f"{d1}×{d2}",
                      "source_name": f"{DECISION_NAMES.get(d1, d1)} × {DECISION_NAMES.get(d2, d2)}",
                      "SS": ss, "df": 1, "type": "interaction"})

    ss_residual = ss_total - ss_explained
    df_residual = n - 1 - len(rows)  # total df minus model terms
    ms_residual = ss_residual / max(df_residual, 1)

    rows.append({"source": "Residual", "source_name": "Residual",
                  "SS": ss_residual, "df": df_residual, "type": "residual"})

    out = pd.DataFrame(rows)
    out["pct_variance"] = out["SS"] / ss_total * 100
    out["MS"] = out["SS"] / out["df"].clip(lower=1)
    out["F"] = np.where(out["type"] != "residual", out["MS"] / ms_residual, np.nan)
    out["p_value"] = np.where(
        out["type"] != "residual",
        [1 - stats.f.cdf(f, 1, df_residual) if not np.isnan(f) else np.nan for f in out["F"]],
        np.nan
    )
    return out


# ===================================================================
# 3. Pairwise interaction effects (with actual option names)
# ===================================================================
def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair (D_i, D_j), compute the conditional effect of D_i
    at each level of D_j.  Interaction = difference of conditional effects.
    """
    rows = []
    for i, di in enumerate(VARIED_DECISIONS):
        for dj in VARIED_DECISIONS[i + 1:]:
            if di not in df.columns or dj not in df.columns:
                continue
            oi = sorted(df[di].unique())
            oj = sorted(df[dj].unique())
            if len(oi) != 2 or len(oj) != 2:
                continue

            for metric in ["kge"]:
                if metric not in df.columns:
                    continue
                # Effect of D_i when D_j = oj[0]
                sub_j0 = df[df[dj] == oj[0]]
                eff_at_j0 = sub_j0.loc[sub_j0[di] == oi[0], metric].mean() - sub_j0.loc[sub_j0[di] == oi[1], metric].mean()
                # Effect of D_i when D_j = oj[1]
                sub_j1 = df[df[dj] == oj[1]]
                eff_at_j1 = sub_j1.loc[sub_j1[di] == oi[0], metric].mean() - sub_j1.loc[sub_j1[di] == oi[1], metric].mean()
                interaction = eff_at_j0 - eff_at_j1

                rows.append({
                    "decision_1": di,
                    "decision_1_name": DECISION_NAMES.get(di, di),
                    "decision_2": dj,
                    "decision_2_name": DECISION_NAMES.get(dj, dj),
                    "condition": f"{OPTION_NAMES.get(oj[0], oj[0])}",
                    "effect_at_cond_a": eff_at_j0,
                    "condition_b": f"{OPTION_NAMES.get(oj[1], oj[1])}",
                    "effect_at_cond_b": eff_at_j1,
                    "interaction": interaction,
                    "abs_interaction": abs(interaction),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_interaction", ascending=False)
    return out


# ===================================================================
# 4. Failure-mode analysis
# ===================================================================
def failure_mode_analysis(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Identify structures with KGE < threshold and tabulate the
    frequency of each decision option among failures vs successes.
    """
    if "kge" not in df.columns:
        return pd.DataFrame()

    failures = df[df["kge"] < threshold]
    successes = df[df["kge"] >= threshold]
    n_fail, n_succ = len(failures), len(successes)

    rows = []
    for dec in VARIED_DECISIONS:
        if dec not in df.columns:
            continue
        for opt in sorted(df[dec].unique()):
            f_count = (failures[dec] == opt).sum()
            s_count = (successes[dec] == opt).sum()
            f_rate = f_count / n_fail if n_fail > 0 else 0
            s_rate = s_count / n_succ if n_succ > 0 else 0
            rows.append({
                "decision": dec,
                "option": opt,
                "option_name": OPTION_NAMES.get(opt, opt),
                "n_failures": f_count,
                "pct_failures": f_rate * 100,
                "n_successes": s_count,
                "pct_successes": s_rate * 100,
                "enrichment": f_rate / s_rate if s_rate > 0 else np.inf,
            })
    return pd.DataFrame(rows)


# ===================================================================
# 5. Rankings
# ===================================================================
def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rankings = df.copy()
    for m in ALL_METRICS:
        if m not in rankings.columns:
            continue
        asc = m in LOWER_BETTER
        rankings[f"{m}_rank"] = rankings[m].rank(ascending=asc)
    rank_cols = [c for c in rankings.columns if c.endswith("_rank")]
    if rank_cols:
        rankings["avg_rank"] = rankings[rank_cols].mean(axis=1)
        rankings = rankings.sort_values("avg_rank")
    return rankings


# ===================================================================
# 6. Report
# ===================================================================
def generate_report(
    df: pd.DataFrame,
    sensitivity: pd.DataFrame,
    anova: pd.DataFrame,
    interactions: pd.DataFrame,
    failures: pd.DataFrame,
    output_dir: Path,
) -> None:
    path = output_dir / "analysis_report.txt"
    decs = [d for d in VARIED_DECISIONS if d in df.columns]
    n_fail = len(df[df["kge"] < 0]) if "kge" in df.columns else 0

    with open(path, "w") as f:
        f.write("FUSE Decision Ensemble Analysis Report\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("=" * 72 + "\n\n")

        # --- 1. Overview ---
        f.write("1. EXPERIMENTAL DESIGN\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Combinations evaluated : {len(df)}\n")
        f.write(f"  Varied decisions (6)   : {', '.join(decs)}\n")
        f.write("  Fixed decisions (3)    : RFERR=multiplc_e, Q_TDH=rout_gamma, SNOWM=temp_index\n")
        f.write("  Optimisation           : SCE-UA (1000 trials per structure)\n")
        f.write("  Calibration metric     : KGE\n\n")

        # --- 2. Summary statistics ---
        f.write("2. PERFORMANCE SUMMARY\n")
        f.write("-" * 72 + "\n")
        f.write(f"  {'Metric':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'IQR':>8}\n")
        for m in ALL_METRICS:
            if m not in df.columns:
                continue
            v = df[m].dropna()
            iqr = v.quantile(0.75) - v.quantile(0.25)
            f.write(f"  {m.upper():<8} {v.mean():8.3f} {v.std():8.3f} {v.min():8.3f} {v.max():8.3f} {iqr:8.3f}\n")
        f.write(f"\n  Structures with KGE < 0: {n_fail} / {len(df)} ({n_fail/len(df)*100:.0f}%)\n\n")

        # --- 3. Decision sensitivity ---
        f.write("3. DECISION SENSITIVITY (KGE)\n")
        f.write("-" * 72 + "\n")
        f.write(f"  {'Decision':<26} {'|Δ KGE|':>8} {'η²':>6} {'Cohen d':>8} {'p-value':>10} {'Sig':>5}\n")
        for _, r in sensitivity.iterrows():
            sig = _sig_stars(r["p_value"])
            f.write(
                f"  {r['decision_name']:<26} {r['abs_delta']:8.3f} {r['eta_squared']:6.3f} "
                f"{r['cohens_d']:+8.3f} {r['p_value']:10.4f} {sig:>5}\n"
            )
            f.write(
                f"    {r['option_a_name']:<24} mean={r['mean_a']:.3f} (±{r['std_a']:.3f}, n={r['n_a']})\n"
            )
            f.write(
                f"    {r['option_b_name']:<24} mean={r['mean_b']:.3f} (±{r['std_b']:.3f}, n={r['n_b']})\n"
            )

        # --- 4. Variance decomposition ---
        if not anova.empty:
            f.write("\n4. VARIANCE DECOMPOSITION (KGE, Type-I SS)\n")
            f.write("-" * 72 + "\n")
            f.write(f"  {'Source':<40} {'%Var':>7} {'F':>8} {'p':>10} {'Sig':>5}\n")
            for _, r in anova.iterrows():
                sig = _sig_stars(r["p_value"]) if not np.isnan(r["p_value"]) else ""
                f_str = f"{r['F']:8.2f}" if not np.isnan(r["F"]) else "       —"
                p_str = f"{r['p_value']:10.4f}" if not np.isnan(r["p_value"]) else "         —"
                f.write(f"  {r['source_name']:<40} {r['pct_variance']:7.1f} {f_str} {p_str} {sig:>5}\n")

            # Summarise main vs interaction vs residual
            main_pct = anova.loc[anova["type"] == "main", "pct_variance"].sum()
            int_pct = anova.loc[anova["type"] == "interaction", "pct_variance"].sum()
            res_pct = anova.loc[anova["type"] == "residual", "pct_variance"].sum()
            f.write(f"\n  Total main effects     : {main_pct:5.1f}%\n")
            f.write(f"  Total 2-way interactions: {int_pct:5.1f}%\n")
            f.write(f"  Residual               : {res_pct:5.1f}%\n")

        # --- 5. Key interactions ---
        if not interactions.empty:
            f.write("\n5. DOMINANT INTERACTIONS (KGE)\n")
            f.write("-" * 72 + "\n")
            for _, r in interactions.head(5).iterrows():
                f.write(f"\n  {r['decision_1_name']} × {r['decision_2_name']}  "
                        f"(|interaction| = {r['abs_interaction']:.3f})\n")
                f.write(f"    Effect of {r['decision_1_name']} when {r['decision_2_name']} = {r['condition']}:  "
                        f"Δ = {r['effect_at_cond_a']:+.3f}\n")
                f.write(f"    Effect of {r['decision_1_name']} when {r['decision_2_name']} = {r['condition_b']}:  "
                        f"Δ = {r['effect_at_cond_b']:+.3f}\n")

        # --- 6. Failure-mode analysis ---
        if not failures.empty and n_fail > 0:
            f.write(f"\n6. FAILURE-MODE ANALYSIS (KGE < 0, n={n_fail})\n")
            f.write("-" * 72 + "\n")
            enriched = failures[failures["enrichment"] > 1.5].sort_values("enrichment", ascending=False)
            if not enriched.empty:
                f.write("  Options enriched among failures (enrichment > 1.5×):\n")
                for _, r in enriched.iterrows():
                    f.write(
                        f"    {r['decision']}/{r['option_name']:<24}: "
                        f"{r['pct_failures']:.0f}% of failures vs {r['pct_successes']:.0f}% of successes "
                        f"(enrichment {r['enrichment']:.1f}×)\n"
                    )
            depleted = failures[failures["enrichment"] < 0.5].sort_values("enrichment")
            if not depleted.empty:
                f.write("\n  Options depleted among failures (enrichment < 0.5×):\n")
                for _, r in depleted.iterrows():
                    f.write(
                        f"    {r['decision']}/{r['option_name']:<24}: "
                        f"{r['pct_failures']:.0f}% of failures vs {r['pct_successes']:.0f}% of successes "
                        f"(enrichment {r['enrichment']:.1f}×)\n"
                    )

        # --- 7. Best / worst ---
        f.write("\n7. BEST AND WORST STRUCTURES\n")
        f.write("-" * 72 + "\n")
        best5 = df.nlargest(5, "kge")
        worst5 = df.nsmallest(5, "kge")
        for label, sub in [("Best", best5), ("Worst", worst5)]:
            f.write(f"\n  {label} 5:\n")
            for rank, (_, r) in enumerate(sub.iterrows(), 1):
                dec_str = "  ".join(f"{d}={OPTION_NAMES.get(r[d], r[d])}" for d in decs)
                f.write(f"    #{rank}: KGE={r['kge']:+.3f}  NSE={r['nse']:+.3f}  RMSE={r['rmse']:.1f}\n")
                f.write(f"         {dec_str}\n")

    logger.info(f"Report saved to: {path}")


# ===================================================================
# Main pipeline
# ===================================================================
def run_analysis(results_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_csv)
    if df.empty:
        logger.error("No valid results to analyse")
        return

    # Sensitivity
    logger.info("Computing per-decision sensitivity (Welch t-test, Cohen d, η²)...")
    sensitivity = compute_sensitivity(df)
    sensitivity.to_csv(output_dir / "decision_sensitivity.csv", index=False)

    # ANOVA
    logger.info("Computing variance decomposition (ANOVA)...")
    anova = variance_decomposition(df)
    anova.to_csv(output_dir / "variance_decomposition.csv", index=False)

    # Interactions
    logger.info("Computing pairwise interaction effects...")
    interactions = compute_interactions(df)
    interactions.to_csv(output_dir / "interaction_effects.csv", index=False)

    # Failure modes
    logger.info("Analysing failure modes (KGE < 0)...")
    failures = failure_mode_analysis(df, threshold=0.0)
    failures.to_csv(output_dir / "failure_modes.csv", index=False)

    # Rankings
    logger.info("Computing combination rankings...")
    rankings = compute_rankings(df)
    rankings.to_csv(output_dir / "combination_rankings.csv", index=False)

    # Best / worst
    best5 = df.nlargest(5, "kge")
    worst5 = df.nsmallest(5, "kge")
    pd.concat([best5.assign(category="top_5"), worst5.assign(category="bottom_5")]).to_csv(
        output_dir / "best_worst_structures.csv", index=False
    )

    # Report
    generate_report(df, sensitivity, anova, interactions, failures, output_dir)

    # Console summary
    print("\n" + "=" * 72)
    print("DECISION ENSEMBLE ANALYSIS SUMMARY")
    print("=" * 72)
    if "kge" in df.columns:
        v = df["kge"]
        print(f"\nKGE: {v.mean():.3f} ± {v.std():.3f}  [{v.min():.3f}, {v.max():.3f}]")
        print(f"Structures with KGE < 0: {(v < 0).sum()} / {len(v)}")

    print("\nDecision sensitivity (sorted by |Δ KGE|):")
    print(f"  {'Decision':<26} {'|Δ KGE|':>8} {'η²':>6} {'p':>10} {'Sig':>5}")
    for _, r in sensitivity.iterrows():
        sig = _sig_stars(r["p_value"])
        print(f"  {r['decision_name']:<26} {r['abs_delta']:8.3f} {r['eta_squared']:6.3f} {r['p_value']:10.4f} {sig:>5}")

    if not anova.empty:
        main_pct = anova.loc[anova["type"] == "main", "pct_variance"].sum()
        int_pct = anova.loc[anova["type"] == "interaction", "pct_variance"].sum()
        print(f"\nVariance explained: main effects {main_pct:.1f}%, interactions {int_pct:.1f}%")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyse FUSE decision ensemble (Section 4.6)")
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR

    if not results_csv.exists():
        logger.error(f"Results CSV not found: {results_csv}")
        sys.exit(1)

    run_analysis(results_csv, output_dir)


if __name__ == "__main__":
    main()
