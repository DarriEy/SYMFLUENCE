#!/usr/bin/env python3
"""Cross-catchment analysis for the Large Sample study.

Loads FUSE simulation results from all LamaH-Ice catchments, computes
performance metrics (KGE, NSE, bias), and generates comparison plots.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"


def get_domain_ids():
    """Discover all configured domain IDs from config files."""
    ids = []
    for f in CONFIGS_DIR.glob("config_lamahice_*_FUSE.yaml"):
        parts = f.stem.replace("config_lamahice_", "").replace("_FUSE", "")
        try:
            ids.append(int(parts))
        except ValueError:
            continue
    return sorted(ids)


DOMAIN_IDS = get_domain_ids()
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")


def compute_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Compute Kling-Gupta Efficiency."""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def compute_nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency."""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)


def compute_pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    """Compute percent bias."""
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan
    return 100 * np.sum(sim - obs) / np.sum(obs)


def load_results(domain_id: int) -> dict:
    """Load simulation and observation data for a domain.

    Returns dict with 'sim', 'obs' arrays and 'metadata', or None if not found.
    """
    domain_dir = SYMFLUENCE_DATA / f"domain_lamahice_{domain_id}"
    eval_dir = domain_dir / "evaluation"

    if not eval_dir.exists():
        print(f"  Warning: No evaluation results for domain {domain_id}")
        return None

    # Look for standard SYMFLUENCE output files
    sim_files = list(eval_dir.glob("*simulated*.csv")) + list(eval_dir.glob("*sim*.csv"))
    obs_files = list(eval_dir.glob("*observed*.csv")) + list(eval_dir.glob("*obs*.csv"))

    if not sim_files or not obs_files:
        print(f"  Warning: Missing sim/obs files for domain {domain_id}")
        return None

    sim_df = pd.read_csv(sim_files[0], parse_dates=True, index_col=0)
    obs_df = pd.read_csv(obs_files[0], parse_dates=True, index_col=0)

    return {
        "sim": sim_df.values.flatten(),
        "obs": obs_df.values.flatten(),
        "domain_id": domain_id,
    }


def analyze_all(domain_ids: list) -> pd.DataFrame:
    """Compute metrics for all domains."""
    rows = []
    for did in domain_ids:
        data = load_results(did)
        if data is None:
            rows.append({"domain_id": did, "KGE": np.nan, "NSE": np.nan, "PBIAS": np.nan})
            continue

        sim, obs = data["sim"], data["obs"]
        rows.append({
            "domain_id": did,
            "KGE": compute_kge(sim, obs),
            "NSE": compute_nse(sim, obs),
            "PBIAS": compute_pbias(sim, obs),
        })

    return pd.DataFrame(rows)


def plot_metric_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate bar chart comparing metrics across catchments."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, ["KGE", "NSE", "PBIAS"]):
        bars = ax.bar(
            [str(d) for d in df["domain_id"]],
            df[metric],
            color="steelblue" if metric != "PBIAS" else "coral",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Domain ID")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} across LamaH-Ice catchments")
        if metric in ("KGE", "NSE"):
            ax.set_ylim(-0.5, 1.0)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_large_sample_metrics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved metric comparison plot to {output_dir}")


def plot_kge_distribution(df: pd.DataFrame, output_dir: Path):
    """Generate box/violin plot of KGE distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))

    valid_kge = df["KGE"].dropna()
    if len(valid_kge) > 0:
        ax.boxplot(valid_kge, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.scatter(np.ones(len(valid_kge)), valid_kge, color="steelblue",
                   zorder=3, s=60, edgecolors="black", linewidths=0.5)

    ax.set_ylabel("KGE")
    ax.set_title("KGE Distribution - LamaH-Ice Large Sample")
    ax.set_xticks([1])
    ax.set_xticklabels(["FUSE"])

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_large_sample_kge_dist.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cross-catchment analysis for Large Sample study")
    parser.add_argument("--domains", nargs="+", type=int, default=DOMAIN_IDS)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Large Sample Study - Cross-catchment Analysis")
    print(f"Domains: {args.domains}")
    print("=" * 60)

    df = analyze_all(args.domains)

    # Save metrics table
    metrics_path = args.output_dir / "large_sample_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    print(df.to_string(index=False))

    # Summary statistics
    print("\nSummary Statistics:")
    for metric in ["KGE", "NSE", "PBIAS"]:
        vals = df[metric].dropna()
        if len(vals) > 0:
            print(f"  {metric}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
                  f"std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")

    # Generate figures
    print("\nGenerating figures...")
    plot_metric_comparison(df, args.figures_dir)
    plot_kge_distribution(df, args.figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
