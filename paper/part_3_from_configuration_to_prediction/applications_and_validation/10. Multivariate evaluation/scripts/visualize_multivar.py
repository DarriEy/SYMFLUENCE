#!/usr/bin/env python3
"""
Multivariate Evaluation Visualization for SYMFLUENCE Paper Section 4.10

Creates publication-quality figures for the three multivariate evaluation experiments:
  a) Bow: TWS anomaly time series and Taylor diagram
  b) Paradise: SCA maps + soil moisture scatter/time series
  c) Iceland: SCF trend maps and elevation-band trend plots

Usage:
    python visualize_multivar.py [--study bow|paradise|iceland|all] [--format pdf|png]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")

STUDY_DOMAINS = {
    "bow": "Bow_at_Banff_multivar",
    "paradise": "paradise_multivar",
    "iceland": "Iceland_multivar",
}


def setup_logging() -> logging.Logger:
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("multivar_visualization")


def create_bow_figures(logger: logging.Logger, fmt: str = "pdf") -> List[Path]:
    """
    Create figures for the Bow River GRACE TWS study.

    Planned figures:
    1. TWS anomaly time series: 3-panel (GRACE, uncalibrated SUMMA, calibrated SUMMA)
    2. Scatter plot: simulated vs observed TWS anomalies
    3. Seasonal cycle comparison
    4. Taylor diagram comparing both SUMMA runs against GRACE
    """
    logger.info("Creating Bow GRACE TWS figures...")
    figures_created = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig_dir = FIGURES_DIR / "bow_grace_tws"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: TWS anomaly time series placeholder
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Bow River TWS Anomaly Comparison (Section 4.10a)", fontsize=14)

        labels = ["GRACE TWS Anomaly", "SUMMA Uncalibrated TWS", "SUMMA Calibrated TWS"]
        colors = ["#2166ac", "#b2182b", "#1b7837"]

        for ax, label, color in zip(axes, labels, colors):
            ax.set_ylabel("TWS Anomaly [mm]")
            ax.set_title(label)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.text(0.5, 0.5, "Awaiting simulation data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="gray", style="italic")
            ax.set_xlim(2002, 2018)
            ax.set_ylim(-150, 150)

        axes[-1].set_xlabel("Year")
        plt.tight_layout()

        fig_path = fig_dir / f"bow_tws_anomaly_timeseries.{fmt}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_created.append(fig_path)
        logger.info(f"  Created: {fig_path}")

        # Figure 2: Taylor diagram placeholder
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title("Taylor Diagram: SUMMA vs GRACE TWS (Section 4.10a)")
        ax.text(0.5, 0.5, "Awaiting simulation data\n\nWill show:\n- Uncalibrated SUMMA\n- Calibrated SUMMA\n- vs GRACE reference",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray", style="italic")

        fig_path = fig_dir / f"bow_tws_taylor_diagram.{fmt}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_created.append(fig_path)
        logger.info(f"  Created: {fig_path}")

    except ImportError:
        logger.warning("  matplotlib not available. Skipping figure generation.")

    return figures_created


def create_paradise_figures(logger: logging.Logger, fmt: str = "pdf") -> List[Path]:
    """
    Create figures for the Paradise SCA & soil moisture study.

    Planned figures:
    1. SCA comparison maps: side-by-side MODIS vs simulated for selected dates
    2. SCA monthly accuracy metrics bar chart
    3. Soil moisture time series: simulated vs SMAP
    4. Soil moisture scatter plot with 1:1 line
    5. Combined performance summary heatmap
    """
    logger.info("Creating Paradise SCA & SM figures...")
    figures_created = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = FIGURES_DIR / "paradise_sca_sm"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: SCA accuracy metrics placeholder
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Paradise Multivariate Evaluation (Section 4.10b)", fontsize=14)

        axes[0].set_title("Snow Cover Area: Simulated vs MODIS MOD10A1")
        axes[0].set_ylabel("SCA Metric Score")
        axes[0].text(0.5, 0.5, "Awaiting simulation data\n\nMetrics: Accuracy, F1-Score, CSI",
                     transform=axes[0].transAxes, ha="center", va="center",
                     fontsize=12, color="gray", style="italic")

        axes[1].set_title("Soil Moisture: Simulated vs SMAP L3")
        axes[1].set_ylabel("Soil Moisture [m³/m³]")
        axes[1].set_xlabel("Date")
        axes[1].text(0.5, 0.5, "Awaiting simulation data\n\nMetrics: Correlation, RMSE, Bias",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=12, color="gray", style="italic")

        plt.tight_layout()
        fig_path = fig_dir / f"paradise_sca_sm_summary.{fmt}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_created.append(fig_path)
        logger.info(f"  Created: {fig_path}")

    except ImportError:
        logger.warning("  matplotlib not available. Skipping figure generation.")

    return figures_created


def create_iceland_figures(logger: logging.Logger, fmt: str = "pdf") -> List[Path]:
    """
    Create figures for the Iceland SCF trend study.

    Planned figures:
    1. Annual mean SCF time series (simulated vs MODIS) with trend lines
    2. Seasonal SCF trend decomposition (accumulation, ablation, snow-free)
    3. SCF trend map over Iceland (spatial Sen's slope)
    4. Elevation-band stratified trend bar chart
    5. Trend significance map (p-values)
    """
    logger.info("Creating Iceland SCF trend figures...")
    figures_created = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig_dir = FIGURES_DIR / "iceland_scf_trend"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Annual SCF time series with trends
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Iceland SCF Trend Analysis (Section 4.10c)\nFull MODIS Record 2000-2023", fontsize=14)

        years = np.arange(2000, 2024)

        axes[0].set_title("Annual Mean Snow Cover Fraction")
        axes[0].set_ylabel("SCF [%]")
        axes[0].set_xlim(1999, 2024)
        axes[0].set_ylim(0, 100)
        axes[0].text(0.5, 0.5, "Awaiting simulation data\n\nMODIS MOD10A2 vs Simulated SCF\nMann-Kendall trend test",
                     transform=axes[0].transAxes, ha="center", va="center",
                     fontsize=12, color="gray", style="italic")

        axes[1].set_title("SCF Trends by Elevation Band")
        axes[1].set_ylabel("Sen's Slope [% SCF / decade]")
        axes[1].set_xlabel("Elevation Band [m]")
        elev_labels = ["0-200", "200-400", "400-600", "600-800", "800-1000", "1000-1200", "1200-1500", "1500-2000"]
        axes[1].set_xticks(range(len(elev_labels)))
        axes[1].set_xticklabels(elev_labels, rotation=45)
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].text(0.5, 0.5, "Awaiting simulation data",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=12, color="gray", style="italic")

        plt.tight_layout()
        fig_path = fig_dir / f"iceland_scf_trend_summary.{fmt}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_created.append(fig_path)
        logger.info(f"  Created: {fig_path}")

        # Figure 2: Seasonal decomposition placeholder
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Seasonal SCF Trends over Iceland (2000-2023)", fontsize=14)

        seasons = ["Accumulation (Oct-Mar)", "Ablation (Apr-Jun)", "Snow-Free (Jul-Sep)"]
        for ax, season in zip(axes, seasons):
            ax.set_title(season)
            ax.set_xlabel("Year")
            ax.set_ylabel("SCF [%]")
            ax.text(0.5, 0.5, "Awaiting data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray", style="italic")

        plt.tight_layout()
        fig_path = fig_dir / f"iceland_scf_seasonal_trends.{fmt}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_created.append(fig_path)
        logger.info(f"  Created: {fig_path}")

    except ImportError:
        logger.warning("  matplotlib not available. Skipping figure generation.")

    return figures_created


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create publication figures for Section 4.10 multivariate evaluation",
    )
    parser.add_argument(
        "--study",
        type=str,
        nargs="+",
        default=["all"],
        choices=["bow", "paradise", "iceland", "all"],
        help="Which study/studies to visualize (default: all)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output figure format (default: pdf)",
    )

    args = parser.parse_args()
    logger = setup_logging()

    studies = args.study
    if "all" in studies:
        studies = ["bow", "paradise", "iceland"]

    logger.info("Starting Multivariate Evaluation Visualization (Section 4.10)")

    visualizers = {
        "bow": create_bow_figures,
        "paradise": create_paradise_figures,
        "iceland": create_iceland_figures,
    }

    all_figures = []
    for study_key in studies:
        if study_key in visualizers:
            figures = visualizers[study_key](logger, fmt=args.format)
            all_figures.extend(figures)

    logger.info(f"\nTotal figures created: {len(all_figures)}")
    for fig_path in all_figures:
        logger.info(f"  {fig_path}")


if __name__ == "__main__":
    main()
