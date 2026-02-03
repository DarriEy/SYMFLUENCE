#!/usr/bin/env python3
"""
Attribute Analysis Results Analyzer for SYMFLUENCE Paper Section 4.13

This script analyzes and compares results across all 9 discretization scenarios,
producing cross-scenario metrics, HRU statistics, and a narrative report.

Analysis components:
  1. Discretization summary: HRU counts, area distributions, attribute coverage
  2. Zonal statistics comparison: Elevation, soil, and land cover statistics per scenario
  3. Performance comparison: KGE, NSE, RMSE across scenarios (calibration & evaluation periods)
  4. Complexity-performance trade-off: HRU count vs model skill
  5. Calibration efficiency: Convergence rates per scenario
  6. Hydrological signature analysis: Flow duration curves, seasonality, peak timing

Usage:
    python analyze_attribute_analysis.py [--output-dir PATH]
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = DATA_DIR / "domain_Bow_at_Banff_attribute_analysis"

# Scenario definitions matching the run script
SCENARIOS = {
    "lumped_baseline": {"experiment_id": "attr_lumped_baseline", "label": "Lumped", "category": "baseline"},
    "elevation_200m": {"experiment_id": "attr_elevation_200m", "label": "Elev 200m", "category": "single"},
    "elevation_400m": {"experiment_id": "attr_elevation_400m", "label": "Elev 400m", "category": "single"},
    "landclass": {"experiment_id": "attr_landclass", "label": "Land Cover", "category": "single"},
    "soilclass": {"experiment_id": "attr_soilclass", "label": "Soil Type", "category": "single"},
    "aspect": {"experiment_id": "attr_aspect", "label": "Aspect", "category": "single"},
    "radiation": {"experiment_id": "attr_radiation", "label": "Radiation", "category": "single"},
    "elev_land": {"experiment_id": "attr_elev_land", "label": "Elev+Land", "category": "combined"},
    "elev_soil_land": {"experiment_id": "attr_elev_soil_land", "label": "Elev+Soil+Land", "category": "combined"},
}


def setup_logging() -> logging.Logger:
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("attribute_analyzer")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Discretization summary
# ─────────────────────────────────────────────────────────────────────────────

def count_hrus_from_shapefile(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Read HRU shapefile and compute statistics."""
    hru_dir = DOMAIN_DIR / "shapefiles" / "catchment"
    if not hru_dir.exists():
        return None

    try:
        import geopandas as gpd

        for pattern in [f"*{experiment_id}*HRUs*.shp", f"*{experiment_id}*HRUs*.gpkg"]:
            for shp in hru_dir.rglob(pattern):
                gdf = gpd.read_file(shp)
                areas = gdf["HRU_area"].values / 1e6 if "HRU_area" in gdf.columns else None
                stats = {
                    "n_hrus": len(gdf),
                    "shapefile": str(shp.name),
                }
                if areas is not None:
                    stats["area_mean_km2"] = float(np.mean(areas))
                    stats["area_min_km2"] = float(np.min(areas))
                    stats["area_max_km2"] = float(np.max(areas))
                    stats["area_std_km2"] = float(np.std(areas))
                    stats["total_area_km2"] = float(np.sum(areas))

                # Extract attribute columns if present
                for col in ["elevClass", "aspectClass", "landClass", "soilClass", "radiationClass"]:
                    if col in gdf.columns:
                        stats[f"{col}_unique"] = int(gdf[col].nunique())

                return stats
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: Could not read shapefile for {experiment_id}: {e}")

    return None


def analyze_discretization(logger: logging.Logger) -> Dict[str, Dict]:
    """Summarize HRU discretization across all scenarios."""
    logger.info("Analyzing discretization results...")
    results = {}
    for scenario_name, scenario in SCENARIOS.items():
        stats = count_hrus_from_shapefile(scenario["experiment_id"])
        if stats is not None:
            results[scenario_name] = stats
            logger.info(f"  {scenario['label']}: {stats['n_hrus']} HRUs")
        else:
            logger.info(f"  {scenario['label']}: No shapefile found")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Zonal statistics comparison
# ─────────────────────────────────────────────────────────────────────────────

def load_zonal_statistics(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load zonal statistics from intersection shapefiles."""
    intersection_dir = DOMAIN_DIR / "shapefiles" / "catchment_intersection"
    stats = {}

    try:
        import geopandas as gpd

        # Elevation stats
        dem_path = intersection_dir / "with_dem" / "catchment_with_dem.shp"
        if dem_path.exists():
            gdf = gpd.read_file(dem_path)
            for col in ["elev_mean", "elev_min", "elev_max", "elev_std"]:
                if col in gdf.columns:
                    stats[col] = float(gdf[col].mean())

        # Soil stats
        soil_path = intersection_dir / "with_soilgrids" / "catchment_with_soilclass.shp"
        if soil_path.exists():
            gdf = gpd.read_file(soil_path)
            usda_cols = [c for c in gdf.columns if c.startswith("USDA_")]
            stats["soil_classes"] = len(usda_cols)
            if "soil_dominant" in gdf.columns:
                stats["dominant_soil"] = gdf["soil_dominant"].mode().iloc[0] if len(gdf) > 0 else None

        # Land cover stats
        land_path = intersection_dir / "with_landclass" / "catchment_with_landclass.shp"
        if land_path.exists():
            gdf = gpd.read_file(land_path)
            igbp_cols = [c for c in gdf.columns if c.startswith("IGBP_")]
            stats["land_classes"] = len(igbp_cols)
            if "landcover_dominant" in gdf.columns:
                stats["dominant_landcover"] = gdf["landcover_dominant"].mode().iloc[0] if len(gdf) > 0 else None

    except ImportError:
        pass
    except Exception:
        pass

    return stats if stats else None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Performance comparison
# ─────────────────────────────────────────────────────────────────────────────

def find_performance_metrics(experiment_id: str) -> Optional[Dict[str, float]]:
    """Find and load performance metrics from optimization/evaluation results."""
    metrics = {}

    # Check optimization directory for calibration results
    opt_dir = DOMAIN_DIR / "optimization"
    if opt_dir.exists():
        for csv_file in opt_dir.rglob(f"*{experiment_id}*iteration_results*.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if "KGE" in df.columns:
                    metrics["calibration_best_kge"] = float(df["KGE"].max())
                    metrics["calibration_final_kge"] = float(df["KGE"].iloc[-1])
                    metrics["calibration_iterations"] = len(df)
                if "RMSE" in df.columns:
                    metrics["calibration_best_rmse"] = float(df["RMSE"].min())
                if "NSE" in df.columns:
                    metrics["calibration_best_nse"] = float(df["NSE"].max())
                break
            except Exception:
                continue

    # Check reporting directory for evaluation metrics
    report_dir = DOMAIN_DIR / "reporting"
    if report_dir.exists():
        for csv_file in report_dir.rglob(f"*{experiment_id}*metrics*.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                for col in df.columns:
                    if col.lower() in ("kge", "nse", "rmse", "pbias"):
                        metrics[f"evaluation_{col.lower()}"] = float(df[col].iloc[0])
                break
            except Exception:
                continue

    return metrics if metrics else None


def analyze_performance(logger: logging.Logger) -> Dict[str, Dict]:
    """Compare model performance across discretization scenarios."""
    logger.info("Analyzing performance metrics...")
    results = {}
    for scenario_name, scenario in SCENARIOS.items():
        metrics = find_performance_metrics(scenario["experiment_id"])
        if metrics is not None:
            results[scenario_name] = metrics
            kge = metrics.get("calibration_best_kge", "N/A")
            logger.info(f"  {scenario['label']}: Best KGE = {kge}")
        else:
            logger.info(f"  {scenario['label']}: No performance data found")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration convergence
# ─────────────────────────────────────────────────────────────────────────────

def analyze_convergence(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Analyze calibration convergence trajectory."""
    opt_dir = DOMAIN_DIR / "optimization"
    if not opt_dir.exists():
        return None

    for csv_file in opt_dir.rglob(f"*{experiment_id}*iteration_results*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            if "KGE" not in df.columns:
                continue

            kge_values = df["KGE"].values
            best_so_far = np.maximum.accumulate(kge_values)

            # Find iteration at which 90% and 95% of final performance is reached
            final_kge = best_so_far[-1]
            threshold_90 = 0.9 * final_kge if final_kge > 0 else final_kge * 1.1
            threshold_95 = 0.95 * final_kge if final_kge > 0 else final_kge * 1.05

            iter_90 = int(np.argmax(best_so_far >= threshold_90)) + 1 if np.any(best_so_far >= threshold_90) else len(kge_values)
            iter_95 = int(np.argmax(best_so_far >= threshold_95)) + 1 if np.any(best_so_far >= threshold_95) else len(kge_values)

            return {
                "total_iterations": len(kge_values),
                "final_best_kge": float(best_so_far[-1]),
                "iter_to_90pct": iter_90,
                "iter_to_95pct": iter_95,
                "convergence_trajectory": best_so_far[::max(1, len(best_so_far)//20)].tolist(),
            }
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(output_dir: Path):
    """Run the full analysis pipeline."""
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ATTRIBUTE ANALYSIS - SECTION 4.13")
    logger.info("Domain: Bow River at Banff (2210 km²)")
    logger.info("Model: HBV (14 parameters)")
    logger.info("Calibration: DDS, 1000 iterations, KGE metric")
    logger.info("Period: 2002-2009 (spinup: 2002-2003, cal: 2004-2007, eval: 2008-2009)")
    logger.info(f"Scenarios: {len(SCENARIOS)}")
    logger.info("=" * 70)

    # ── 1. Discretization summary ──────────────────────────────────────────
    discretization = analyze_discretization(logger)

    if discretization:
        csv_path = output_dir / "discretization_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["scenario", "label", "category", "n_hrus", "area_mean_km2",
                       "area_min_km2", "area_max_km2", "area_std_km2", "total_area_km2"]
            writer.writerow(header)
            for name, stats in discretization.items():
                scenario = SCENARIOS[name]
                writer.writerow([
                    name, scenario["label"], scenario["category"],
                    stats.get("n_hrus", ""),
                    f"{stats.get('area_mean_km2', ''):.2f}" if stats.get("area_mean_km2") else "",
                    f"{stats.get('area_min_km2', ''):.2f}" if stats.get("area_min_km2") else "",
                    f"{stats.get('area_max_km2', ''):.2f}" if stats.get("area_max_km2") else "",
                    f"{stats.get('area_std_km2', ''):.2f}" if stats.get("area_std_km2") else "",
                    f"{stats.get('total_area_km2', ''):.2f}" if stats.get("total_area_km2") else "",
                ])
        logger.info(f"Discretization summary saved to: {csv_path}")

    # ── 2. Zonal statistics ────────────────────────────────────────────────
    zonal_stats = load_zonal_statistics("attr_lumped_baseline")
    if zonal_stats:
        json_path = output_dir / "zonal_statistics.json"
        with open(json_path, "w") as f:
            json.dump(zonal_stats, f, indent=2, default=str)
        logger.info(f"Zonal statistics saved to: {json_path}")

    # ── 3. Performance comparison ──────────────────────────────────────────
    performance = analyze_performance(logger)

    if performance:
        csv_path = output_dir / "performance_comparison.csv"
        all_metric_keys = set()
        for m in performance.values():
            all_metric_keys.update(m.keys())
        all_metric_keys = sorted(all_metric_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "label", "category", "n_hrus"] + list(all_metric_keys))
            for name, metrics in performance.items():
                scenario = SCENARIOS[name]
                n_hrus = discretization.get(name, {}).get("n_hrus", "")
                row = [name, scenario["label"], scenario["category"], n_hrus]
                row += [f"{metrics.get(k, ''):.4f}" if isinstance(metrics.get(k), float) else "" for k in all_metric_keys]
                writer.writerow(row)
        logger.info(f"Performance comparison saved to: {csv_path}")

    # ── 4. Convergence analysis ────────────────────────────────────────────
    logger.info("Analyzing calibration convergence...")
    convergence = {}
    for scenario_name, scenario in SCENARIOS.items():
        conv = analyze_convergence(scenario["experiment_id"])
        if conv:
            convergence[scenario_name] = conv
            logger.info(f"  {scenario['label']}: 95% at iter {conv['iter_to_95pct']}")

    if convergence:
        csv_path = output_dir / "convergence_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "label", "total_iterations", "final_best_kge",
                             "iter_to_90pct", "iter_to_95pct"])
            for name, conv in convergence.items():
                scenario = SCENARIOS[name]
                writer.writerow([
                    name, scenario["label"],
                    conv["total_iterations"],
                    f"{conv['final_best_kge']:.4f}",
                    conv["iter_to_90pct"],
                    conv["iter_to_95pct"],
                ])
        logger.info(f"Convergence summary saved to: {csv_path}")

    # ── 5. Narrative report ────────────────────────────────────────────────
    report_path = output_dir / f"analysis_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ATTRIBUTE ANALYSIS REPORT - SECTION 4.13\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("DOMAIN: Bow River at Banff (WSC 05BB001)\n")
        f.write("  Catchment area: ~2210 km²\n")
        f.write("  Elevation range: ~1400-3400 m\n")
        f.write("  Model: HBV (14 parameters)\n")
        f.write("  Calibration: DDS, 1000 iterations, KGE\n")
        f.write("  Period: 2004-2007 (calibration), 2008-2009 (evaluation)\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. DISCRETIZATION SUMMARY\n")
        f.write("-" * 70 + "\n\n")

        if discretization:
            f.write(f"{'Scenario':<20} {'Category':<10} {'HRUs':>6} {'Mean Area (km²)':>16} {'Std (km²)':>12}\n")
            f.write("-" * 70 + "\n")
            for name, stats in discretization.items():
                scenario = SCENARIOS[name]
                f.write(
                    f"{scenario['label']:<20} {scenario['category']:<10} "
                    f"{stats.get('n_hrus', 'N/A'):>6} "
                    f"{stats.get('area_mean_km2', 0):>16.2f} "
                    f"{stats.get('area_std_km2', 0):>12.2f}\n"
                )
        else:
            f.write("  No discretization data available. Run the experiment first.\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("2. PERFORMANCE COMPARISON\n")
        f.write("-" * 70 + "\n\n")

        if performance:
            f.write(f"{'Scenario':<20} {'HRUs':>6} {'Cal KGE':>10} {'Cal RMSE':>10} {'Eval KGE':>10}\n")
            f.write("-" * 70 + "\n")
            for name, metrics in performance.items():
                scenario = SCENARIOS[name]
                n_hrus = discretization.get(name, {}).get("n_hrus", "N/A")
                cal_kge = metrics.get("calibration_best_kge", None)
                cal_rmse = metrics.get("calibration_best_rmse", None)
                eval_kge = metrics.get("evaluation_kge", None)
                f.write(
                    f"{scenario['label']:<20} {n_hrus:>6} "
                    f"{cal_kge:>10.4f}" if cal_kge is not None else f"{'N/A':>10}"
                )
                f.write(
                    f"{cal_rmse:>10.4f}" if cal_rmse is not None else f"{'N/A':>10}"
                )
                f.write(
                    f"{eval_kge:>10.4f}\n" if eval_kge is not None else f"{'N/A':>10}\n"
                )
        else:
            f.write("  No performance data available. Run calibration first.\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("3. CALIBRATION CONVERGENCE\n")
        f.write("-" * 70 + "\n\n")

        if convergence:
            f.write(f"{'Scenario':<20} {'Final KGE':>10} {'Iter@90%':>10} {'Iter@95%':>10}\n")
            f.write("-" * 55 + "\n")
            for name, conv in convergence.items():
                scenario = SCENARIOS[name]
                f.write(
                    f"{scenario['label']:<20} "
                    f"{conv['final_best_kge']:>10.4f} "
                    f"{conv['iter_to_90pct']:>10d} "
                    f"{conv['iter_to_95pct']:>10d}\n"
                )
        else:
            f.write("  No convergence data available.\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("4. METHODOLOGY NOTES\n")
        f.write("-" * 70 + "\n\n")
        f.write("This experiment evaluates the sensitivity of HBV model performance\n")
        f.write("to the choice of geospatial attribute-based domain discretization.\n")
        f.write("All scenarios share identical:\n")
        f.write("  - Forcing data (ERA5, 2002-2009)\n")
        f.write("  - Model structure (HBV, 14 parameters)\n")
        f.write("  - Calibration budget (DDS, 1000 iterations)\n")
        f.write("  - Evaluation metric (KGE)\n\n")
        f.write("The only variable across scenarios is the sub-grid discretization\n")
        f.write("configuration, which controls how the catchment is partitioned into\n")
        f.write("Hydrologic Response Units (HRUs) based on geospatial attributes:\n")
        f.write("  - Elevation: Continuous variable discretized into bands\n")
        f.write("  - Land cover: Categorical variable (IGBP classification)\n")
        f.write("  - Soil type: Categorical variable (USDA classification)\n")
        f.write("  - Aspect: Derived from DEM, 8 cardinal directions\n")
        f.write("  - Radiation: Annual solar radiation computed from DEM\n\n")
        f.write("Combined scenarios intersect multiple attribute layers,\n")
        f.write("potentially producing n_elev × n_land × n_soil HRUs per GRU.\n")

    logger.info(f"Analysis report saved to: {report_path}")
    logger.info("\nAnalysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attribute analysis experiment results (Section 4.13)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_DIR,
        help="Directory for analysis output files",
    )
    args = parser.parse_args()
    run_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
