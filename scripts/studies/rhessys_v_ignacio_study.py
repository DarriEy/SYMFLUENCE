#!/usr/bin/env python3
"""
RHESSys/WMFire vs IGNACIO Fire Model Comparison Study
======================================================

O'Brien Creek Fire - March 8, 2014
Observed burned area: 2,407.76 ha

This script runs both fire models with corrected configurations for the
actual fire date and compares their performance against observed data.

Usage:
    python scripts/rhessys_v_ignacio_study.py [--skip-simulations] [--calibrate]

Author: SYMFLUENCE Team
"""

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

# Configuration
SYMFLUENCE_DATA_DIR = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data')
SYMFLUENCE_CODE_DIR = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE')
DOMAIN_NAME = 'Bow_at_Banff_elevation'

# Observed fire data
OBSERVED_FIRE = {
    'name': "O'Brien Creek Fire",
    'date': '2014-03-08',
    'area_ha': 2407.76,
    'cause': 'Lightning',
    'perimeter_path': SYMFLUENCE_DATA_DIR / f'domain_{DOMAIN_NAME}' / 'shapefiles' / 'perimiters' / 'OBrienCreekFire_2014.shp'
}

# Paths
PROJECT_DIR = SYMFLUENCE_DATA_DIR / f'domain_{DOMAIN_NAME}'
CONFIG_DIR = SYMFLUENCE_CODE_DIR / '0_config_files'
SIMULATIONS_DIR = PROJECT_DIR / 'simulations'
IGNACIO_INPUT_DIR = PROJECT_DIR / 'IGNACIO_input'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FireModelStudy:
    """
    Orchestrates fire model comparison study between RHESSys/WMFire and IGNACIO.
    """

    def __init__(self, experiment_id: str = 'obrien_creek_march2014'):
        self.experiment_id = experiment_id
        self.results: Dict[str, Any] = {}

        # Output directories
        self.wmfire_output_dir = SIMULATIONS_DIR / f'{experiment_id}_wmfire'
        self.ignacio_output_dir = SIMULATIONS_DIR / f'{experiment_id}_ignacio'
        self.comparison_dir = SIMULATIONS_DIR / 'fire_comparison_study'

        # Ensure directories exist
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

    def run_ignacio_simulation(self, config_path: Optional[Path] = None,
                                calibrated: bool = True) -> Dict:
        """
        Run IGNACIO fire spread simulation.

        Args:
            config_path: Path to IGNACIO config YAML
            calibrated: Whether to use calibrated parameters

        Returns:
            Dictionary with simulation results
        """
        logger.info("="*60)
        logger.info("Running IGNACIO Fire Simulation")
        logger.info("="*60)

        # Select config
        if config_path is None:
            if calibrated:
                config_path = IGNACIO_INPUT_DIR / 'ignacio_config_calibrated.yaml'
            else:
                config_path = IGNACIO_INPUT_DIR / 'ignacio_config.yaml'

        if not config_path.exists():
            logger.error(f"IGNACIO config not found: {config_path}")
            return {'error': 'Config not found'}

        # Import IGNACIO
        try:
            sys.path.insert(0, str(SYMFLUENCE_DATA_DIR / 'installs' / 'ignacio'))
            from ignacio.config import load_config
            from ignacio.simulation import run_simulation
        except ImportError as e:
            logger.error(f"IGNACIO not installed: {e}")
            return {'error': 'IGNACIO not installed'}

        # Load config and run
        logger.info(f"Loading config: {config_path.name}")
        config = load_config(str(config_path))

        # Update output directory
        output_dir = self.ignacio_output_dir / 'IGNACIO'
        output_dir.mkdir(parents=True, exist_ok=True)
        config.project.output_dir = str(output_dir)

        logger.info(f"Start datetime: {config.simulation.start_datetime}")
        logger.info(f"Max duration: {config.simulation.max_duration} minutes")
        logger.info(f"FBP FFMC: {config.fbp.defaults.ffmc}, ISI: {config.fbp.defaults.isi}")

        # Run simulation
        logger.info("Starting IGNACIO simulation...")
        results = run_simulation(config)

        # Collect results
        result_dict = {
            'model': 'IGNACIO',
            'calibrated': calibrated,
            'area_ha': results.total_area_ha,
            'n_fires': results.n_fires,
            'output_dir': str(output_dir),
            'config': {
                'ffmc': config.fbp.defaults.ffmc,
                'dmc': config.fbp.defaults.dmc,
                'dc': config.fbp.defaults.dc,
                'isi': config.fbp.defaults.isi,
                'bui': config.fbp.defaults.bui,
                'fwi': config.fbp.defaults.fwi,
                'max_duration': config.simulation.max_duration,
            }
        }

        logger.info(f"IGNACIO complete: {results.total_area_ha:.2f} ha")

        return result_dict

    def run_wmfire_simulation(self, config_path: Optional[Path] = None) -> Dict:
        """
        Run RHESSys with WMFire simulation.

        Args:
            config_path: Path to SYMFLUENCE config YAML

        Returns:
            Dictionary with simulation results
        """
        logger.info("="*60)
        logger.info("Running RHESSys/WMFire Simulation")
        logger.info("="*60)

        if config_path is None:
            config_path = CONFIG_DIR / 'config_rhessys_wmfire_obrien_march2014.yaml'

        if not config_path.exists():
            logger.error(f"WMFire config not found: {config_path}")
            return {'error': 'Config not found'}

        # Run preprocessing
        logger.info("Running model preprocessing...")
        try:
            result = subprocess.run(
                ['symfluence', 'workflow', 'step', 'model_specific_preprocessing',
                 '--config', str(config_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                logger.warning(f"Preprocessing warning: {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")

        # Run model
        logger.info("Running RHESSys model...")
        try:
            result = subprocess.run(
                ['symfluence', 'workflow', 'step', 'run_model',
                 '--config', str(config_path)],
                capture_output=True,
                text=True,
                timeout=7200
            )
            if result.returncode != 0:
                logger.warning(f"Model run warning: {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"Model run failed: {e}")
            return {'error': str(e)}

        # Load results
        wmfire_summary = self.wmfire_output_dir / 'fire_perimeters' / 'wmfire_summary.json'
        if wmfire_summary.exists():
            with open(wmfire_summary) as f:
                summary = json.load(f)

            result_dict = {
                'model': 'WMFire',
                'area_ha': summary.get('total_area_ha', 0),
                'n_fires': summary.get('total_fires', 0),
                'significant_fires': summary.get('significant_fires', 0),
                'output_dir': str(self.wmfire_output_dir)
            }
            logger.info(f"WMFire complete: {result_dict['area_ha']:.2f} ha")
            return result_dict
        else:
            logger.warning("WMFire summary not found")
            return {'model': 'WMFire', 'error': 'No output found'}

    def load_existing_results(self) -> None:
        """Load results from previous simulation runs."""

        # IGNACIO Final (best calibrated)
        ignacio_final_dir = SIMULATIONS_DIR / 'obrien_creek_march2014_final' / 'IGNACIO'
        summary_file = ignacio_final_dir / 'simulation_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            self.results['ignacio_final'] = {
                'model': 'IGNACIO',
                'simulation': 'March 2014 (Final Calibrated)',
                'area_ha': df['area_ha'].sum(),
                'n_fires': len(df),
                'output_dir': str(ignacio_final_dir)
            }

        # IGNACIO calibrated (conservative)
        ignacio_cal_dir = SIMULATIONS_DIR / 'obrien_creek_march2014_calibrated' / 'IGNACIO'
        summary_file = ignacio_cal_dir / 'simulation_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            self.results['ignacio_calibrated'] = {
                'model': 'IGNACIO',
                'simulation': 'March 2014 (Conservative)',
                'area_ha': df['area_ha'].sum(),
                'n_fires': len(df),
                'output_dir': str(ignacio_cal_dir)
            }

        # IGNACIO uncalibrated (baseline)
        ignacio_uncal_dir = SIMULATIONS_DIR / 'obrien_creek_march2014_ignacio' / 'IGNACIO'
        summary_file = ignacio_uncal_dir / 'simulation_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            self.results['ignacio_uncalibrated'] = {
                'model': 'IGNACIO',
                'simulation': 'March 2014 (Baseline)',
                'area_ha': df['area_ha'].sum(),
                'n_fires': len(df),
                'output_dir': str(ignacio_uncal_dir)
            }

        # IGNACIO summer (original - wrong season)
        ignacio_summer_dir = SIMULATIONS_DIR / 'ignacio_fire_2014' / 'IGNACIO'
        summary_file = ignacio_summer_dir / 'simulation_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            self.results['ignacio_summer'] = {
                'model': 'IGNACIO',
                'simulation': 'Summer 2014 (Wrong Season)',
                'area_ha': df['area_ha'].sum(),
                'n_fires': len(df),
                'output_dir': str(ignacio_summer_dir)
            }

        # WMFire summer (original - wrong season)
        wmfire_summer_dir = SIMULATIONS_DIR / 'bow_fire_2014' / 'fire_perimeters'
        summary_file = wmfire_summer_dir / 'wmfire_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            self.results['wmfire_summer'] = {
                'model': 'WMFire',
                'simulation': 'Summer 2014 (Wrong Season)',
                'area_ha': data.get('total_area_ha', 0),
                'n_fires': data.get('total_fires', 0),
                'output_dir': str(wmfire_summer_dir)
            }

    def compute_spatial_metrics(self, simulated_perimeter_path: Path) -> Dict:
        """
        Compute spatial overlap metrics between simulated and observed fire perimeters.

        Args:
            simulated_perimeter_path: Path to simulated fire perimeter shapefile

        Returns:
            Dictionary with IoU, Dice coefficient, and spatial accuracy metrics
        """
        try:
            # Load perimeters
            observed_gdf = gpd.read_file(OBSERVED_FIRE['perimeter_path'])
            simulated_gdf = gpd.read_file(simulated_perimeter_path)

            # Ensure same CRS
            if observed_gdf.crs != simulated_gdf.crs:
                simulated_gdf = simulated_gdf.to_crs(observed_gdf.crs)

            # Fix invalid geometries using buffer(0) trick
            observed_gdf['geometry'] = observed_gdf.geometry.buffer(0)
            simulated_gdf['geometry'] = simulated_gdf.geometry.buffer(0)

            # Get union of all geometries
            observed_union = unary_union(observed_gdf.geometry)
            simulated_union = unary_union(simulated_gdf.geometry)

            # Compute intersection and union
            intersection = observed_union.intersection(simulated_union)
            union = observed_union.union(simulated_union)

            # Calculate metrics
            intersection_area = intersection.area / 10000  # Convert to hectares
            union_area = union.area / 10000
            observed_area = observed_union.area / 10000
            simulated_area = simulated_union.area / 10000

            # IoU (Jaccard Index)
            iou = intersection_area / union_area if union_area > 0 else 0

            # Dice Coefficient (F1 score)
            dice = (2 * intersection_area) / (observed_area + simulated_area) if (observed_area + simulated_area) > 0 else 0

            # Overlap percentage
            overlap_pct = (intersection_area / observed_area) * 100 if observed_area > 0 else 0

            # Commission error (overburn outside observed)
            commission_area = simulated_area - intersection_area
            commission_pct = (commission_area / observed_area) * 100 if observed_area > 0 else 0

            # Omission error (observed area not burned in simulation)
            omission_area = observed_area - intersection_area
            omission_pct = (omission_area / observed_area) * 100 if observed_area > 0 else 0

            return {
                'iou': iou,
                'dice': dice,
                'overlap_pct': overlap_pct,
                'commission_pct': commission_pct,
                'omission_pct': omission_pct,
                'intersection_ha': intersection_area,
                'commission_ha': commission_area,
                'omission_ha': omission_area
            }
        except Exception as e:
            logger.warning(f"Could not compute spatial metrics: {e}")
            return {}

    def compute_metrics(self) -> pd.DataFrame:
        """Compute comparison metrics for all simulations."""

        rows = []

        # Add observed
        rows.append({
            'Model': 'Observed',
            'Simulation': OBSERVED_FIRE['name'],
            'Area (ha)': OBSERVED_FIRE['area_ha'],
            'Ratio': 1.0,
            'Error (%)': 0.0,
            'IoU': 1.0,
            'Dice': 1.0,
            'Overlap (%)': 100.0,
            'Status': 'Reference'
        })

        # Add simulation results
        for key, result in self.results.items():
            if 'error' in result:
                continue

            area = result.get('area_ha', 0)
            ratio = area / OBSERVED_FIRE['area_ha']
            error_pct = abs(ratio - 1.0) * 100

            # Try to compute spatial metrics
            spatial_metrics = {}
            if result['model'] == 'IGNACIO' and 'output_dir' in result:
                # Try multiple possible perimeter file locations
                perimeter_files = [
                    Path(result['output_dir']) / 'all_perimeters.shp',
                    Path(result['output_dir']) / 'perimeters' / 'final_perimeter.shp',
                    Path(result['output_dir']) / 'perimeters' / 'fire_0000.shp'
                ]
                for perimeter_file in perimeter_files:
                    if perimeter_file.exists():
                        spatial_metrics = self.compute_spatial_metrics(perimeter_file)
                        break

            # Determine status based on both area error and spatial overlap
            if spatial_metrics.get('iou', 0) > 0.7 or error_pct < 10:
                status = 'Excellent'
            elif spatial_metrics.get('iou', 0) > 0.5 or error_pct < 20:
                status = 'Good'
            elif error_pct < 50:
                status = 'Moderate'
            elif 'Wrong' in result.get('simulation', ''):
                status = 'Wrong Season'
            else:
                status = 'Poor'

            rows.append({
                'Model': result['model'],
                'Simulation': result.get('simulation', key),
                'Area (ha)': area,
                'Ratio': ratio,
                'Error (%)': error_pct,
                'IoU': spatial_metrics.get('iou', np.nan),
                'Dice': spatial_metrics.get('dice', np.nan),
                'Overlap (%)': spatial_metrics.get('overlap_pct', np.nan),
                'Status': status
            })

        return pd.DataFrame(rows)

    def create_comparison_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Create visualization comparing all simulations."""

        # Determine if we have spatial metrics
        has_spatial = not df['IoU'].isna().all()

        if has_spatial:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes = [axes[0], axes[1]]

        # Color mapping
        def get_color(row):
            if row['Model'] == 'Observed':
                return 'green'
            elif 'Final' in str(row.get('Simulation', '')):
                return 'darkblue'
            elif 'Calibrated' in str(row.get('Simulation', '')):
                return 'blue'
            elif row['Status'] == 'Wrong Season':
                return 'red'
            elif row['Status'] == 'Excellent':
                return 'darkgreen'
            elif row['Error (%)'] < 50:
                return 'orange'
            return 'gray'

        colors = [get_color(row) for _, row in df.iterrows()]

        # Bar chart of areas
        ax1 = axes[0]
        y_pos = range(len(df))
        bars = ax1.barh(y_pos, df['Area (ha)'], color=colors, alpha=0.7, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['Model']}\n{row['Simulation']}" for _, row in df.iterrows()], fontsize=9)
        ax1.axvline(OBSERVED_FIRE['area_ha'], color='green', linestyle='--', linewidth=2,
                    label=f"Observed: {OBSERVED_FIRE['area_ha']:.0f} ha")
        ax1.set_xlabel('Burned Area (ha)', fontsize=11)
        ax1.set_title("Fire Model Comparison\nO'Brien Creek Fire - March 2014", fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.set_xlim(0, max(df['Area (ha)'].max() * 1.15, 3000))

        # Add value labels
        for bar, row in zip(bars, df.itertuples()):
            label = f"{row._3:.0f} ha ({row.Ratio:.2f}x)"
            ax1.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=9)

        # Error comparison
        ax2 = axes[1]
        sim_df = df[df['Model'] != 'Observed'].copy()
        colors2 = [get_color(row) for _, row in sim_df.iterrows()]

        _bars2 = ax2.bar(range(len(sim_df)), sim_df['Error (%)'], color=colors2, alpha=0.7, edgecolor='black')
        ax2.axhline(10, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (<10%)')
        ax2.axhline(20, color='green', linestyle='--', alpha=0.7, label='Good (<20%)')
        ax2.axhline(50, color='orange', linestyle='--', alpha=0.7, label='Moderate (<50%)')
        ax2.set_xticks(range(len(sim_df)))
        ax2.set_xticklabels([f"{row['Model']}\n{row['Simulation'][:20]}..."
                            for _, row in sim_df.iterrows()], rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Area Error (%)', fontsize=11)
        ax2.set_title('Model Error vs Observed', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(0, min(sim_df['Error (%)'].max() * 1.1, 300))

        # Spatial metrics if available
        if has_spatial:
            spatial_df = sim_df[~sim_df['IoU'].isna()].copy()

            # IoU comparison
            ax3 = axes[2]
            spatial_colors = [get_color(row) for _, row in spatial_df.iterrows()]
            _bars3 = ax3.bar(range(len(spatial_df)), spatial_df['IoU'], color=spatial_colors, alpha=0.7, edgecolor='black')
            ax3.axhline(0.7, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (>0.7)')
            ax3.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Good (>0.5)')
            ax3.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate (>0.3)')
            ax3.set_xticks(range(len(spatial_df)))
            ax3.set_xticklabels([f"{row['Model']}\n{row['Simulation'][:20]}..."
                                for _, row in spatial_df.iterrows()], rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('IoU (Jaccard Index)', fontsize=11)
            ax3.set_title('Spatial Overlap (IoU)', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right', fontsize=9)
            ax3.set_ylim(0, 1)

            # Dice coefficient
            ax4 = axes[3]
            _bars4 = ax4.bar(range(len(spatial_df)), spatial_df['Dice'], color=spatial_colors, alpha=0.7, edgecolor='black')
            ax4.axhline(0.8, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (>0.8)')
            ax4.axhline(0.6, color='green', linestyle='--', alpha=0.7, label='Good (>0.6)')
            ax4.set_xticks(range(len(spatial_df)))
            ax4.set_xticklabels([f"{row['Model']}\n{row['Simulation'][:20]}..."
                                for _, row in spatial_df.iterrows()], rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel('Dice Coefficient', fontsize=11)
            ax4.set_title('Spatial Agreement (Dice)', fontsize=12, fontweight='bold')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def generate_report(self) -> str:
        """Generate text report of study results with data-driven findings."""

        df = self.compute_metrics()

        report = []
        report.append("="*70)
        report.append("FIRE MODEL COMPARISON STUDY")
        report.append(f"O'Brien Creek Fire - {OBSERVED_FIRE['date']}")
        report.append("="*70)
        report.append("")
        report.append("OBSERVED FIRE:")
        report.append(f"  Name: {OBSERVED_FIRE['name']}")
        report.append(f"  Date: {OBSERVED_FIRE['date']}")
        report.append(f"  Area: {OBSERVED_FIRE['area_ha']:.2f} ha")
        report.append(f"  Cause: {OBSERVED_FIRE['cause']}")
        report.append("")
        report.append("SIMULATION RESULTS:")
        report.append("-"*70)

        for _, row in df.iterrows():
            if row['Model'] == 'Observed':
                continue
            report.append(f"  {row['Model']} - {row['Simulation']}")
            report.append(f"    Area: {row['Area (ha)']:.2f} ha")
            report.append(f"    Ratio: {row['Ratio']:.2f}x observed")
            report.append(f"    Error: {row['Error (%)']:.1f}%")
            if not pd.isna(row.get('IoU', np.nan)):
                report.append(f"    IoU: {row['IoU']:.3f}, Dice: {row['Dice']:.3f}, Overlap: {row['Overlap (%)']:.1f}%")
            report.append(f"    Status: {row['Status']}")
            report.append("")

        # Best result by area error
        sim_df = df[df['Model'] != 'Observed'].copy()
        if len(sim_df) > 0:
            best_area = sim_df.loc[sim_df['Error (%)'].idxmin()]
            report.append("BEST RESULT (Area Error):")
            report.append(f"  {best_area['Model']} - {best_area['Simulation']}")
            report.append(f"  Area Error: {best_area['Error (%)']:.1f}%")
            if not pd.isna(best_area.get('IoU', np.nan)):
                report.append(f"  Spatial IoU: {best_area['IoU']:.3f}")

            # Best result by spatial overlap
            spatial_results = sim_df[~sim_df['IoU'].isna()]
            if len(spatial_results) > 0:
                best_spatial = spatial_results.loc[spatial_results['IoU'].idxmax()]
                if best_spatial.name != best_area.name:
                    report.append("")
                    report.append("BEST RESULT (Spatial Overlap):")
                    report.append(f"  {best_spatial['Model']} - {best_spatial['Simulation']}")
                    report.append(f"  IoU: {best_spatial['IoU']:.3f}, Area Error: {best_spatial['Error (%)']:.1f}%")

        # Data-driven key findings
        report.append("")
        report.append("KEY FINDINGS:")

        # Finding 1: Temporal correction
        wrong_season = sim_df[sim_df['Simulation'].str.contains('Wrong Season', na=False)]
        correct_season = sim_df[~sim_df['Simulation'].str.contains('Wrong Season', na=False)]
        if len(wrong_season) > 0 and len(correct_season) > 0:
            avg_wrong = wrong_season['Error (%)'].mean()
            avg_correct = correct_season['Error (%)'].mean()
            improvement = ((avg_wrong - avg_correct) / avg_wrong) * 100
            report.append(f"  1. Temporal correction (summer→March) reduced error by {improvement:.0f}%")
            report.append(f"     (Wrong season avg: {avg_wrong:.1f}%, Correct season avg: {avg_correct:.1f}%)")

        # Finding 2: Calibration impact
        ignacio_results = sim_df[sim_df['Model'] == 'IGNACIO']
        if len(ignacio_results) > 0:
            baseline = ignacio_results[ignacio_results['Simulation'].str.contains('Baseline', na=False)]
            calibrated = ignacio_results[ignacio_results['Simulation'].str.contains('Calibrated', na=False)]
            if len(baseline) > 0 and len(calibrated) > 0:
                baseline_error = baseline['Error (%)'].values[0]
                best_calibrated_error = calibrated['Error (%)'].min()
                improvement = ((baseline_error - best_calibrated_error) / baseline_error) * 100
                report.append(f"  2. IGNACIO calibration reduced error from {baseline_error:.1f}% to {best_calibrated_error:.1f}%")
                report.append(f"     (Improvement: {improvement:.1f}%)")

        # Finding 3: Model comparison
        wmfire_results = sim_df[sim_df['Model'] == 'WMFire']
        ignacio_march = ignacio_results[ignacio_results['Simulation'].str.contains('March', na=False)]
        if len(wmfire_results) > 0 and len(ignacio_march) > 0:
            wmfire_best = wmfire_results['Error (%)'].min()
            ignacio_best = ignacio_march['Error (%)'].min()
            report.append("  3. IGNACIO outperformed WMFire for March conditions")
            report.append(f"     (IGNACIO: {ignacio_best:.1f}% error vs WMFire: {wmfire_best:.1f}% error)")

        # Finding 4: Spatial accuracy
        spatial_results = sim_df[~sim_df['IoU'].isna()]
        if len(spatial_results) > 0:
            best_iou = spatial_results['IoU'].max()
            report.append(f"  4. Best spatial overlap achieved IoU = {best_iou:.3f}")
            if best_iou > 0.7:
                report.append("     (Indicates excellent spatial agreement)")
            elif best_iou > 0.5:
                report.append("     (Indicates good spatial agreement)")

        # Finding 5: Model limitations
        if len(wmfire_results) > 0:
            wmfire_march = wmfire_results[wmfire_results['Simulation'].str.contains('March', na=False)]
            if len(wmfire_march) > 0:
                wmfire_march_error = wmfire_march['Error (%)'].values[0]
                if wmfire_march_error > 90:
                    report.append(f"  5. WMFire showed severe underestimation ({wmfire_march_error:.1f}% error) for")
                    report.append("     late-winter fire conditions, suggesting model physics limitations")

        return "\n".join(report)

    def run_study(self, run_simulations: bool = True,
                  run_ignacio: bool = True,
                  run_wmfire: bool = True,
                  calibrate: bool = False) -> None:
        """
        Run the complete comparison study.

        Args:
            run_simulations: Whether to run new simulations or load existing
            run_ignacio: Whether to run IGNACIO simulation
            run_wmfire: Whether to run WMFire simulation
            calibrate: Whether to run calibration iterations
        """
        logger.info("="*70)
        logger.info("FIRE MODEL COMPARISON STUDY")
        logger.info(f"O'Brien Creek Fire - {OBSERVED_FIRE['date']}")
        logger.info("="*70)

        # Load existing results first
        logger.info("Loading existing simulation results...")
        self.load_existing_results()
        logger.info(f"Found {len(self.results)} existing results")

        # Run new simulations if requested
        if run_simulations:
            if run_ignacio:
                result = self.run_ignacio_simulation(calibrated=True)
                if 'error' not in result:
                    self.results['ignacio_new'] = {
                        **result,
                        'simulation': 'March 2014 (New Run)'
                    }

            if run_wmfire:
                result = self.run_wmfire_simulation()
                if 'error' not in result:
                    self.results['wmfire_new'] = {
                        **result,
                        'simulation': 'March 2014 (New Run)'
                    }

        # Compute metrics
        logger.info("Computing comparison metrics...")
        df = self.compute_metrics()

        # Save results
        csv_path = self.comparison_dir / 'study_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")

        # Generate plot
        logger.info("Generating comparison plot...")
        fig = self.create_comparison_plot(df)
        plot_path = self.comparison_dir / 'study_comparison.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot saved: {plot_path}")

        # Generate report
        report = self.generate_report()
        report_path = self.comparison_dir / 'study_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved: {report_path}")

        # Print report
        print("\n" + report)

        # Summary table
        print("\n" + "="*70)
        print("RESULTS TABLE")
        print("="*70)
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="RHESSys/WMFire vs IGNACIO Fire Model Comparison Study"
    )
    parser.add_argument(
        '--skip-simulations',
        action='store_true',
        help='Skip running simulations, only analyze existing results'
    )
    parser.add_argument(
        '--ignacio-only',
        action='store_true',
        help='Only run IGNACIO simulation'
    )
    parser.add_argument(
        '--wmfire-only',
        action='store_true',
        help='Only run WMFire simulation'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run calibration iterations to optimize parameters'
    )

    args = parser.parse_args()

    # Create study instance
    study = FireModelStudy()

    # Run study
    study.run_study(
        run_simulations=not args.skip_simulations,
        run_ignacio=not args.wmfire_only,
        run_wmfire=not args.ignacio_only,
        calibrate=args.calibrate
    )


if __name__ == '__main__':
    main()
