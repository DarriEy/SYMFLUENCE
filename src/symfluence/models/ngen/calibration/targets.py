#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Calibration Targets

Provides calibration target classes for ngen model outputs.
Currently supports streamflow calibration with plans for snow, ET, etc.

Note: This module has been refactored to use the centralized evaluators in 
symfluence.evaluation.evaluators.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.evaluation import metrics
from symfluence.evaluation.evaluators import ModelEvaluator, StreamflowEvaluator


class NgenStreamflowTarget(StreamflowEvaluator):
    """NextGen-specific streamflow evaluator that handles nexus-style outputs."""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        self.station_id = config.get('STATION_ID', None)

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """
        Find ngen output files for calibration.

        Priority order:
        1. T-Route routed outputs (if available) - proper accumulated flow at outlet
        2. Single nexus output (if CALIBRATION_NEXUS_ID specified)
        3. All nexus outputs summed (fallback - approximates basin total)
        """
        # Check for t-route outputs first (NetCDF format)
        troute_dir = sim_dir / "troute_output"
        if troute_dir.exists():
            # Look for t-route NetCDF output
            troute_nc_files = list(troute_dir.glob("*.nc")) + list(troute_dir.glob("*.parquet"))
            if troute_nc_files:
                self.logger.info(f"Found t-route routing outputs: {len(troute_nc_files)} files")
                self.logger.info("Using routed flow from t-route (proper accumulated upstream flow)")
                return troute_nc_files

        # Fallback to raw nexus outputs
        files = list(sim_dir.glob('nex-*_output.csv'))
        if not files:
            # Try recursive search if not found in top level
            files = list(sim_dir.glob('**/nex-*_output.csv'))

        # Filter by CALIBRATION_NEXUS_ID if configured
        target_nexus = self.config.get('CALIBRATION_NEXUS_ID')
        if target_nexus:
            # Normalize ID (ensure it has nex- prefix if file has it)
            # Assuming config might say "nex-57" or just "57"
            target_files = [f for f in files if f.stem == f"{target_nexus}_output" or f.stem == target_nexus]

            if target_files:
                self.logger.info(f"Using calibration nexus: {target_nexus}")
                self.logger.warning(
                    f"T-Route routing outputs not found. Using raw nexus output. "
                    f"{target_nexus} will only show LOCAL catchment runoff, not accumulated upstream flow. "
                    f"To enable proper routing, ensure NGEN_RUN_TROUTE: true in config and t-route is installed."
                )
                return target_files
            else:
                self.logger.warning(f"Configured CALIBRATION_NEXUS_ID '{target_nexus}' not found in output files. Available: {[f.stem for f in files[:10]]}")
                self.logger.warning(f"Falling back to summing ALL nexus outputs")
                return files

        # If no CALIBRATION_NEXUS_ID specified, sum all nexuses
        self.logger.info(f"No CALIBRATION_NEXUS_ID specified. Summing all {len(files)} nexus outputs")
        self.logger.info("Note: For proper routed flow, enable t-route routing in config")
        return files

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """
        Extract streamflow from NGEN outputs.

        Handles both:
        - T-Route routed outputs (NetCDF/Parquet format)
        - Raw NGEN nexus outputs (CSV format)
        """
        if not sim_files:
            return pd.Series(dtype=float)

        # Check if we have t-route outputs (NetCDF or Parquet)
        first_file = sim_files[0]
        if first_file.suffix in ['.nc', '.parquet']:
            return self._extract_troute_data(sim_files)
        else:
            return self._extract_nexus_data(sim_files)

    def _extract_troute_data(self, troute_files: List[Path]) -> pd.Series:
        """Extract routed streamflow from t-route NetCDF/Parquet outputs."""
        import xarray as xr

        # Get target nexus ID (outlet)
        target_nexus = self.config.get('CALIBRATION_NEXUS_ID', 'nex-57')  # Default to outlet

        try:
            # Read t-route output (typically NetCDF with time and feature_id dimensions)
            for troute_file in troute_files:
                if troute_file.suffix == '.nc':
                    ds = xr.open_dataset(troute_file)

                    # T-Route outputs streamflow by feature_id (nexus)
                    # Look for streamflow variable (usually 'streamflow' or 'q_out')
                    flow_vars = [v for v in ds.data_vars if 'flow' in v.lower() or 'q' in v.lower()]

                    if not flow_vars:
                        self.logger.warning(f"No streamflow variable found in {troute_file}")
                        continue

                    flow_var = flow_vars[0]
                    self.logger.info(f"Using t-route variable: {flow_var}")

                    # Extract flow at target nexus
                    if 'feature_id' in ds.dims:
                        # Find target nexus in feature_id dimension
                        nexus_id_str = target_nexus.replace('nex-', '')  # May need to strip prefix
                        flow_data = ds[flow_var].sel(feature_id=nexus_id_str)
                    else:
                        # Single location or need to select differently
                        flow_data = ds[flow_var]

                    # Convert to pandas Series
                    flow_series = flow_data.to_series()
                    flow_series.name = f'{target_nexus}_routed'

                    ds.close()
                    self.logger.info(f"Extracted routed flow from t-route: {len(flow_series)} timesteps")
                    return flow_series.sort_index()

        except Exception as e:
            self.logger.error(f"Error reading t-route outputs: {e}")
            self.logger.warning("Falling back to raw nexus outputs")

        return pd.Series(dtype=float)

    def _extract_nexus_data(self, nexus_files: List[Path]) -> pd.Series:
        """Extract streamflow from raw NGEN nexus CSV outputs."""
        all_streamflow = []

        for nexus_file in nexus_files:
            try:
                # ngen output format: index, datetime, flow
                # Check for headerless format (common in NGEN)
                df = pd.read_csv(nexus_file)

                # Check for standard NGEN headerless format (index, time, flow)
                is_headerless = False
                if len(df.columns) == 3:
                    try:
                        # Try parsing the FIRST row's second column as date
                        pd.to_datetime(df.columns[1])
                        is_headerless = True
                    except (ValueError, TypeError):
                        pass

                if is_headerless:
                    # Reload with header=None
                    df = pd.read_csv(nexus_file, header=None, names=['index', 'datetime', 'flow'])

                if df.empty:
                    continue

                # Standardize columns if not headerless but weird
                if 'time' in df.columns:
                    df = df.rename(columns={'time': 'datetime'})
                if 'Time' in df.columns:
                    df = df.rename(columns={'Time': 'datetime'})

                # Find flow column
                if 'flow' not in df.columns:
                    for col in ['Flow', 'Q_OUT', 'streamflow', 'discharge', 'q_cms']:
                        if col in df.columns:
                            df = df.rename(columns={col: 'flow'})
                            break

                if 'datetime' not in df.columns or 'flow' not in df.columns:
                    self.logger.warning(f"Could not identify datetime/flow columns in {nexus_file}. Columns: {df.columns.tolist()}")
                    continue

                index = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None)
                s = pd.Series(
                    df['flow'].values,
                    index=index,
                    name=nexus_file.stem
                )
                all_streamflow.append(s)
            except Exception as e:
                self.logger.error(f"Error reading {nexus_file}: {e}")
                continue

        if not all_streamflow:
            return pd.Series(dtype=float)

        # Handle single vs multiple nexus outputs
        if len(all_streamflow) == 1:
            self.logger.debug(f"Using single nexus output: {all_streamflow[0].name}")
            return all_streamflow[0].sort_index()
        else:
            # Sum all nexus outputs for total catchment outflow
            self.logger.debug(f"Summing {len(all_streamflow)} nexus outputs for basin total")
            combined = pd.concat(all_streamflow, axis=1).sum(axis=1)
            combined.name = 'basin_total'
            return combined.sort_index()

    def calculate_metrics(self, sim: Any = None, experiment_id: str = None, 
                         output_dir: Optional[Path] = None, **kwargs) -> Dict[str, float]:
        """
        Standardized metrics calculation for NextGen.
        
        Args:
            sim: Optional Path to simulation directory or pre-loaded pd.Series.
                 If None, determined from experiment_id/output_dir.
            experiment_id: Experiment identifier (legacy, uses config if None)
            output_dir: Optional output directory (for parallel mode)
        """
        if sim is None:
            # Determine simulation directory
            if output_dir is not None:
                sim = Path(output_dir)
            else:
                exp_id = experiment_id or self.config.get('EXPERIMENT_ID')
                sim = self.project_dir / 'simulations' / exp_id / 'NGEN'
            
        # Use base class method with our specialized data extraction
        return super().calculate_metrics(sim=sim, **kwargs)

    def _get_catchment_area(self) -> float:
        """Detailed catchment area calculation for NextGen."""
        from pathlib import Path
        import geopandas as gpd
        import numpy as np

        cfg_area = (self.config.get("catchment", {}) or {}).get("area_km2")
        if cfg_area:
            return float(cfg_area)

        domain_dir = self.project_dir
        shp_dir = domain_dir / "shapefiles" / "catchment"
        if not shp_dir.exists():
            return 100.0

        candidates = sorted(shp_dir.glob("*HRUs_GRUs.shp")) + sorted(shp_dir.glob("*.shp"))
        shp_path = next((p for p in candidates if p.exists()), None)
        if not shp_path:
            return 100.0

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is None or gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:5070")

            area_km2 = float(gdf.geometry.area.sum() / 1e6)
            return area_km2
        except Exception as e:
            self.logger.warning(f"Error calculating catchment area from {shp_path}: {e}")
            return 100.0
