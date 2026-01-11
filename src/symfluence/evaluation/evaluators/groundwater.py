#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Groundwater Evaluator.

Evaluates simulated groundwater from SUMMA against well observations (depth) or
GRACE satellite water storage anomalies (total water storage).

Groundwater Targets:
    - gw_depth: Well water table depth (positive values, meters below surface)
    - gw_grace: GRACE total water storage anomalies (for aquifer storage changes)

Model Output (SUMMA):
    - scalarTotalSoilWat: Total soil water (kg/m² → mm, converted to meters)
    - scalarAquiferStorage: Aquifer storage (m directly)
    - Water storage components: SWE, soil water, aquifer, canopy

Observations:
    - gw_depth: Well observations (depth below surface, meters)
    - gw_grace: GRACE monthly anomalies (mm water thickness)

Well Observations Characteristics:
    - Variable frequency (daily, weekly, monthly, quarterly)
    - Can have gaps and inconsistent measurements
    - May require datum correction/offset
    - Often need auto-alignment to match simulated mean

Configuration:
    GW_BASE_DEPTH: Reference depth for groundwater (default: 50.0 m)
    GW_AUTO_ALIGN: Auto-align simulated mean to observed (default: True)
    GRACE_PROCESSING_CENTER: GRACE center ('csr', 'jpl', 'gsfc', default: 'csr')
"""

import logging
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('GROUNDWATER')
class GroundwaterEvaluator(ModelEvaluator):
    """Groundwater evaluator comparing SUMMA to well or GRACE observations.

    Supports two evaluation modes for groundwater:
    1. Well observations: Comparison of simulated vs observed water table depth
    2. GRACE observations: Comparison of storage anomalies

    Well Depth Evaluation (gw_depth):
        - Converts simulated storage to depth below surface
        - Formula: gw_depth = (base_depth - storage_m).abs()
        - Handles two storage variables:
          * scalarTotalSoilWat: Soil water (kg/m² → mm → m)
          * scalarAquiferStorage: Aquifer storage (m directly)
        - Auto-alignment: Shifts simulated to match observed mean (useful for datum offsets)

    GRACE Evaluation (gw_grace):
        - Sums water storage components (SWE, soil, aquifer, canopy)
        - Compares with GRACE satellite anomalies
        - Auto unit conversion based on data range

    Configuration:
        GW_BASE_DEPTH: Reference depth for depth calculation (default: 50.0 m)
        GW_AUTO_ALIGN: Auto-align simulated mean to observed (default: True)
        GRACE_PROCESSING_CENTER: GRACE center to use (default: 'csr')

    Attributes:
        optimization_target: 'gw_depth' or 'gw_grace'
        variable_name: Same as optimization_target
        grace_center: GRACE processing center ('csr', 'jpl', 'gsfc')
    """

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger):
        """Initialize groundwater evaluator with target determination.

        Determines evaluation target (well depth vs GRACE) from configuration
        and initializes GRACE processing center selection.

        Args:
            config: Typed configuration object
            project_dir: Project root directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)

        self.optimization_target = self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow'
        )
        if self.optimization_target not in ['gw_depth', 'gw_grace']:
            eval_var = self.config_dict.get('EVALUATION_VARIABLE', '')
            if 'gw_' in eval_var:
                self.optimization_target = eval_var

        self.variable_name = self.optimization_target
        self.grace_center = self.config_dict.get('GRACE_PROCESSING_CENTER', 'csr')
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate SUMMA output files containing groundwater storage variables.

        Searches for NetCDF files with scalarTotalSoilWat, scalarAquiferStorage,
        or other water storage components needed for groundwater evaluation.

        Args:
            sim_dir: Directory containing SUMMA simulation output

        Returns:
            List[Path]: Paths to groundwater output files (NetCDF)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_groundwater_files(sim_dir)
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        # Sort files to try daily first
        sim_files.sort(key=lambda x: "day" in x.name, reverse=True)
        
        for sim_file in sim_files:
            try:
                self.logger.debug(f"Trying to extract groundwater from {sim_file.name}")
                with xr.open_dataset(sim_file) as ds:
                    data = None
                    if self.optimization_target == 'gw_depth':
                        data = self._extract_groundwater_depth(ds)
                    elif self.optimization_target == 'gw_grace':
                        data = self._extract_total_water_storage(ds)
                    else:
                        data = self._extract_groundwater_depth(ds)
                    
                    if data is not None and not data.empty:
                        self.logger.info(f"Successfully extracted {len(data)} points from {sim_file.name}")
                        return data
            except Exception as e:
                self.logger.warning(f"Failed to extract from {sim_file.name}: {e}")
                
        raise ValueError(f"Could not extract groundwater data from any of {sim_files}")
    
    def _extract_groundwater_depth(self, ds: xr.Dataset) -> pd.Series:
        if 'scalarTotalSoilWat' in ds.variables:
            gw_var = ds['scalarTotalSoilWat']
            
            # Collapse spatial dimensions
            sim_xr = gw_var
            for dim in ['hru', 'gru']:
                if dim in sim_xr.dims:
                    if sim_xr.sizes[dim] == 1:
                        sim_xr = sim_xr.isel({dim: 0})
                    else:
                        sim_xr = sim_xr.mean(dim=dim)
            
            # Handle any remaining non-time dimensions
            non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
            if non_time_dims:
                sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
                
            sim_data = sim_xr.to_pandas()

            # Convert storage to depth-below-surface if comparing to GGMN
            # TotalSoilWat is in kg/m2 (mm). Convert to meters.
            sim_data_m = sim_data / 1000.0

            base_depth = float(self.config_dict.get('GW_BASE_DEPTH', 50.0))
            gw_depth_sim = (base_depth - sim_data_m).abs()

            # Auto-align: If the means are wildly different, shift the simulation to match the observation mean
            if self.config_dict.get('GW_AUTO_ALIGN', True):
                obs = self._load_observed_data()
                if obs is not None and not obs.empty:
                    offset = obs.mean() - gw_depth_sim.mean()
                    self.logger.info(f"Auto-aligning groundwater simulated mean with offset: {offset:.3f}")
                    gw_depth_sim = gw_depth_sim + offset

            return gw_depth_sim
        elif 'scalarAquiferStorage' in ds.variables:
            gw_var = ds['scalarAquiferStorage']
            sim_xr = gw_var
            for dim in ['hru', 'gru']:
                if dim in sim_xr.dims:
                    if sim_xr.sizes[dim] == 1:
                        sim_xr = sim_xr.isel({dim: 0})
                    else:
                        sim_xr = sim_xr.mean(dim=dim)

            sim_data = sim_xr.to_pandas()
            base_depth = float(self.config_dict.get('GW_BASE_DEPTH', 50.0))
            gw_depth_sim = (base_depth - sim_data).abs()

            if self.config_dict.get('GW_AUTO_ALIGN', True):
                obs = self._load_observed_data()
                if obs is not None and not obs.empty:
                    offset = obs.mean() - gw_depth_sim.mean()
                    self.logger.info(f"Auto-aligning groundwater simulated mean with offset: {offset:.3f}")
                    gw_depth_sim = gw_depth_sim + offset
            return gw_depth_sim
        else:
            return pd.Series()
        try:
            storage_components = {}
            if 'scalarSWE' in ds.variables:
                storage_components['swe'] = ds['scalarSWE']
            if 'scalarTotalSoilWat' in ds.variables:
                storage_components['soil'] = ds['scalarTotalSoilWat']
            if 'scalarAquiferStorage' in ds.variables:
                storage_components['aquifer'] = ds['scalarAquiferStorage']
            if 'scalarCanopyWat' in ds.variables:
                storage_components['canopy'] = ds['scalarCanopyWat']
            
            if not storage_components:
                raise ValueError("No water storage components found")
            
            total_storage = None
            for component_name, component_data in storage_components.items():
                # Collapse spatial dimensions for this component
                sim_xr = component_data
                for dim in ['hru', 'gru']:
                    if dim in sim_xr.dims:
                        if sim_xr.sizes[dim] == 1:
                            sim_xr = sim_xr.isel({dim: 0})
                        else:
                            sim_xr = sim_xr.mean(dim=dim)
                
                # Handle any remaining non-time dimensions
                non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
                if non_time_dims:
                    sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
                
                if total_storage is None:
                    total_storage = sim_xr
                else:
                    total_storage = total_storage + sim_xr
            
            sim_data = total_storage.to_pandas()
            return self._convert_tws_units(sim_data)
        except Exception as e:
            self.logger.error(f"Error calculating TWS: {str(e)}")
            raise
    
    def _convert_tws_units(self, tws_data: pd.Series) -> pd.Series:
        data_range = tws_data.max() - tws_data.min()
        if data_range > 1000:
            return tws_data / 10.0
        elif data_range > 10:
            return tws_data * 100.0
        return tws_data
    
    def get_observed_data_path(self) -> Path:
        if self.optimization_target == 'gw_depth':
            return self.project_dir / "observations" / "groundwater" / "depth" / "processed" / f"{self.domain_name}_gw_processed.csv"
        elif self.optimization_target == 'gw_grace':
            return self.project_dir / "observations" / "groundwater" / "grace" / "processed" / f"{self.domain_name}_grace_processed.csv"
        else:
             return self.project_dir / "observations" / "groundwater" / "depth" / "processed" / f"{self.domain_name}_gw_processed.csv"

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'gw_depth':
            for col in columns:
                if any(term in col.lower() for term in ['depth', 'depth_m', 'water_level']):
                    return col
            if 'Depth_m' in columns:
                return 'Depth_m'
        elif self.optimization_target == 'gw_grace':
            grace_columns = {
                'jpl': ['grace_jpl_tws'],
                'csr': ['grace_csr_tws'],
                'gsfc': ['grace_gsfc_tws']
            }
            preferred_cols = grace_columns.get(self.grace_center, ['grace_csr_tws'])
            for col in preferred_cols:
                if col in columns:
                    return col
            for col in columns:
                if 'grace' in col.lower() and 'tws' in col.lower():
                    return col
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None
            
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                return None
            
            if self.optimization_target == 'gw_depth':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%m/%d/%Y', errors='coerce')
            
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            obs_series = obs_series.dropna()
            
            if obs_series.index.tz is not None:
                obs_series.index = obs_series.index.tz_convert('UTC').tz_localize(None)

            return obs_series
        except Exception as e:
            self.logger.error(f"Error loading observed groundwater data: {str(e)}")
            return None

    def needs_routing(self) -> bool:
        """Determine if groundwater evaluation requires streamflow routing.

        Groundwater is measured at point-scale (wells) and stored at basin scale
        (GRACE) without requiring streamflow routing models. Storage is evaluated
        directly without downstream propagation.

        Returns:
            bool: False (groundwater evaluator never requires routing)
        """
        return False
