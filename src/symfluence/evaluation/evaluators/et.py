#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evapotranspiration (ET) Evaluator

Supports multiple ET observation sources:
- MODIS MOD16A2 (cloud-based acquisition via AppEEARS)
- FLUXCOM gridded ET
- FluxNet tower observations
- GLEAM ET products
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


@EvaluationRegistry.register('ET')
@EvaluationRegistry.register('MODIS_ET')
@EvaluationRegistry.register('MOD16')
@EvaluationRegistry.register('FLUXNET')
@EvaluationRegistry.register('FLUXNET_ET')
class ETEvaluator(ModelEvaluator):
    """
    Evapotranspiration evaluator supporting multiple observation sources.

    Supported observation sources (set via ET_OBS_SOURCE config):
    - 'mod16', 'modis', 'modis_et': MODIS MOD16A2 8-day ET product
    - 'fluxcom', 'fluxcom_et': FLUXCOM gridded ET
    - 'fluxnet': FluxNet tower observations
    - 'gleam': GLEAM ET products

    Configuration:
        ET_OBS_SOURCE: Observation data source (default: 'mod16')
        ET_OBS_PATH: Direct path to observation file (overrides source)
        ET_TEMPORAL_AGGREGATION: 'daily_mean' or 'daily_sum'
        ET_USE_QUALITY_CONTROL: Apply QC filtering (default: True)
        OPTIMIZATION_TARGET: 'et' or 'latent_heat'
    """

    # Supported observation sources
    SUPPORTED_SOURCES = {
        'mod16', 'modis', 'modis_et', 'mod16a2',
        'fluxcom', 'fluxcom_et',
        'fluxnet',
        'gleam'
    }

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)

        # Determine ET variable type from config
        self.optimization_target = self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow'
        )
        if self.optimization_target not in ['et', 'latent_heat']:
            eval_var = self.config_dict.get('EVALUATION_VARIABLE', '')
            if eval_var in ['et', 'latent_heat']:
                self.optimization_target = eval_var
            else:
                self.optimization_target = 'et'

        self.variable_name = self.optimization_target

        # Observation source configuration
        self.obs_source = str(self.config_dict.get('ET_OBS_SOURCE', 'mod16')).lower()
        if self.obs_source not in self.SUPPORTED_SOURCES:
            self.logger.warning(
                f"Unknown ET_OBS_SOURCE '{self.obs_source}', defaulting to 'mod16'"
            )
            self.obs_source = 'mod16'

        # Temporal aggregation method
        self.temporal_aggregation = self.config_dict.get('ET_TEMPORAL_AGGREGATION', 'daily_mean')

        # Quality control settings
        self.use_quality_control = self.config_dict.get('ET_USE_QUALITY_CONTROL', True)
        self.max_quality_flag = self.config_dict.get('ET_MAX_QUALITY_FLAG', 2)

        self.logger.info(
            f"Initialized ETEvaluator for {self.optimization_target.upper()} "
            f"evaluation using {self.obs_source.upper()} observations"
        )
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing ET variables."""
        locator = OutputFileLocator(self.logger)
        return locator.find_et_files(sim_dir)
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract ET data from SUMMA simulation files"""
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'et':
                    return self._extract_et_data(ds)
                elif self.optimization_target == 'latent_heat':
                    return self._extract_latent_heat_data(ds)
                else:
                    # Default to ET if target not set correctly
                    return self._extract_et_data(ds)
        except Exception as e:
            self.logger.error(f"Error extracting ET data from {sim_file}: {str(e)}")
            raise
    
    def _extract_et_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract total ET from SUMMA output"""
        if 'scalarTotalET' in ds.variables:
            et_var = ds['scalarTotalET']
            
            # Collapse spatial dimensions
            sim_xr = et_var
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
            
            # Convert units: SUMMA outputs kg m-2 s-1, convert to mm/day
            sim_data = self._convert_et_units(sim_data, from_unit='kg_m2_s', to_unit='mm_day')
            return sim_data
        else:
            return self._sum_et_components(ds)
    
    def _sum_et_components(self, ds: xr.Dataset) -> pd.Series:
        """Sum individual ET components to get total ET"""
        try:
            et_components = {}
            component_vars = {
                'canopy_transpiration': 'scalarCanopyTranspiration',
                'canopy_evaporation': 'scalarCanopyEvaporation', 
                'ground_evaporation': 'scalarGroundEvaporation',
                'snow_sublimation': 'scalarSnowSublimation',
                'canopy_sublimation': 'scalarCanopySublimation'
            }
            total_et = None
            for component_name, var_name in component_vars.items():
                if var_name in ds.variables:
                    component_var = ds[var_name]
                    if len(component_var.shape) > 1:
                        if 'hru' in component_var.dims:
                            if component_var.shape[component_var.dims.index('hru')] == 1:
                                component_data = component_var.isel(hru=0)
                            else:
                                component_data = component_var.mean(dim='hru')
                        else:
                            non_time_dims = [dim for dim in component_var.dims if dim != 'time']
                            if non_time_dims:
                                component_data = component_var.isel({non_time_dims[0]: 0})
                            else:
                                component_data = component_var
                    else:
                        component_data = component_var
                    
                    if total_et is None:
                        total_et = component_data
                    else:
                        total_et = total_et + component_data
            
            if total_et is None:
                raise ValueError("No ET component variables found in SUMMA output")
            
            sim_data = total_et.to_pandas()
            sim_data = self._convert_et_units(sim_data, from_unit='kg_m2_s', to_unit='mm_day')
            return sim_data
        except Exception as e:
            self.logger.error(f"Error summing ET components: {str(e)}")
            raise
    
    def _extract_latent_heat_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract latent heat flux from SUMMA output"""
        if 'scalarLatHeatTotal' in ds.variables:
            lh_var = ds['scalarLatHeatTotal']
            
            # Collapse spatial dimensions
            sim_xr = lh_var
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
            return sim_data
        else:
            raise ValueError("scalarLatHeatTotal not found in SUMMA output")
    
    def _convert_et_units(self, et_data: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert ET units"""
        if from_unit == 'kg_m2_s' and to_unit == 'mm_day':
            # Convert kg m-2 s-1 to mm/day
            return et_data * 86400.0
        elif from_unit == to_unit:
            return et_data
        else:
            return et_data
    
    def get_observed_data_path(self) -> Path:
        """
        Get path to observed ET data based on configured source.

        Supports:
        - MOD16/MODIS: MODIS MOD16A2 processed data
        - FLUXCOM: FLUXCOM gridded ET
        - FluxNet: Tower observations
        - GLEAM: GLEAM ET products
        """
        # Direct path override
        et_obs_path = self.config_dict.get('ET_OBS_PATH')
        if et_obs_path:
            return Path(et_obs_path)

        # MOD16/MODIS ET
        if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_modis_et_processed.csv"
            )

        # FLUXCOM ET
        if self.obs_source in {'fluxcom', 'fluxcom_et'}:
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_fluxcom_et_processed.csv"
            )

        # GLEAM ET
        if self.obs_source == 'gleam':
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_gleam_et_processed.csv"
            )

        # FluxNet
        if self.obs_source == 'fluxnet':
            # Try multiple possible paths for FLUXNET data
            fluxnet_station = self._get_config_value(
                lambda: self.config.evaluation.fluxnet.station,
                default=''
            )
            possible_paths = [
                self.project_dir / "observations" / "et" / "preprocessed" / f"{self.domain_name}_fluxnet_et_processed.csv",
                self.project_dir / "observations" / "energy_fluxes" / "processed" / f"{self.domain_name}_fluxnet_processed.csv",
                self.project_dir / "observations" / "fluxnet" / f"{self.domain_name}_FLUXNET_{fluxnet_station}.csv",
            ]
            for path in possible_paths:
                if path.exists():
                    return path
            # Return default path (will trigger acquisition if not exists)
            return possible_paths[0]

        # Default: FluxNet energy fluxes
        return (
            self.project_dir
            / "observations"
            / "energy_fluxes"
            / "processed"
            / f"{self.domain_name}_fluxnet_processed.csv"
        )
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """
        Identify the ET data column based on observation source.

        Column naming varies by source:
        - MOD16: et_mm_day, ET, et
        - FLUXCOM: et, ET, evapotranspiration
        - FluxNet: ET_from_LE_mm_per_day, LE_F_MDS
        - GLEAM: E, Et, et
        """
        columns_lower = [c.lower() for c in columns]

        if self.optimization_target == 'et':
            # MOD16 column names (highest priority for MOD16 source)
            if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
                for col in columns:
                    if col.lower() in ['et_mm_day', 'et', 'et_daily_mm']:
                        return col

            # General ET column search
            priority_terms = [
                'et_mm_day',  # MOD16 processed
                'et_from_le',  # FluxNet
                'evapotranspiration',
                'et'  # Generic
            ]

            for term in priority_terms:
                for col in columns:
                    if term in col.lower():
                        return col

            # Specific fallbacks
            if 'ET_from_LE_mm_per_day' in columns:
                return 'ET_from_LE_mm_per_day'
            if 'ET' in columns:
                return 'ET'
            if 'et' in columns:
                return 'et'

        elif self.optimization_target == 'latent_heat':
            for col in columns:
                if any(term in col.lower() for term in ['le_f_mds', 'le_', 'latent']):
                    return col
            if 'LE_F_MDS' in columns:
                return 'LE_F_MDS'

        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """
        Load observed ET data with quality control and temporal aggregation.

        Handles different file formats based on observation source:
        - MOD16: Simple CSV with date index and et_mm_day column
        - FluxNet: Complex CSV with multiple columns and QC flags
        - FLUXCOM/GLEAM: Similar to MOD16 format
        """
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.warning(f"Observation file not found: {obs_path}")
                # Try to trigger acquisition based on source
                if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
                    self._try_acquire_mod16_data()
                    if obs_path.exists():
                        self.logger.info(f"MOD16 data acquired: {obs_path}")
                    else:
                        return None
                elif self.obs_source == 'fluxnet':
                    result = self._try_acquire_fluxnet_data()
                    if result and result.exists():
                        obs_path = result
                        self.logger.info(f"FLUXNET data acquired: {obs_path}")
                    else:
                        return None
                else:
                    return None

            # Try loading with date as index first (MOD16 format)
            try:
                obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
                if isinstance(obs_df.index, pd.DatetimeIndex):
                    data_col = self._get_observed_data_column(obs_df.columns)
                    if data_col:
                        obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')
                        obs_data = obs_data.dropna()
                        self.logger.info(f"Loaded {len(obs_data)} ET observations from {obs_path.name}")
                        return obs_data
            except Exception:
                pass

            # Fall back to standard loading
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            if not date_col or not data_col:
                self.logger.warning(f"Could not find date or data columns in {obs_path}")
                return None

            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')

            # Apply quality control (mainly for FluxNet)
            if self.use_quality_control and self.obs_source == 'fluxnet':
                obs_data = self._apply_quality_control(obs_df, obs_data, data_col)

            obs_data = obs_data.dropna()

            # Temporal aggregation (for high-frequency data)
            if self.temporal_aggregation == 'daily_mean':
                obs_daily = obs_data.resample('D').mean()
            elif self.temporal_aggregation == 'daily_sum':
                obs_daily = obs_data.resample('D').sum()
            else:
                obs_daily = obs_data

            self.logger.info(f"Loaded {len(obs_daily)} ET observations from {obs_path.name}")
            return obs_daily.dropna()

        except Exception as e:
            self.logger.error(f"Error loading observed ET data: {str(e)}")
            return None

    def _try_acquire_mod16_data(self):
        """Attempt to acquire MOD16 data if not present."""
        try:
            from symfluence.data.observation.handlers.modis_et import MODISETHandler

            handler = MODISETHandler(self.config, self.logger)
            raw_dir = handler.acquire()
            handler.process(raw_dir)
            self.logger.info("MOD16 ET data acquisition completed")
        except Exception as e:
            self.logger.warning(f"Could not acquire MOD16 data: {e}")

    def _try_acquire_fluxnet_data(self):
        """Attempt to acquire FLUXNET data if not present."""
        try:
            from symfluence.data.acquisition.handlers.fluxnet import FLUXNETETAcquirer

            output_dir = self.project_dir / "observations" / "et" / "preprocessed"
            output_dir.mkdir(parents=True, exist_ok=True)

            acquirer = FLUXNETETAcquirer(self.config, self.logger)
            result_path = acquirer.download(output_dir)
            self.logger.info(f"FLUXNET ET data acquisition completed: {result_path}")
            return result_path
        except Exception as e:
            self.logger.warning(f"Could not acquire FLUXNET data: {e}")
            return None
    
    def _apply_quality_control(self, obs_df: pd.DataFrame, obs_data: pd.Series, data_col: str) -> pd.Series:
        """Apply FluxNet quality control filters"""
        try:
            qc_col = None
            if self.optimization_target == 'et':
                if 'LE_F_MDS_QC' in obs_df.columns:
                    qc_col = 'LE_F_MDS_QC'
            elif self.optimization_target == 'latent_heat':
                if 'LE_F_MDS_QC' in obs_df.columns:
                    qc_col = 'LE_F_MDS_QC'
            
            if qc_col and qc_col in obs_df.columns:
                qc_flags = pd.to_numeric(obs_df[qc_col], errors='coerce')
                quality_mask = qc_flags <= self.max_quality_flag
                obs_data = obs_data[quality_mask]
            
            return obs_data
        except Exception as e:
            self.logger.warning(f"Error applying quality control: {str(e)}")
            return obs_data
    
    def needs_routing(self) -> bool:
        return False
