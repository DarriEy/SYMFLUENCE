#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SYMFLUENCE Model Evaluators

This module provides evaluators for different hydrological variables including:
- Streamflow (routed and non-routed)
- Snow (SWE, SCA, depth)
- Groundwater (depth, GRACE TWS)
- Evapotranspiration (ET, latent heat)
- Soil moisture (point, SMAP, ESA)

Each evaluator handles data loading, processing, and metric calculation for its specific variable.
Refactored from symfluence.utils.optimization.calibration_targets.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod

from symfluence.utils.common import metrics
from symfluence.utils.evaluation.registry import EvaluationRegistry

class ModelEvaluator(ABC):
    """Abstract base class for different evaluation variables (streamflow, snow, etc.)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        
        # Parse time periods
        self.calibration_period = self._parse_date_range(config.get('CALIBRATION_PERIOD', ''))
        self.evaluation_period = self._parse_date_range(config.get('EVALUATION_PERIOD', ''))
        
        # Parse calibration/evaluation timestep
        self.eval_timestep = config.get('CALIBRATION_TIMESTEP', 'native').lower()
        if self.eval_timestep not in ['native', 'hourly', 'daily']:
            self.logger.warning(
                f"Invalid CALIBRATION_TIMESTEP '{self.eval_timestep}'. "
                "Using 'native'. Valid options: 'native', 'hourly', 'daily'"
            )
            self.eval_timestep = 'native'
        
        if self.eval_timestep != 'native':
            self.logger.info(f"Evaluation will use {self.eval_timestep} timestep")
    
    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None, 
                         mizuroute_dir: Optional[Path] = None, 
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for this target.
        
        Args:
            sim: Either a Path to simulation directory or a pre-loaded pd.Series
            obs: Optional pre-loaded pd.Series of observations. If None, loads from file.
            mizuroute_dir: mizuRoute simulation directory (if needed and sim is Path)
            calibration_only: If True, only calculate calibration period metrics
        """
        try:
            # 1. Prepare simulated data
            if isinstance(sim, (str, Path)):
                sim_dir = Path(sim)
                # Determine which simulation directory to use
                if self.needs_routing() and mizuroute_dir:
                    output_dir = mizuroute_dir
                else:
                    output_dir = sim_dir
                
                # Get simulation files
                sim_files = self.get_simulation_files(output_dir)
                if not sim_files:
                    self.logger.error(f"No simulation files found in {output_dir}")
                    return None
                
                # Extract simulated data
                sim_data = self.extract_simulated_data(sim_files)
            else:
                sim_data = sim
                
            if sim_data is None:
                self.logger.error("Failed to extract simulated data")
                return None
            if isinstance(sim_data, os.PathLike):
                sim_path = Path(sim_data)
                if sim_path.is_dir():
                    sim_files = self.get_simulation_files(sim_path)
                else:
                    sim_files = [sim_path]
                if not sim_files:
                    self.logger.error(f"No simulation files found in {sim_path}")
                    return None
                sim_data = self.extract_simulated_data(sim_files)
            if len(sim_data) == 0:
                self.logger.error("Simulated data is empty")
                return None
            
            # 2. Prepare observed data
            if obs is None:
                obs_data = self._load_observed_data()
            else:
                obs_data = obs

            if isinstance(obs_data, os.PathLike):
                obs_data = self._load_observed_data_from_path(Path(obs_data))
                
            if obs_data is None or len(obs_data) == 0:
                self.logger.error("Failed to load observed data")
                return None
            
            # 3. Align time series and calculate metrics
            metrics_dict = {}
            
            # Always calculate metrics for calibration period if available
            if self.calibration_period[0] and self.calibration_period[1]:
                calib_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.calibration_period, "Calib"
                )
                metrics_dict.update(calib_metrics)
            
            # Only calculate evaluation period metrics if requested (final evaluation)
            if not calibration_only and self.evaluation_period[0] and self.evaluation_period[1]:
                eval_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.evaluation_period, "Eval"
                )
                metrics_dict.update(eval_metrics)
            
            # If no specific periods, calculate for full overlap (fallback)
            if not metrics_dict:
                full_metrics = self._calculate_period_metrics(obs_data, sim_data, (None, None), "")
                metrics_dict.update(full_metrics)
            
            return metrics_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {self.__class__.__name__}: {str(e)}")
            return None
    
    @abstractmethod
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get relevant simulation output files for this target"""
        pass
    
    @abstractmethod
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract simulated data from output files"""
        pass
    
    @abstractmethod
    def get_observed_data_path(self) -> Path:
        """Get path to observed data file"""
        pass
    
    @abstractmethod
    def needs_routing(self) -> bool:
        """Whether this target requires mizuRoute routing"""
        pass
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed data from file"""
        try:
            obs_path = self.get_observed_data_path()
            return self._load_observed_data_from_path(obs_path)
            
        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None

    def _load_observed_data_from_path(self, obs_path: Path) -> Optional[pd.Series]:
        """Load observed data from a specific path."""
        if not obs_path.exists():
            self.logger.error(f"Observed data file not found: {obs_path}")
            return None

        obs_df = pd.read_csv(obs_path)

        # Find date and data columns
        date_col = next((col for col in obs_df.columns
                         if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)

        data_col = self._get_observed_data_column(obs_df.columns)

        if not date_col or not data_col:
            self.logger.error(f"Could not identify date/data columns in {obs_path}")
            return None

        # Process data
        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)

        return obs_df[data_col]
    
    @abstractmethod
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the data column in observed data file"""
        pass
        
    def _calculate_period_metrics(self, obs_data: pd.Series, sim_data: pd.Series, 
                                period: Tuple, prefix: str) -> Dict[str, float]:
        """Calculate metrics for a specific time period with explicit filtering"""
        try:
            # EXPLICIT filtering for both datasets (consistent with parallel worker)
            if period[0] and period[1]:
                # Filter observed data to period
                period_mask = (obs_data.index >= period[0]) & (obs_data.index <= period[1])
                obs_period = obs_data[period_mask]
                
                # Explicitly filter simulated data to same period (like parallel worker)
                sim_data.index = sim_data.index.round('h')  # Round first for consistency
                sim_period_mask = (sim_data.index >= period[0]) & (sim_data.index <= period[1])
                sim_period = sim_data[sim_period_mask]
                
                # Log filtering results for debugging
                self.logger.debug(f"{prefix} period filtering: {period[0]} to {period[1]}")
                self.logger.debug(f"{prefix} observed points: {len(obs_period)}")
                self.logger.debug(f"{prefix} simulated points: {len(sim_period)}")
            else:
                obs_period = obs_data
                sim_period = sim_data
                sim_period.index = sim_period.index.round('h')
            
            # Resample to evaluation timestep if specified in config
            if self.eval_timestep != 'native':
                self.logger.info(f"Resampling data to {self.eval_timestep} timestep")
                obs_period = self._resample_to_timestep(obs_period, self.eval_timestep)
                sim_period = self._resample_to_timestep(sim_period, self.eval_timestep)
                
                self.logger.debug(f"After resampling - obs points: {len(obs_period)}, sim points: {len(sim_period)}")
            
            # Find common time indices
            common_idx = obs_period.index.intersection(sim_period.index)
            
            if len(common_idx) == 0:
                self.logger.warning(f"No common time indices for {prefix} period")
                return {}
            
            obs_common = obs_period.loc[common_idx]
            sim_common = sim_period.loc[common_idx]
            
            # Log final aligned data for debugging
            self.logger.debug(f"{prefix} aligned data points: {len(common_idx)}")
            self.logger.debug(f"{prefix} obs range: {obs_common.min():.3f} to {obs_common.max():.3f}")
            self.logger.debug(f"{prefix} sim range: {sim_common.min():.3f} to {sim_common.max():.3f}")
            
            # Calculate metrics
            base_metrics = self._calculate_performance_metrics(obs_common, sim_common)
            
            # Add prefix if specified
            if prefix:
                return {f"{prefix}_{k}": v for k, v in base_metrics.items()}
            else:
                return base_metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating period metrics: {str(e)}")
            return {}
    
    def _resample_to_timestep(self, data: pd.Series, target_timestep: str) -> pd.Series:
        """
        Resample time series data to target timestep
        
        Args:
            data: Time series data with DatetimeIndex
            target_timestep: Target timestep ('hourly' or 'daily')
            
        Returns:
            Resampled time series
        """
        if target_timestep == 'native' or data is None or len(data) == 0:
            return data
        
        try:
            # Infer current frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is None:
                # Try to infer from first few differences
                if len(data) > 1:
                    time_diff = data.index[1] - data.index[0]
                    self.logger.debug(f"Inferred time difference: {time_diff}")
                else:
                    self.logger.warning("Cannot infer frequency from single data point")
                    return data
            else:
                self.logger.debug(f"Inferred frequency: {inferred_freq}")
            
            # Determine current timestep
            time_diff = data.index[1] - data.index[0] if len(data) > 1 else pd.Timedelta(hours=1)
            
            # Check if already at target timestep
            if target_timestep == 'hourly' and pd.Timedelta(minutes=45) <= time_diff <= pd.Timedelta(minutes=75):
                self.logger.debug("Data already at hourly timestep")
                return data
            elif target_timestep == 'daily' and pd.Timedelta(hours=20) <= time_diff <= pd.Timedelta(hours=28):
                self.logger.debug("Data already at daily timestep")
                return data
            
            # Perform resampling
            if target_timestep == 'hourly':
                if time_diff < pd.Timedelta(hours=1):
                    # Upsampling: sub-hourly to hourly (mean aggregation)
                    self.logger.info(f"Aggregating {time_diff} data to hourly using mean")
                    resampled = data.resample('H').mean()
                elif time_diff > pd.Timedelta(hours=1):
                    # Downsampling: daily/coarser to hourly (interpolation)
                    self.logger.info(f"Interpolating {time_diff} data to hourly")
                    # First resample to hourly (creates NaNs)
                    resampled = data.resample('H').asfreq()
                    # Then interpolate
                    resampled = resampled.interpolate(method='time', limit_direction='both')
                else:
                    resampled = data
                    
            elif target_timestep == 'daily':
                if time_diff < pd.Timedelta(days=1):
                    # Upsampling: hourly/sub-daily to daily (mean aggregation)
                    self.logger.info(f"Aggregating {time_diff} data to daily using mean")
                    resampled = data.resample('D').mean()
                elif time_diff > pd.Timedelta(days=1):
                    # Downsampling: weekly/monthly to daily (interpolation)
                    self.logger.info(f"Interpolating {time_diff} data to daily")
                    resampled = data.resample('D').asfreq()
                    resampled = resampled.interpolate(method='time', limit_direction='both')
                else:
                    resampled = data
            else:
                resampled = data
            
            # Remove any NaN values introduced by resampling at edges
            resampled = resampled.dropna()
            
            self.logger.info(
                f"Resampled from {len(data)} to {len(resampled)} points "
                f"(target: {target_timestep})"
            )
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling to {target_timestep}: {str(e)}")
            self.logger.warning("Returning original data without resampling")
            return data

    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics between observed and simulated data"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')

            # Use centralized metrics module for all calculations
            result = metrics.calculate_all_metrics(observed, simulated)

            # Return subset of metrics for compatibility
            return {
                'KGE': result['KGE'],
                'NSE': result['NSE'],
                'RMSE': result['RMSE'],
                'PBIAS': result['PBIAS'],
                'MAE': result['MAE'],
                'r': result['r'],
                'alpha': result['alpha'],
                'beta': result['beta']
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _parse_date_range(self, date_range_str: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse date range string from config"""
        if not date_range_str:
            return None, None
        
        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except Exception as e:
            self.logger.warning(f"Could not parse date range '{date_range_str}': {str(e)}")
        
        return None, None


@EvaluationRegistry.register('ET')
class ETEvaluator(ModelEvaluator):
    """Evapotranspiration evaluator for FluxNet data"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        # Determine ET variable type from config
        self.optimization_target = config.get('OPTIMIZATION_TARGET', 'streamflow')
        if self.optimization_target not in ['et', 'latent_heat']:
            # Fallback/default if used in a context where target isn't explicitly set to ET
            if config.get('EVALUATION_VARIABLE', '') in ['et', 'latent_heat']:
                self.optimization_target = config.get('EVALUATION_VARIABLE')
            
        self.variable_name = self.optimization_target
        
        # Temporal aggregation method for high-frequency FluxNet data
        self.temporal_aggregation = config.get('ET_TEMPORAL_AGGREGATION', 'daily_mean')  # daily_mean, daily_sum
        
        # Quality control settings
        self.use_quality_control = config.get('ET_USE_QUALITY_CONTROL', True)
        self.max_quality_flag = config.get('ET_MAX_QUALITY_FLAG', 2)  # FluxNet QC flags: 0=best, 3=worst
        
        self.logger.info(f"Initialized ETEvaluator for {self.optimization_target.upper()} evaluation")
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing ET variables"""
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        return list(sim_dir.glob("*timestep.nc"))
    
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
            if len(et_var.shape) > 1:
                if 'hru' in et_var.dims:
                    if et_var.shape[et_var.dims.index('hru')] == 1:
                        sim_data = et_var.isel(hru=0).to_pandas()
                    else:
                        sim_data = et_var.mean(dim='hru').to_pandas()
                else:
                    non_time_dims = [dim for dim in et_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = et_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = et_var.to_pandas()
            else:
                sim_data = et_var.to_pandas()
            
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
            if len(lh_var.shape) > 1:
                if 'hru' in lh_var.dims:
                    if lh_var.shape[lh_var.dims.index('hru')] == 1:
                        sim_data = lh_var.isel(hru=0).to_pandas()
                    else:
                        sim_data = lh_var.mean(dim='hru').to_pandas()
                else:
                    non_time_dims = [dim for dim in lh_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = lh_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = lh_var.to_pandas()
            else:
                sim_data = lh_var.to_pandas()
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
        """Get path to observed FluxNet data"""
        return self.project_dir / "observations" / "energy_fluxes" / "processed" / f"{self.domain_name}_fluxnet_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the ET data column in FluxNet file"""
        if self.optimization_target == 'et':
            for col in columns:
                if any(term in col.lower() for term in ['et_from_le', 'et', 'evapotranspiration']):
                    return col
            if 'ET_from_LE_mm_per_day' in columns:
                return 'ET_from_LE_mm_per_day'
        elif self.optimization_target == 'latent_heat':
            for col in columns:
                if any(term in col.lower() for term in ['le_f_mds', 'le_', 'latent']):
                    return col
            if 'LE_F_MDS' in columns:
                return 'LE_F_MDS'
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed FluxNet data with quality control and temporal aggregation"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None
            
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                return None
            
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            if self.use_quality_control:
                obs_data = self._apply_quality_control(obs_df, obs_data, data_col)
            
            obs_data = obs_data.dropna()
            
            if self.temporal_aggregation == 'daily_mean':
                obs_daily = obs_data.resample('D').mean()
            elif self.temporal_aggregation == 'daily_sum':
                obs_daily = obs_data.resample('D').sum() 
            else:
                obs_daily = obs_data
            
            return obs_daily.dropna()
        except Exception as e:
            self.logger.error(f"Error loading observed FluxNet data: {str(e)}")
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
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find timestamp column"""
        timestamp_candidates = ['timestamp', 'TIMESTAMP_START', 'TIMESTAMP_END', 'datetime', 'time']
        for candidate in timestamp_candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            if any(term in col.lower() for term in ['timestamp', 'time', 'date']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        return False


@EvaluationRegistry.register('STREAMFLOW')
class StreamflowEvaluator(ModelEvaluator):
    """Streamflow evaluator"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA timestep files or mizuRoute output files"""
        if 'mizuRoute' in str(sim_dir):
            mizu_files = list(sim_dir.glob("*.nc"))
            if mizu_files:
                return mizu_files
        timestep_files = list(sim_dir.glob("*timestep.nc"))
        if timestep_files:
            return timestep_files

        streamflow_files = list(sim_dir.glob("*_streamflow.nc"))
        if streamflow_files:
            return streamflow_files

        runs_best_files = list(sim_dir.glob("*_runs_best.nc"))
        if runs_best_files:
            return runs_best_files

        return list(sim_dir.glob("*.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from simulation files"""
        sim_file = sim_files[0]
        try:
            if self._is_mizuroute_output(sim_file):
                return self._extract_mizuroute_streamflow(sim_file)
            else:
                return self._extract_summa_streamflow(sim_file)
        except Exception as e:
            self.logger.error(f"Error extracting streamflow data from {sim_file}: {str(e)}")
            raise
    
    def _is_mizuroute_output(self, sim_file: Path) -> bool:
        """Check if file is mizuRoute output"""
        try:
            with xr.open_dataset(sim_file) as ds:
                mizuroute_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'reachID']
                return any(var in ds.variables for var in mizuroute_vars)
        except:
            return False
            
    def _extract_mizuroute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from mizuRoute output"""
        with xr.open_dataset(sim_file) as ds:
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        segment_means = var.mean(dim='time').values
                        outlet_seg_idx = np.argmax(segment_means)
                        result = var.isel(seg=outlet_seg_idx).to_pandas()
                    elif 'reachID' in var.dims:
                        reach_means = var.mean(dim='time').values
                        outlet_reach_idx = np.argmax(reach_means)
                        result = var.isel(reachID=outlet_reach_idx).to_pandas()
                    else:
                        continue
                    return result
            raise ValueError("No suitable streamflow variable found in mizuRoute output")

    def _extract_summa_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from SUMMA output"""
        with xr.open_dataset(sim_file) as ds:
            streamflow_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if len(var.shape) > 1:
                        if 'hru' in var.dims:
                            sim_data = var.isel(hru=0).to_pandas()
                        elif 'gru' in var.dims:
                            sim_data = var.isel(gru=0).to_pandas()
                        else:
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                            else:
                                sim_data = var.to_pandas()
                    else:
                        sim_data = var.to_pandas()
                    
                    catchment_area = self._get_catchment_area()
                    return sim_data * catchment_area
            raise ValueError("No suitable streamflow variable found in SUMMA output")
    
    def _get_catchment_area(self) -> float:
        """Get catchment area for unit conversion"""
        try:
            import geopandas as gpd
            basin_path = self.project_dir / "shapefiles" / "river_basins"
            basin_files = list(basin_path.glob("*.shp"))
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        return total_area
                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                return gdf.geometry.area.sum()
        except Exception as e:
            self.logger.warning(f"Could not calculate catchment area: {str(e)}")
        return 1e6  # 1 km² fallback
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data"""
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default' or not obs_path:
            return self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        return Path(obs_path)
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find streamflow data column"""
        for col in columns:
            if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        """Check if streamflow calibration needs mizuRoute routing"""
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
        
        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        return False


@EvaluationRegistry.register('SOIL_MOISTURE')
class SoilMoistureEvaluator(ModelEvaluator):
    """Soil moisture evaluator"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        self.optimization_target = config.get('OPTIMIZATION_TARGET', 'streamflow')
        if self.optimization_target not in ['sm_point', 'sm_smap', 'sm_esa']:
             if any(x in config.get('EVALUATION_VARIABLE', '') for x in ['sm_', 'soil']):
                self.optimization_target = config.get('EVALUATION_VARIABLE')
        
        self.variable_name = self.optimization_target
        
        if self.optimization_target == 'sm_point':
            self.target_depth = config.get('SM_TARGET_DEPTH', 'auto')
            self.depth_tolerance = config.get('SM_DEPTH_TOLERANCE', 0.05)
        elif self.optimization_target == 'sm_smap':
            self.smap_layer = config.get('SMAP_LAYER', 'surface_sm')
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        elif self.optimization_target == 'sm_esa':
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        
        self.use_quality_control = config.get('SM_USE_QUALITY_CONTROL', True)
        self.min_valid_pixels = config.get('SM_MIN_VALID_PIXELS', 10)
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'sm_point':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_smap':
                    return self._extract_smap_soil_moisture(ds)
                elif self.optimization_target == 'sm_esa':
                    return self._extract_esa_soil_moisture(ds)
                else:
                    return self._extract_point_soil_moisture(ds)
        except Exception as e:
            self.logger.error(f"Error extracting soil moisture data from {sim_file}: {str(e)}")
            raise
    
    def _extract_point_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found")
        soil_moisture_var = ds['mLayerVolFracLiq']
        layer_depths = ds['mLayerDepth']
        
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
                layer_depths_data = layer_depths.isel(hru=0)
            else:
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
                layer_depths_data = layer_depths.mean(dim='hru')
        else:
            soil_moisture_data = soil_moisture_var
            layer_depths_data = layer_depths
        
        target_layer_idx = self._find_target_layer(layer_depths_data)
        layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        sim_data = soil_moisture_data.isel({layer_dim: target_layer_idx}).to_pandas()
        return sim_data
    
    def _find_target_layer(self, layer_depths: xr.DataArray) -> int:
        try:
            if self.target_depth == 'auto':
                return 0
            try:
                target_depth_m = float(self.target_depth)
            except (ValueError, TypeError):
                return 0
            
            if len(layer_depths.shape) >= 2:
                depths_sample = layer_depths.isel(time=0).values
            else:
                depths_sample = layer_depths.values
            
            cumulative_depths = np.cumsum(depths_sample) - depths_sample / 2
            depth_differences = np.abs(cumulative_depths - target_depth_m)
            best_layer_idx = np.argmin(depth_differences)
            return int(best_layer_idx)
        except Exception:
            return 0
    
    def _extract_smap_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
            else:
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
        else:
            soil_moisture_data = soil_moisture_var
        
        if self.smap_layer == 'surface_sm':
            layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
            sim_data = soil_moisture_data.isel({layer_dim: 0}).to_pandas()
        elif self.smap_layer == 'rootzone_sm':
            layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
            top_layers = soil_moisture_data.isel({layer_dim: slice(0, 3)}).mean(dim=layer_dim)
            sim_data = top_layers.to_pandas()
        else:
            raise ValueError(f"Unknown SMAP layer: {self.smap_layer}")
        return sim_data
    
    def _extract_esa_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        soil_moisture_var = ds['mLayerVolFracLiq']
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
            else:
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
        else:
            soil_moisture_data = soil_moisture_var
        
        layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        sim_data = soil_moisture_data.isel({layer_dim: 0}).to_pandas()
        return sim_data
    
    def get_observed_data_path(self) -> Path:
        if self.optimization_target == 'sm_point':
            return self.project_dir / "observations" / "soil_moisture" / "point" / "processed" / f"{self.domain_name}_sm_processed.csv"
        elif self.optimization_target == 'sm_smap':
            return self.project_dir / "observations" / "soil_moisture" / "smap" / "processed" / f"{self.domain_name}_smap_processed.csv"
        elif self.optimization_target == 'sm_esa':
            return self.project_dir / "observations" / "soil_moisture" / "esa_sm" / "processed" / f"{self.domain_name}_esa_processed.csv"
        else:
            # Fallback path if target not perfectly set
             return self.project_dir / "observations" / "soil_moisture" / "processed" / f"{self.domain_name}_sm_processed.csv"

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'sm_point':
            if self.target_depth == 'auto':
                depth_columns = [col for col in columns if col.startswith('sm_')]
                if depth_columns:
                    depths = []
                    for col in depth_columns:
                        try:
                            depth_str = col.split('_')[1]
                            depths.append((float(depth_str), col))
                        except:
                            continue
                    if depths:
                        depths.sort()
                        self.target_depth = str(depths[0][0])
                        return depths[0][1]
            else:
                target_depth_str = str(self.target_depth)
                for col in columns:
                    if col.startswith('sm_') and target_depth_str in col:
                        return col
        elif self.optimization_target == 'sm_smap':
            if self.smap_layer in columns:
                return self.smap_layer
            for col in columns:
                if 'surface_sm' in col.lower() or 'rootzone_sm' in col.lower():
                    return col
        elif self.optimization_target == 'sm_esa':
            for col in columns:
                if any(term in col.lower() for term in ['esa', 'soil_moisture', 'sm']):
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
            
            if self.optimization_target == 'sm_esa':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%d/%m/%Y', errors='coerce')
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            if self.optimization_target == 'sm_smap' and self.use_quality_control:
                if 'valid_px' in obs_df.columns:
                    valid_pixels = pd.to_numeric(obs_df['valid_px'], errors='coerce')
                    quality_mask = valid_pixels >= self.min_valid_pixels
                    obs_series = obs_series[quality_mask]
            
            obs_series = obs_series.dropna()
            
            if hasattr(self, 'temporal_aggregation') and self.temporal_aggregation == 'daily_mean':
                obs_series = obs_series.resample('D').mean().dropna()
            
            return obs_series
        except Exception as e:
            self.logger.error(f"Error loading observed soil moisture data: {str(e)}")
            return None
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        date_candidates = ['timestamp', 'date', 'time', 'DateTime', 'TIMESTAMP_START']
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time', 'timestamp']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        return False


@EvaluationRegistry.register('SNOW')
class SnowEvaluator(ModelEvaluator):
    """Snow evaluator (SWE/SCA)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        self.optimization_target = config.get('OPTIMIZATION_TARGET', config.get('CALIBRATION_VARIABLE', 'streamflow'))
        if self.optimization_target not in ['swe', 'sca', 'snow_depth']:
            calibration_var = config.get('CALIBRATION_VARIABLE', '').lower()
            if 'swe' in calibration_var or 'snow' in calibration_var:
                self.optimization_target = 'swe'
        
        self.variable_name = self.optimization_target
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'swe':
                    return self._extract_swe_data(ds)
                elif self.optimization_target == 'sca':
                    return self._extract_sca_data(ds)
                else:
                    return self._extract_swe_data(ds)
        except Exception as e:
            self.logger.error(f"Error extracting snow data from {sim_file}: {str(e)}")
            raise
    
    def _extract_swe_data(self, ds: xr.Dataset) -> pd.Series:
        if 'scalarSWE' not in ds.variables:
            raise ValueError("scalarSWE variable not found")
        swe_var = ds['scalarSWE']
        
        if len(swe_var.shape) > 1:
            if 'hru' in swe_var.dims:
                if swe_var.shape[swe_var.dims.index('hru')] == 1:
                    sim_data = swe_var.isel(hru=0).to_pandas()
                else:
                    sim_data = swe_var.mean(dim='hru').to_pandas()
            else:
                non_time_dims = [dim for dim in swe_var.dims if dim != 'time']
                if non_time_dims:
                    sim_data = swe_var.isel({non_time_dims[0]: 0}).to_pandas()
                else:
                    sim_data = swe_var.to_pandas()
        else:
            sim_data = swe_var.to_pandas()
        return sim_data
    
    def _extract_sca_data(self, ds: xr.Dataset) -> pd.Series:
        sca_vars = ['scalarGroundSnowFraction', 'scalarSWE']
        for var_name in sca_vars:
            if var_name in ds.variables:
                sca_var = ds[var_name]
                if len(sca_var.shape) > 1:
                    if 'hru' in sca_var.dims:
                        if sca_var.shape[sca_var.dims.index('hru')] == 1:
                            sim_data = sca_var.isel(hru=0).to_pandas()
                        else:
                            sim_data = sca_var.mean(dim='hru').to_pandas()
                    else:
                        non_time_dims = [dim for dim in sca_var.dims if dim != 'time']
                        if non_time_dims:
                            sim_data = sca_var.isel({non_time_dims[0]: 0}).to_pandas()
                        else:
                            sim_data = sca_var.to_pandas()
                else:
                    sim_data = sca_var.to_pandas()
                
                if var_name == 'scalarSWE':
                    swe_threshold = 1.0
                    sim_data = (sim_data > swe_threshold).astype(float)
                
                return sim_data
        raise ValueError("No suitable SCA variable found")
    
    def get_observed_data_path(self) -> Path:
        if self.optimization_target == 'swe':
            return self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv"
        elif self.optimization_target == 'sca':
            return self.project_dir / "observations" / "snow" / "sca" / "processed" / f"{self.domain_name}_sca_processed.csv"
        else:
             return self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'swe':
            for col in columns:
                if any(term in col.lower() for term in ['swe', 'snow_water_equivalent']):
                    return col
            if 'SWE' in columns:
                return 'SWE'
        elif self.optimization_target == 'sca':
            for col in columns:
                if any(term in col.lower() for term in ['snow_cover_ratio', 'sca', 'snow_cover']):
                    return col
            if 'snow_cover_ratio' in columns:
                return 'snow_cover_ratio'
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
            
            if self.optimization_target == 'swe':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%d/%m/%Y', errors='coerce')
            else:
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_series = obs_df[data_col].copy()
            missing_indicators = ['', ' ', 'NA', 'na', 'N/A', 'n/a', 'NULL', 'null', '-', '--', '---', 'missing', 'Missing', 'MISSING']
            for indicator in missing_indicators:
                obs_series = obs_series.replace(indicator, np.nan)
            
            obs_series = pd.to_numeric(obs_series, errors='coerce')
            
            if self.optimization_target == 'swe':
                obs_series = self._convert_swe_units(obs_series)
                obs_series = obs_series[obs_series >= 0]
            
            return obs_series.dropna()
        except Exception as e:
            self.logger.error(f"Error loading observed snow data: {str(e)}")
            return None
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        date_candidates = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 'time', 'Time']
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time']):
                return col
        return None
        
    def _convert_swe_units(self, obs_swe: pd.Series) -> pd.Series:
        """Convert SWE units from inches to kg/m² (mm water equivalent)"""
        # Assume inches if set up that way, otherwise just return
        return obs_swe * 25.4

    def needs_routing(self) -> bool:
        return False


@EvaluationRegistry.register('GROUNDWATER')
class GroundwaterEvaluator(ModelEvaluator):
    """Groundwater evaluator (depth/GRACE)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        self.optimization_target = config.get('OPTIMIZATION_TARGET', 'streamflow')
        if self.optimization_target not in ['gw_depth', 'gw_grace']:
             if 'gw_' in config.get('EVALUATION_VARIABLE', ''):
                self.optimization_target = config.get('EVALUATION_VARIABLE')

        self.variable_name = self.optimization_target
        self.grace_center = config.get('GRACE_PROCESSING_CENTER', 'csr')
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'gw_depth':
                    return self._extract_groundwater_depth(ds)
                elif self.optimization_target == 'gw_grace':
                    return self._extract_total_water_storage(ds)
                else:
                    return self._extract_groundwater_depth(ds)
        except Exception as e:
            self.logger.error(f"Error extracting groundwater data from {sim_file}: {str(e)}")
            raise
    
    def _extract_groundwater_depth(self, ds: xr.Dataset) -> pd.Series:
        if 'scalarAquiferStorage' in ds.variables:
            gw_var = ds['scalarAquiferStorage']
            if len(gw_var.shape) > 1:
                if 'hru' in gw_var.dims:
                    if gw_var.shape[gw_var.dims.index('hru')] == 1:
                        sim_data = gw_var.isel(hru=0).to_pandas()
                    else:
                        sim_data = gw_var.mean(dim='hru').to_pandas()
                else:
                    non_time_dims = [dim for dim in gw_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = gw_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = gw_var.to_pandas()
            else:
                sim_data = gw_var.to_pandas()
            return sim_data.abs()
        else:
            # Simplified derivation
            return pd.Series()
    
    def _extract_total_water_storage(self, ds: xr.Dataset) -> pd.Series:
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
            for component_data in storage_components.values():
                if len(component_data.shape) > 1:
                    if 'hru' in component_data.dims:
                        if component_data.shape[component_data.dims.index('hru')] == 1:
                            component_series = component_data.isel(hru=0)
                        else:
                            component_series = component_data.mean(dim='hru')
                    else:
                        non_time_dims = [dim for dim in component_data.dims if dim != 'time']
                        if non_time_dims:
                            component_series = component_data.isel({non_time_dims[0]: 0})
                        else:
                            component_series = component_data
                else:
                    component_series = component_data
                
                if total_storage is None:
                    total_storage = component_series
                else:
                    total_storage = total_storage + component_series
            
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
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        date_candidates = ['time', 'Time', 'date', 'Date', 'DATE', 'datetime', 'DateTime']
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        return False


@EvaluationRegistry.register('TWS')
class TWSEvaluator(ModelEvaluator):
    """
    Total Water Storage evaluator comparing SUMMA storage to GRACE TWS anomalies.
    Adapted to inherit from ModelEvaluator.
    """
    
    DEFAULT_STORAGE_VARS = [
        'scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage'
    ]
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        self.grace_column = config.get('TWS_GRACE_COLUMN', 'grace_jpl_anomaly')
        self.anomaly_baseline = config.get('TWS_ANOMALY_BASELINE', 'overlap')
        self.unit_conversion = config.get('TWS_UNIT_CONVERSION', 1.0)
        
        storage_str = config.get('TWS_STORAGE_COMPONENTS', '')
        if storage_str:
            self.storage_vars = [v.strip() for v in storage_str.split(',') if v.strip()]
        else:
            self.storage_vars = self.DEFAULT_STORAGE_VARS.copy()
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        patterns = ['*_timestep.nc', '*_day.nc', '*output*.nc', '*.nc']
        for pattern in patterns:
            files = list(sim_dir.glob(pattern))
            if files:
                return [max(files, key=lambda f: f.stat().st_mtime)]
        return []
        
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        output_file = sim_files[0]
        try:
            ds = xr.open_dataset(output_file)
            total_tws = None
            
            for var in self.storage_vars:
                if var in ds.data_vars:
                    data = ds[var].values
                    if 'aquifer' in var.lower():
                        data = data * 1000.0
                    if data.ndim > 1:
                        axes_to_sum = tuple(range(1, data.ndim))
                        data = np.nanmean(data, axis=axes_to_sum)
                    
                    if total_tws is None:
                        total_tws = data.copy()
                    else:
                        total_tws = total_tws + data
            
            if total_tws is None:
                raise ValueError("No storage variables found")
            
            # Find time coordinate
            time_coord = None
            for var in ['time', 'Time', 'datetime']:
                if var in ds.coords or var in ds.dims:
                    time_coord = pd.to_datetime(ds[var].values)
                    break
            
            if time_coord is None:
                raise ValueError("Could not extract time coordinate")
            
            tws_series = pd.Series(total_tws.flatten(), index=time_coord, name='simulated_tws')
            tws_series = tws_series * self.unit_conversion
            ds.close()
            return tws_series
        except Exception as e:
            self.logger.error(f"Error loading SUMMA output: {e}")
            raise
    
    def get_observed_data_path(self) -> Path:
        if 'TWS_OBS_PATH' in self.config:
            return Path(self.config.get('TWS_OBS_PATH'))
        
        obs_dir = self.project_dir / 'observations' / 'grace'
        potential_paths = [
            obs_dir / f'{self.domain_name}_grace_tws_anomaly.csv',
            obs_dir / f'grace_tws_anomaly.csv',
            obs_dir / 'tws_anomaly.csv',
        ]
        
        for path in potential_paths:
            if path.exists():
                return path
        return obs_dir / f'{self.domain_name}_grace_tws_anomaly.csv'

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.grace_column in columns:
            return self.grace_column
        return next((c for c in columns if 'grace' in c.lower()), None)
    
    def needs_routing(self) -> bool:
        return False
        
    def calculate_metrics(self, sim_dir: Path, mizuroute_dir: Optional[Path] = None, 
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """Override to handle anomaly calculation which is specific to TWS"""
        try:
            # Load simulated data (absolute storage)
            sim_files = self.get_simulation_files(sim_dir)
            if not sim_files:
                return None
            sim_tws = self.extract_simulated_data(sim_files)
            
            # Load observed data (anomalies)
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                return None
            
            obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
            col = self._get_observed_data_column(obs_df.columns)
            if not col:
                return None
            
            obs_tws = obs_df[col].dropna()
            
            # Aggregate simulated to monthly means to match GRACE
            sim_monthly = sim_tws.resample('MS').mean()
            
            # Find common period
            common_idx = sim_monthly.index.intersection(obs_tws.index)
            if len(common_idx) == 0:
                return None
            
            sim_matched = sim_monthly.loc[common_idx]
            obs_matched = obs_tws.loc[common_idx]
            
            # Calculate anomaly for simulated data
            # Baseline depends on config, but default to overlapping period mean
            baseline_mean = sim_matched.mean()
            sim_anomaly = sim_matched - baseline_mean
            
            # Calculate metrics
            return self._calculate_performance_metrics(obs_matched, sim_anomaly)
            
        except Exception as e:
            self.logger.error(f"Error calculating TWS metrics: {str(e)}")
            return None

    def get_diagnostic_data(self, sim_dir: Path) -> Dict[str, Any]:
        """
        Get detailed diagnostic data for analysis and plotting.
        
        Returns matched time series, component breakdown, etc.
        """
        sim_files = self.get_simulation_files(sim_dir)
        if not sim_files:
            return {}
        
        sim_tws = self.extract_simulated_data(sim_files)
        
        obs_path = self.get_observed_data_path()
        if not obs_path.exists():
            return {}
            
        obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
        col = self._get_observed_data_column(obs_df.columns)
        if not col:
            return {}
        
        sim_monthly = sim_tws.resample('MS').mean()
        obs_monthly = obs_df[col].copy()
        
        common_index = sim_monthly.index.intersection(obs_monthly.index)
        valid_mask = ~(sim_monthly.loc[common_index].isna() | obs_monthly.loc[common_index].isna())
        
        sim_matched = sim_monthly.loc[common_index][valid_mask]
        obs_matched = obs_monthly.loc[common_index][valid_mask]
        
        baseline_mean = sim_matched.mean()
        sim_anomaly = sim_matched - baseline_mean
        
        return {
            'time': sim_matched.index,
            'sim_tws': sim_matched.values,
            'sim_anomaly': sim_anomaly.values,
            'obs_anomaly': obs_matched.values,
            'grace_column': self.grace_column,
            'grace_all_columns': obs_df.loc[common_index][valid_mask]
        }
