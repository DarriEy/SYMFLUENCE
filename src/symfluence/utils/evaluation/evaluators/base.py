#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Model Evaluator

This module provides the abstract base class for different evaluation variables.
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

class ModelEvaluator(ABC):
    """Abstract base class for different evaluation variables (streamflow, snow, etc.)"""
    
    def __init__(
        self,
        config: Dict,
        project_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.project_dir = project_dir or Path(".")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
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

    def align_series(self, sim: pd.Series, obs: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align simulation and observation series after dropping spinup years."""
        spinup_years = self.config.get('EVALUATION_SPINUP_YEARS', 0)
        try:
            spinup_years = int(float(spinup_years))
        except (TypeError, ValueError):
            spinup_years = 0
        spinup_years = max(0, spinup_years)

        if sim.empty or obs.empty:
            return sim, obs

        common_start = max(sim.index.min(), obs.index.min())
        cutoff = common_start + pd.DateOffset(years=spinup_years) if spinup_years else common_start

        sim_trimmed = sim[sim.index >= cutoff]
        obs_trimmed = obs[obs.index >= cutoff]

        common_idx = sim_trimmed.index.intersection(obs_trimmed.index)
        if not common_idx.empty:
            sim_trimmed = sim_trimmed.loc[common_idx]
            obs_trimmed = obs_trimmed.loc[common_idx]
        else:
            self.logger.warning("No overlapping indices after alignment; returning trimmed series")

        return sim_trimmed, obs_trimmed

    def _collapse_spatial_dims(self, data_array: xr.DataArray, aggregate: str = 'mean') -> pd.Series:
        """
        Collapse spatial dimensions from xarray DataArray to pandas Series.

        Handles common spatial dimension patterns in SUMMA/FUSE/NGEN output:
        - Single HRU/GRU: select index 0
        - Multiple HRU/GRU: aggregate (mean by default)
        - Other spatial dims: select first or aggregate

        Args:
            data_array: xarray DataArray with time and possibly spatial dimensions
            aggregate: Aggregation method for multiple spatial units ('mean', 'sum', 'first')

        Returns:
            pandas Series with time index
        """
        spatial_dims = ['hru', 'gru', 'param_set', 'latitude', 'longitude', 'seg', 'reachID']

        result = data_array

        for dim in spatial_dims:
            if dim in result.dims:
                dim_size = result.shape[result.dims.index(dim)]
                if dim_size == 1:
                    result = result.isel({dim: 0})
                elif aggregate == 'mean':
                    result = result.mean(dim=dim)
                elif aggregate == 'sum':
                    result = result.sum(dim=dim)
                elif aggregate == 'first':
                    result = result.isel({dim: 0})

        # Handle any remaining non-time dimensions
        non_time_dims = [dim for dim in result.dims if dim != 'time']
        for dim in non_time_dims:
            dim_size = result.shape[result.dims.index(dim)]
            if dim_size == 1:
                result = result.isel({dim: 0})
            elif aggregate == 'mean':
                result = result.mean(dim=dim)
            elif aggregate == 'sum':
                result = result.sum(dim=dim)
            else:
                result = result.isel({dim: 0})

        return result.to_pandas()

    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """
        Find timestamp/date column in a DataFrame.

        Searches for common date column names used across different data sources.

        Args:
            columns: List of column names from DataFrame

        Returns:
            Name of date column, or None if not found
        """
        # Priority order for timestamp column candidates
        timestamp_candidates = [
            'timestamp', 'TIMESTAMP_START', 'TIMESTAMP_END',
            'datetime', 'DateTime', 'time', 'Time',
            'date', 'Date', 'DATE'
        ]

        # First check exact matches
        for candidate in timestamp_candidates:
            if candidate in columns:
                return candidate

        # Then check partial matches
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['timestamp', 'datetime', 'date', 'time']):
                return col

        return None
