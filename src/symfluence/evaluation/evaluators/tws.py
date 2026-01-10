#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Total Water Storage (TWS) Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('TWS')
class TWSEvaluator(ModelEvaluator):
    """
    Total Water Storage evaluator comparing SUMMA storage to GRACE TWS anomalies.
    Adapted to inherit from ModelEvaluator.
    """

    DEFAULT_STORAGE_VARS = [
        'scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage'
    ]

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)

        self.grace_column = self.config_dict.get('TWS_GRACE_COLUMN', 'grace_jpl_anomaly')
        self.anomaly_baseline = self.config_dict.get('TWS_ANOMALY_BASELINE', 'overlap')
        self.unit_conversion = self.config_dict.get('TWS_UNIT_CONVERSION', 1.0)

        storage_str = self.config_dict.get('TWS_STORAGE_COMPONENTS', '')
        if storage_str:
            self.storage_vars = [v.strip() for v in storage_str.split(',') if v.strip()]
        else:
            self.storage_vars = self.DEFAULT_STORAGE_VARS.copy()
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get simulation files containing storage variables for TWS calculation."""
        locator = OutputFileLocator(self.logger)
        # TWS needs the most recent file with storage components
        most_recent = locator.get_most_recent(sim_dir, 'tws')
        return [most_recent] if most_recent else []
        
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        output_file = sim_files[0]

        # Try to find a file with storage variables
        # For glacier mode, glacMass4AreaChange is often only in timestep files
        output_dir = output_file.parent
        potential_files = [
            output_file,  # Original file
        ]
        # Add daily file alternatives (storage vars are often in daily output)
        for f in output_dir.glob('*_day.nc'):
            if f not in potential_files:
                potential_files.insert(0, f)  # Prefer daily files
        # Also check timestep files for glacier variables like glacMass4AreaChange
        for f in output_dir.glob('*timestep.nc'):
            if f not in potential_files:
                potential_files.append(f)

        # Find a file that has ALL requested storage variables (if possible)
        # or at least the most critical ones including glacier mass
        best_file = None
        best_var_count = 0
        for candidate_file in potential_files:
            try:
                with xr.open_dataset(candidate_file) as test_ds:
                    matching_vars = [v for v in self.storage_vars if v in test_ds.data_vars]
                    if len(matching_vars) > best_var_count:
                        best_var_count = len(matching_vars)
                        best_file = candidate_file
                        self.logger.debug(f"Found {len(matching_vars)}/{len(self.storage_vars)} vars in {candidate_file.name}")
                    # If we find all variables, use this file
                    if len(matching_vars) == len(self.storage_vars):
                        break
            except:
                continue

        if best_file:
            output_file = best_file
            self.logger.debug(f"Using {output_file.name} for TWS extraction ({best_var_count} storage vars)")

        try:
            ds = xr.open_dataset(output_file)
            total_tws = None

            for var in self.storage_vars:
                if var in ds.data_vars:
                    data = ds[var].values.astype(float)

                    # Replace fill values (-9999) with NaN
                    # SUMMA uses -9999 for missing/invalid data in some domains
                    data = np.where(data < -999, np.nan, data)

                    # Unit conversion: aquifer storage is in meters, others usually in mm
                    if 'aquifer' in var.lower():
                        self.logger.debug(f"Converting {var} from meters to mm")
                        data = data * 1000.0

                    if data.ndim > 1:
                        axes_to_sum = tuple(range(1, data.ndim))
                        data = np.nanmean(data, axis=axes_to_sum)

                    if total_tws is None:
                        total_tws = data.copy()
                    else:
                        # Use nansum behavior: NaN + value = value
                        total_tws = np.where(np.isnan(total_tws), data,
                                            np.where(np.isnan(data), total_tws, total_tws + data))
                else:
                    self.logger.warning(f"Storage variable {var} not found in {output_file}")
            
            if total_tws is None:
                raise ValueError(f"No storage variables {self.storage_vars} found in {output_file}")
            
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
        tws_obs_path = self.config_dict.get('TWS_OBS_PATH')
        if tws_obs_path:
            return Path(tws_obs_path)

        # Search in multiple possible locations for GRACE data
        # Support both observations/grace and observations/storage/grace paths
        obs_base = self.project_dir / 'observations'
        search_dirs = [
            obs_base / 'storage' / 'grace',  # Ashley's glacier domain setup
            obs_base / 'grace',               # Standard location
        ]

        potential_patterns = [
            f'{self.domain_name}_HRUs_GRUs_grace_tws_anomaly.csv',  # HRU-specific format
            f'{self.domain_name}_HRUs_elevation_grace_tws_anomaly_by_hru.csv',  # Elevation-based HRU
            f'{self.domain_name}_grace_tws_processed.csv',
            f'{self.domain_name}_grace_tws_anomaly.csv',
            'grace_tws_anomaly.csv',
            'tws_anomaly.csv',
        ]

        for obs_dir in search_dirs:
            if not obs_dir.exists():
                continue
            preprocessed_dir = obs_dir / 'preprocessed'
            for pattern in potential_patterns:
                for search_dir in [preprocessed_dir, obs_dir]:
                    path = search_dir / pattern
                    if path.exists():
                        return path

        # Fallback to default
        return obs_base / 'grace' / 'preprocessed' / f'{self.domain_name}_grace_tws_processed.csv'

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.grace_column in columns:
            return self.grace_column
        # Fallback to known columns
        for fallback in ['grace_jpl_anomaly', 'grace_csr_anomaly', 'grace_gsfc_anomaly']:
            if fallback in columns:
                return fallback
        return next((c for c in columns if 'grace' in c.lower() and 'anomaly' in c.lower()), None)
    
    def needs_routing(self) -> bool:
        return False
        
    def calculate_metrics(self, sim_dir: Path, mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """Override to handle anomaly calculation which is specific to TWS"""
        try:
            sim_dir = Path(sim_dir)
            self.logger.debug(f"[TWS] Starting metrics calculation for {sim_dir}")

            # Load simulated data (absolute storage)
            sim_files = self.get_simulation_files(sim_dir)
            if not sim_files:
                self.logger.error(f"[TWS] No simulation files found in {sim_dir}")
                return None

            sim_tws = self.extract_simulated_data(sim_files)
            self.logger.debug(f"[TWS] Extracted simulated TWS: {len(sim_tws)} points")

            # Load observed data (anomalies)
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"[TWS] Observed data path does not exist: {obs_path}")
                return None

            obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
            col = self._get_observed_data_column(obs_df.columns)
            if not col:
                self.logger.error(f"[TWS] Could not find GRACE column in {obs_path}. Available: {list(obs_df.columns)}")
                return None

            obs_tws = obs_df[col].dropna()

            # Aggregate simulated to monthly means to match GRACE
            sim_monthly = sim_tws.resample('MS').mean()

            # Check if we have any simulated data
            if len(sim_monthly) == 0:
                self.logger.error(f"[TWS] No simulated TWS data available for metrics calculation")
                return None

            # Find common period
            common_idx = sim_monthly.index.intersection(obs_tws.index)
            if len(common_idx) == 0:
                sim_start, sim_end = sim_monthly.index[0], sim_monthly.index[-1]
                obs_start, obs_end = obs_tws.index[0], obs_tws.index[-1]
                self.logger.error(f"[TWS] No overlapping period between simulation ({sim_start} to {sim_end}) and observations ({obs_start} to {obs_end})")
                return None

            self.logger.debug(f"[TWS] Found common period with {len(common_idx)} points")
            sim_matched = sim_monthly.loc[common_idx]
            obs_matched = obs_tws.loc[common_idx]

            # Calculate anomaly for simulated data
            baseline_mean = sim_matched.mean()
            sim_anomaly = sim_matched - baseline_mean

            # Calculate metrics
            metrics = self._calculate_performance_metrics(obs_matched, sim_anomaly)
            self.logger.debug(f"[TWS] Calculated metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"[TWS] Error calculating TWS metrics: {str(e)}")
            import traceback
            self.logger.debug(f"[TWS] Traceback: {traceback.format_exc()}")
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
