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
from typing import List, Dict, Optional, Any

from symfluence.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

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

        # Try to find a file with storage variables - prefer daily files for TWS
        output_dir = output_file.parent
        potential_files = [
            output_file,  # Original file
        ]
        # Add daily file alternatives (storage vars are often in daily output)
        for f in output_dir.glob('*_day.nc'):
            if f not in potential_files:
                potential_files.insert(0, f)  # Prefer daily files

        # Find a file that has at least one storage variable
        for candidate_file in potential_files:
            try:
                with xr.open_dataset(candidate_file) as test_ds:
                    if any(var in test_ds.data_vars for var in self.storage_vars):
                        output_file = candidate_file
                        self.logger.info(f"Using {output_file.name} for TWS extraction")
                        break
            except:
                continue

        try:
            ds = xr.open_dataset(output_file)
            total_tws = None
            
            for var in self.storage_vars:
                if var in ds.data_vars:
                    data = ds[var].values
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
                        total_tws = total_tws + data
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
        if 'TWS_OBS_PATH' in self.config:
            return Path(self.config.get('TWS_OBS_PATH'))

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
        import sys
        try:
            sys.stderr.write(f"[TWS] Starting metrics calculation for {sim_dir}\n")
            sys.stderr.flush()
            # Load simulated data (absolute storage)
            sim_files = self.get_simulation_files(sim_dir)
            if not sim_files:
                sys.stderr.write(f"[TWS] No simulation files found in {sim_dir}\n")
                sys.stderr.flush()
                self.logger.error(f"[TWS] No simulation files found in {sim_dir}")
                return None
            
            sim_tws = self.extract_simulated_data(sim_files)
            sys.stderr.write(f"[TWS] Extracted simulated TWS: {len(sim_tws)} points\n")
            sys.stderr.flush()
            
            # Load observed data (anomalies)
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                sys.stderr.write(f"[TWS] Observed data path does not exist: {obs_path}\n")
                sys.stderr.flush()
                self.logger.error(f"[TWS] Observed data path does not exist: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
            col = self._get_observed_data_column(obs_df.columns)
            if not col:
                sys.stderr.write(f"[TWS] Could not find GRACE column in {obs_path}. Available: {list(obs_df.columns)}\n")
                sys.stderr.flush()
                self.logger.error(f"[TWS] Could not find GRACE column in {obs_path}. Available: {list(obs_df.columns)}")
                return None
            
            obs_tws = obs_df[col].dropna()
            
            # Aggregate simulated to monthly means to match GRACE
            sim_monthly = sim_tws.resample('MS').mean()
            
            # Find common period
            common_idx = sim_monthly.index.intersection(obs_tws.index)
            if len(common_idx) == 0:
                sim_start, sim_end = sim_monthly.index[0], sim_monthly.index[-1]
                obs_start, obs_end = obs_tws.index[0], obs_tws.index[-1]
                sys.stderr.write(f"[TWS] No overlapping period between simulation ({sim_start} to {sim_end}) and observations ({obs_start} to {obs_end})\n")
                sys.stderr.flush()
                self.logger.error(f"[TWS] No overlapping period between simulation ({sim_start} to {sim_end}) and observations ({obs_start} to {obs_end})")
                return None
            
            sys.stderr.write(f"[TWS] Found common period with {len(common_idx)} points\n")
            sys.stderr.flush()
            sim_matched = sim_monthly.loc[common_idx]
            obs_matched = obs_tws.loc[common_idx]
            
            # Calculate anomaly for simulated data
            baseline_mean = sim_matched.mean()
            sim_anomaly = sim_matched - baseline_mean
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(obs_matched, sim_anomaly)
            sys.stderr.write(f"[TWS] Calculated metrics: {metrics}\n")
            sys.stderr.flush()
            return metrics
            
        except Exception as e:
            sys.stderr.write(f"[TWS] FATAL EXCEPTION: {str(e)}\n")
            import traceback
            sys.stderr.write(traceback.format_exc() + "\n")
            sys.stderr.flush()
            self.logger.error(f"[TWS] Error calculating TWS metrics: {str(e)}")
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
