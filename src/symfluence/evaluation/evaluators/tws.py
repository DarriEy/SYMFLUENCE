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
