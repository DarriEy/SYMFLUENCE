#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Snow (SWE/SCA) Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional

from symfluence.utils.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

@EvaluationRegistry.register('SCA')
@EvaluationRegistry.register('SWE')
@EvaluationRegistry.register('SNOW')
class SnowEvaluator(ModelEvaluator):
    """Snow evaluator (SWE/SCA)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger, **kwargs):
        # Allow target override from kwargs (for multivariate calibration)
        self._target_override = kwargs.get('target')
        super().__init__(config, project_dir, logger)
        
        # Determine variable target: swe or sca
        self.optimization_target = self._target_override
        if not self.optimization_target:
            self.optimization_target = config.get('OPTIMIZATION_TARGET', config.get('CALIBRATION_VARIABLE', 'swe')).lower()
            
        if self.optimization_target not in ['swe', 'sca', 'snow_depth']:
            calibration_var = config.get('CALIBRATION_VARIABLE', '').lower()
            if 'swe' in calibration_var or 'snow' in calibration_var:
                self.optimization_target = 'swe'
            elif 'sca' in calibration_var:
                self.optimization_target = 'sca'
        
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
    
    def calculate_metrics(self, simulated_data: pd.Series, calibration_only: bool = True, **kwargs) -> Dict[str, float]:
        """
        Calculate performance metrics for simulated snow data.
        
        Args:
            simulated_data: Simulated snow series
            calibration_only: If True, only use calibration period
            
        Returns:
            Dictionary of metrics
        """
        # Ensure we are using the correct target if provided in kwargs
        if 'target' in kwargs:
            self.optimization_target = kwargs['target'].lower()
            self.variable_name = self.optimization_target

        return super().calculate_metrics(simulated_data, calibration_only, **kwargs)

    def get_observed_data_path(self) -> Path:
        """Get path to preprocessed observed snow data."""
        if self.optimization_target == 'swe':
            # Check for generic SNOW or specific SWE
            paths = [
                self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_snow_processed.csv",
                self.project_dir / "observations" / "snow" / "swe" / "preprocessed" / f"{self.domain_name}_swe_processed.csv"
            ]
            for p in paths:
                if p.exists(): return p
            return paths[0]
        elif self.optimization_target == 'sca':
            return self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_modis_snow_processed.csv"
        else:
             return self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_snow_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'swe':
            for col in columns:
                if any(term in col.lower() for term in ['swe', 'snow_water_equivalent', 'value']):
                    return col
        elif self.optimization_target == 'sca':
            for col in columns:
                if any(term in col.lower() for term in ['snow_cover_ratio', 'sca', 'snow_cover']):
                    return col
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.warning(f"Snow observation file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.warning(f"Could not find required columns in {obs_path}. Need Date and data column.")
                return None
            
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            obs_series = obs_df[data_col].copy()
            missing_indicators = ['', ' ', 'NA', 'na', 'N/A', 'n/a', 'NULL', 'null', '-', '--', '---', 'missing', 'Missing', 'MISSING']
            for indicator in missing_indicators:
                obs_series = obs_series.replace(indicator, np.nan)
            
            obs_series = pd.to_numeric(obs_series, errors='coerce')
            
            if self.optimization_target == 'swe':
                # Convert if data is likely in inches (common for NRCS)
                # If values are large, they are likely already in mm
                if obs_series.max() < 250: # 250 inches is a huge amount of SWE
                    obs_series = self._convert_swe_units(obs_series)
                obs_series = obs_series[obs_series >= 0]

            return obs_series.dropna()
        except Exception as e:
            self.logger.error(f"Error loading observed snow data: {str(e)}")
            return None

    def _convert_swe_units(self, obs_swe: pd.Series) -> pd.Series:
        """Convert SWE units from inches to kg/m² (mm water equivalent)"""
        # Assume inches if set up that way, otherwise just return
        return obs_swe * 25.4

    def needs_routing(self) -> bool:
        return False
