#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Groundwater Evaluator
"""

import logging
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional

from symfluence.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

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

    def needs_routing(self) -> bool:
        return False
