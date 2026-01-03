#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evapotranspiration (ET) Evaluator
"""

import logging
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional

from symfluence.utils.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

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
    
    def needs_routing(self) -> bool:
        return False
