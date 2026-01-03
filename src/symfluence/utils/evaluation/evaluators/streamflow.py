#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamflow Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Optional

from symfluence.utils.evaluation.registry import EvaluationRegistry
from .base import ModelEvaluator

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
