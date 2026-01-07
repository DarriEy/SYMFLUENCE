#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Streamflow Evaluator
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Optional, Dict, Any

from symfluence.evaluation.registry import EvaluationRegistry
from .streamflow import StreamflowEvaluator
from symfluence.core.constants import UnitConversion

@EvaluationRegistry.register('GR_STREAMFLOW')
class GRStreamflowEvaluator(StreamflowEvaluator):
    """Streamflow evaluator for GR models"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get GR output files (CSV for lumped, NetCDF for distributed)"""
        # PRIORITY 1: Check for mizuRoute output in sim_dir or its subdirectories
        # (Needed when evaluating final optimization results)
        mizu_files = list(sim_dir.glob("mizuRoute/*.nc"))
        if not mizu_files:
            mizu_files = list(sim_dir.glob("*.h.*.nc")) # Typical mizuRoute pattern
        if not mizu_files:
            mizu_files = list(sim_dir.glob("mizuRoute/**/*.nc")) # Recursive check
        
        if mizu_files:
            self.logger.debug(f"Found {len(mizu_files)} mizuRoute files in {sim_dir}")
            return mizu_files

        # PRIORITY 2: Lumped mode output
        lumped_file = sim_dir / 'GR_results.csv'
        if lumped_file.exists():
            return [lumped_file]
            
        # PRIORITY 3: Distributed mode output (fallback if routing not used/found)
        experiment_id = self.config.get('EXPERIMENT_ID')
        dist_file = sim_dir / f"{self.domain_name}_{experiment_id}_runs_def.nc"
        if dist_file.exists():
            self.logger.warning(f"No routed output found. Falling back to raw GR output: {dist_file}")
            return [dist_file]
            
        # Fallback to generic NetCDF search
        return super().get_simulation_files(sim_dir)
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from GR output files"""
        sim_file = sim_files[0]
        self.logger.info(f"Extracting simulated streamflow from: {sim_file}")
        
        if sim_file.suffix == '.csv':
            return self._extract_lumped_gr_streamflow(sim_file)
        else:
            # Check if it's mizuRoute output or GR distributed output
            if self._is_mizuroute_output(sim_file):
                sim_data = self._extract_mizuroute_streamflow(sim_file)
                
                # ENHANCED: Check units and convert if needed (mm/day -> cms)
                try:
                    with xr.open_dataset(sim_file) as ds:
                        # Find the variable that was extracted
                        streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                        var_name = next((v for v in streamflow_vars if v in ds.variables), None)
                        
                        if var_name:
                            units = ds[var_name].attrs.get('units', '').lower()
                            if 'm3' in units or 'cms' in units:
                                # Already in cms
                                return sim_data
                            else:
                                # Likely in mm/day or similar, need conversion
                                self.logger.info(f"Converting mizuRoute output units ({units}) to cms")
                                area_m2 = self._get_catchment_area()
                                area_km2 = area_m2 / 1e6
                                return sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
                except Exception as e:
                    self.logger.warning(f"Could not determine units from mizuRoute output: {e}")
                
                return sim_data
            else:
                return self._extract_distributed_gr_streamflow(sim_file)

    def _extract_lumped_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR lumped CSV output"""
        df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
        # GR4J output is in mm/day. Convert to cms.
        area_m2 = self._get_catchment_area()
        area_km2 = area_m2 / 1e6
        simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
        return simulated_streamflow

    def _extract_distributed_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR distributed NetCDF output"""
        with xr.open_dataset(sim_file) as ds:
            # Determine which variable to use (respect config)
            routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            if routing_var in ('default', None, ''):
                routing_var = 'q_routed'
            
            # q_routed is in mm/day or m/s, aggregated across HRUs
            if routing_var in ds.variables:
                var = ds[routing_var]
                # Use mean across GRUs for depth-based runoff
                sim_data = var.mean(dim='gru').to_pandas()
                
                units = var.attrs.get('units', 'mm/d').lower()
                area_m2 = self._get_catchment_area()
                
                if 'm/s' in units or 'm s-1' in units:
                    # Convert m/s to m3/s: m/s * area_m2
                    self.logger.info(f"Extracting GR distributed output in m/s")
                    return sim_data * area_m2
                else:
                    # Assume mm/day
                    self.logger.info(f"Extracting GR distributed output in mm/day")
                    area_km2 = area_m2 / 1e6
                    return sim_data * area_km2 / UnitConversion.MM_DAY_TO_CMS
            else:
                self.logger.warning(f"Variable '{routing_var}' not found in {sim_file}. Trying fallback 'q_routed'.")
                if 'q_routed' in ds.variables:
                    # Fallback to q_routed if available
                    var = ds['q_routed']
                    sim_data = var.mean(dim='gru').to_pandas()
                    area_m2 = self._get_catchment_area()
                    return sim_data * area_m2 if 'm/s' in var.attrs.get('units', '').lower() else sim_data * (area_m2 / 1e6) / UnitConversion.MM_DAY_TO_CMS
                
                raise ValueError(f"Neither '{routing_var}' nor 'q_routed' found in {sim_file}. Available: {list(ds.variables)}")

    def _get_catchment_area(self) -> float:
        """Get catchment area in m2, prioritized for GR"""
        # Priority 1: Try basin shapefile
        try:
            import geopandas as gpd
            # Standard catchment location for GR
            domain_name = self.config.get('DOMAIN_NAME')
            project_dir = self.project_dir
            
            catchment_path = project_dir / 'shapefiles' / 'catchment'
            discretization = self.config.get('DOMAIN_DISCRETIZATION', 'elevation')
            
            catchment_name = self.config.get('CATCHMENT_SHP_NAME')
            if not catchment_name or catchment_name == 'default':
                catchment_name = f"{domain_name}_HRUs_{discretization}.shp"
            
            catchment_file = catchment_path / catchment_name
            if catchment_file.exists():
                gdf = gpd.read_file(catchment_file)
                if 'GRU_area' in gdf.columns:
                    area_m2 = gdf['GRU_area'].sum()
                    if 0 < area_m2 < 1e12:
                        return float(area_m2)
                
                # Fallback: calculate from geometry
                if gdf.crs and not gdf.crs.is_geographic:
                    area_m2 = gdf.geometry.area.sum()
                else:
                    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                    area_m2 = gdf_utm.geometry.area.sum()
                return float(area_m2)
        except Exception as e:
            self.logger.debug(f"Error calculating area from shapefile: {e}")

        # Fallback to base logic
        return super()._get_catchment_area()
