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
from typing import List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .streamflow import StreamflowEvaluator
from symfluence.core.constants import UnitConversion

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('GR_STREAMFLOW')
class GRStreamflowEvaluator(StreamflowEvaluator):
    """Streamflow evaluator for GR models"""

    def _load_observed_data(self) -> Optional[pd.Series]:
        """
        Load observed data and resample to daily frequency (matching GR output).
        GR outputs daily values, so observations must be aggregated to daily for proper comparison.
        """
        try:
            obs_path = self.get_observed_data_path()
            obs_series = self._load_observed_data_from_path(obs_path)

            if obs_series is not None:
                # Resample to daily frequency (mean) to match GR output frequency
                # This matches the behavior in gr_worker._get_observed_streamflow() line 310
                obs_daily = obs_series.resample('D').mean()
                self.logger.info(f"Resampled observations from {len(obs_series)} to {len(obs_daily)} daily values")
                return obs_daily

            return obs_series

        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get GR output files (CSV for lumped, NetCDF for distributed)."""
        locator = OutputFileLocator(self.logger)
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default=None
        )
        return locator.find_gr_output(sim_dir, self.domain_name, experiment_id)
    
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
            routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
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
            project_dir = self.project_dir

            # Robust path resolution handling 'default' values
            c_path = self.config_dict.get('CATCHMENT_PATH', 'default')
            if c_path == 'default' or not c_path:
                catchment_path = project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(c_path)

            discretization = self._get_config_value(
                lambda: self.config.domain.discretization,
                default='elevation'
            )

            c_name = self.config_dict.get('CATCHMENT_SHP_NAME', 'default')
            if not c_name or c_name == 'default':
                catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
            else:
                catchment_name = c_name
            
            catchment_file = catchment_path / catchment_name
            if catchment_file.exists():
                gdf = gpd.read_file(catchment_file)
                if 'GRU_area' in gdf.columns:
                    area_m2 = gdf['GRU_area'].sum()
                    if 0 < area_m2 < 1e12:
                        self.logger.debug(f"Catchment area from GRU_area: {area_m2:.2f} m2")
                        return float(area_m2)
                
                # Fallback: calculate from geometry
                if gdf.crs and not gdf.crs.is_geographic:
                    area_m2 = gdf.geometry.area.sum()
                else:
                    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                    area_m2 = gdf_utm.geometry.area.sum()
                
                self.logger.debug(f"Catchment area calculated from geometry: {area_m2:.2f} m2")
                return float(area_m2)
            else:
                self.logger.warning(f"Catchment file not found: {catchment_file}")
        except Exception as e:
            self.logger.debug(f"Error calculating area from shapefile: {e}")

        # Fallback to base logic
        area_m2 = super()._get_catchment_area()
        self.logger.debug(f"Catchment area from base fallback: {area_m2:.2f} m2")
        return area_m2
