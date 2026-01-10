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
from typing import List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('STREAMFLOW')
class StreamflowEvaluator(ModelEvaluator):
    """Streamflow evaluator"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA timestep files or mizuRoute output files."""
        locator = OutputFileLocator(self.logger)
        return locator.find_streamflow_files(sim_dir)
    
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
                    
                    units = var.attrs.get('units', 'unknown')
                    self.logger.debug(f"Found streamflow variable {var_name} with units: '{units}'")
                    
                    # Unit conversion: Mass flux (kg m-2 s-1) to Volume flux (m s-1)
                    # Check for explicit 'kg' units OR unreasonably high values (> 1e-6 m/s = > 86 mm/day mean)
                    # which indicates the value is likely mass flux but mislabeled as m/s.
                    is_mass_flux = False
                    if 'units' in var.attrs and 'kg' in var.attrs['units'] and 's-1' in var.attrs['units']:
                        is_mass_flux = True
                    elif float(var.mean()) > 1e-6:
                        self.logger.debug(f"Variable {var_name} mean ({float(var.mean()):.2e}) is unreasonably high for m/s. Assuming mislabeled mass flux.")
                        is_mass_flux = True
                    
                    if is_mass_flux:
                        self.logger.debug(f"Converting {var_name} from mass flux to volume flux (dividing by 1000)")
                        var = var / 1000.0  # Divide by density of water
                    
                    # Check if we need spatial aggregation
                    if len(var.shape) > 1 and any(d in var.dims for d in ['hru', 'gru']):
                        try:
                            # Try area-weighted aggregation first
                            attrs_file = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
                            if attrs_file.exists():
                                with xr.open_dataset(attrs_file) as attrs:
                                    # Handle HRU dimension
                                    if 'hru' in var.dims and 'HRUarea' in attrs:
                                        areas = attrs['HRUarea']
                                        if areas.sizes['hru'] == var.sizes['hru']:
                                            total_area = float(areas.values.sum())
                                            self.logger.debug(f"Performing area-weighted aggregation for {var_name} (HRU). Total area: {total_area:.1f} m²")
                                            # Calculate total discharge in m³/s: sum(runoff_i * area_i)
                                            weighted_runoff = (var * areas).sum(dim='hru')
                                            return weighted_runoff.to_pandas()
                                    
                                    # Handle GRU dimension
                                    elif 'gru' in var.dims and 'GRUarea' in attrs:
                                        areas = attrs['GRUarea']
                                        if areas.sizes['gru'] == var.sizes['gru']:
                                            total_area = float(areas.values.sum())
                                            self.logger.debug(f"Performing area-weighted aggregation for {var_name} (GRU). Total area: {total_area:.1f} m²")
                                            weighted_runoff = (var * areas).sum(dim='gru')
                                            return weighted_runoff.to_pandas()
                                            
                                    # Fallback if specific area variable missing but HRUarea available for GRU dim (common in lumped)
                                    elif 'gru' in var.dims and 'HRUarea' in attrs:
                                         # If 1:1 mapping or if we can infer
                                         if attrs.sizes['hru'] == var.sizes['gru']:
                                             areas = attrs['HRUarea'] # Assuming 1:1 mapping for lumped
                                             total_area = float(areas.values.sum())
                                             self.logger.debug(f"Performing area-weighted aggregation for {var_name} (GRU using HRUarea). Total area: {total_area:.1f} m²")
                                             # Use values to avoid dimension mismatch and ensure reduction over 'gru'
                                             weighted_runoff = (var * areas.values).sum(dim='gru')
                                             return weighted_runoff.to_pandas()

                        except Exception as e:
                            self.logger.warning(f"Failed to perform area-weighted aggregation: {e}")

                    # Fallback to selection (original logic)
                    if len(var.shape) > 1:
                        self.logger.warning(f"Using first spatial unit for {var_name} (potential error for multi-unit basins)")
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

        # Priority 0: Manual override from config
        fixed_area = self.config_dict.get('FIXED_CATCHMENT_AREA')
        if fixed_area:
            self.logger.info(f"Using fixed catchment area from config: {fixed_area} m²")
            return float(fixed_area)

        # Priority 1: Try SUMMA attributes file first (most reliable)
        try:
            attrs_file = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
            if attrs_file.exists():
                with xr.open_dataset(attrs_file) as attrs:
                    if 'HRUarea' in attrs.data_vars:
                        catchment_area_m2 = float(attrs['HRUarea'].values.sum())
                        if 0 < catchment_area_m2 < 1e12:  # Reasonable area check
                            self.logger.info(f"Using catchment area from SUMMA attributes: {catchment_area_m2:.0f} m²")
                            return catchment_area_m2
                        else:
                            self.logger.warning(f"Catchment area from attributes.nc is {catchment_area_m2} m², which is out of bounds (0 < area < 1e12).")

        except Exception as e:
            self.logger.warning(f"Error reading SUMMA attributes file: {str(e)}")

        # Priority 2: Try basin shapefile
        try:
            import geopandas as gpd
            basin_path = self.project_dir / "shapefiles" / "river_basins"
            basin_files = list(basin_path.glob("*.shp"))
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                area_col = self.config_dict.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        self.logger.info(f"Using catchment area from basin shapefile: {total_area:.0f} m²")
                        return total_area
                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                geom_area = gdf.geometry.area.sum()
                self.logger.info(f"Using catchment area from basin geometry: {geom_area:.0f} m²")
                return geom_area
        except Exception as e:
            self.logger.warning(f"Could not calculate catchment area from basin shapefile: {str(e)}")

        # Priority 3: Try catchment shapefile
        try:
            import geopandas as gpd
            catchment_path = self.project_dir / "shapefiles" / "catchment"
            catchment_files = list(catchment_path.glob("*.shp"))
            if catchment_files:
                gdf = gpd.read_file(catchment_files[0])
                area_col = self.config_dict.get('CATCHMENT_SHP_AREA', 'HRU_area')
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        self.logger.info(f"Using catchment area from catchment shapefile: {total_area:.0f} m²")
                        return total_area
        except Exception as e:
            self.logger.warning(f"Error reading catchment shapefile: {str(e)}")

        # Fallback
        self.logger.warning("Using default catchment area: 1,000,000 m²")
        return 1e6  # 1 km² fallback
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data"""
        obs_path = self.config_dict.get('OBSERVATIONS_PATH')
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
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            default='lumped'
        )
        routing_delineation = self.config_dict.get('ROUTING_DELINEATION', 'lumped')

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        return False
