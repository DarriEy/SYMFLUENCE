#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GR Streamflow Evaluator.

Specialized streamflow evaluation for GR4J/GR6J hydrological models.
Handles GR-specific output formats (lumped CSV, distributed NetCDF) and
unit conversions (mm/day → m³/s). Extends base StreamflowEvaluator with
GR-specific logic for file location, output parsing, and area determination.
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
    """Streamflow evaluator for GR4J and GR6J conceptual rainfall-runoff models.

    GR models are lumped or semi-distributed conceptual models that output
    streamflow at daily timesteps. This evaluator handles:
    1. Lumped mode: CSV output with daily mean discharge
    2. Distributed mode: NetCDF output with per-HRU runoff requiring aggregation

    GR Output Characteristics:
        - Lumped: Single point output (q_sim column in mm/day or m³/s)
        - Distributed: Per-HRU or per-GRU output (requires spatial aggregation)
        - Timestep: Always daily (requires daily resampling of observations)
        - Units: Typically mm/day for runoff, converted to m³/s for comparison

    Unit Conversions:
        GR outputs in mm/day (depth). Conversion to m³/s:
        - mm/day → m/day (divide by 1000)
        - m/day → m³/day (multiply by catchment area in m²)
        - m³/day → m³/s (divide by 86400)
        - Or use constant: MM_DAY_TO_CMS from UnitConversion

    Output Format Detection:
        - .csv files: Lumped GR output (single location)
        - .nc files: Check for mizuRoute routing (IRFroutedRunoff) or GR distributed (q_routed)

    Observation Resampling:
        GR outputs daily values, so observations must be aggregated to daily
        for proper comparison. This evaluator automatically resamples
        sub-daily observations to daily mean.

    Attributes:
        Inherits from StreamflowEvaluator with GR-specific overrides:
        - domain_name: Basin identifier
        - config_dict: Configuration with GR-specific parameters
        - project_dir: Project root with expected subdirectory structure
    """

    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed streamflow and resample to daily frequency.

        GR models output daily mean discharge, so observations must be aggregated
        to daily timestep for proper comparison. This method:
        1. Loads observed data from CSV (calls parent method)
        2. Resamples from any frequency (hourly, sub-daily) to daily
        3. Uses mean aggregation (appropriate for streamflow)

        Resampling:
            - From: Any timestep (typically hourly or 15-minute)
            - To: Daily (mean)
            - Matches GR output frequency

        Args:
            None (uses configuration for path)

        Returns:
            Optional[pd.Series]: Daily aggregated streamflow (m³/s) or None if load fails
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
        """Locate GR model output files (CSV for lumped, NetCDF for distributed).

        GR models produce:
        - CSV: Lumped outputs (single q_sim column with daily values)
        - NetCDF: Distributed outputs (per-HRU runoff)

        Args:
            sim_dir: Directory containing GR simulation outputs

        Returns:
            List[Path]: Path(s) to GR output file(s)
        """
        locator = OutputFileLocator(self.logger)
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default=None
        )
        return locator.find_gr_output(sim_dir, self.domain_name, experiment_id)
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract GR streamflow from lumped (CSV) or distributed (NetCDF) output.

        Format Detection and Extraction:
        - .csv: Lumped GR output → calls _extract_lumped_gr_streamflow()
        - .nc with mizuRoute vars: Routed output → calls parent _extract_mizuroute_streamflow()
        - .nc with GR vars: Distributed GR output → calls _extract_distributed_gr_streamflow()

        Unit Handling:
            Detects output units and converts to m³/s (cms):
            - Already m³/s: Return as-is
            - mm/day: Convert using catchment area and MM_DAY_TO_CMS constant
            - m/s: Multiply by catchment area (m²)

        Args:
            sim_files: List of simulation output file(s)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Daily streamflow (m³/s)
        """
        sim_file = sim_files[0]
        self.logger.info(f"Extracting simulated streamflow from: {sim_file}")

        if sim_file.suffix == '.csv':
            return self._extract_lumped_gr_streamflow(sim_file)
        else:
            # Check if it's mizuRoute output or GR distributed output
            if self._is_mizuroute_output(sim_file):
                sim_data = self._extract_mizuroute_streamflow(sim_file)

                # Check units and convert if needed (mm/day -> cms)
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
        """Extract streamflow from GR lumped CSV output with unit conversion.

        GR lumped models (GR4J, GR6J) output single-point runoff in mm/day.
        This method:
        1. Reads CSV with q_sim column
        2. Converts mm/day → m³/s using catchment area

        Unit Conversion Formula:
            mm/day × (area_km²) / 86.4 = m³/s
            (from UnitConversion.MM_DAY_TO_CMS constant)

        Args:
            sim_file: Path to GR CSV output file

        Returns:
            pd.Series: Basin-scale daily streamflow (m³/s)
        """
        df_sim = pd.read_csv(sim_file, index_col='datetime', parse_dates=True)
        # GR4J output is in mm/day. Convert to cms.
        area_m2 = self._get_catchment_area()
        area_km2 = area_m2 / 1e6
        simulated_streamflow = df_sim['q_sim'] * area_km2 / UnitConversion.MM_DAY_TO_CMS
        return simulated_streamflow

    def _extract_distributed_gr_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from GR distributed NetCDF output with aggregation.

        GR distributed models output per-HRU/GRU runoff variables. This method:
        1. Loads configured routing variable (q_routed or override)
        2. Aggregates across HRUs/GRUs (mean)
        3. Detects and converts units (mm/day or m/s) to m³/s

        Unit Handling:
            - If m/s: multiply by catchment area (m²) → m³/s
            - If mm/day: convert using catchment area and constant

        Configuration:
            SETTINGS_MIZU_ROUTING_VAR: Variable name (default: q_routed)

        Args:
            sim_file: Path to GR distributed NetCDF output

        Returns:
            pd.Series: Basin-scale daily streamflow (m³/s)

        Raises:
            ValueError: If configured variable not found in file
        """
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
        """Determine catchment area with GR-specific priority strategy.

        GR catchment area resolution priority (optimized for GR models):
        1. River basin shapefile: GRU_area column or geometry
           - Most common for GR lumped configurations
           - Searches shapefiles/river_basins/{domain_name}*.shp
        2. Catchment/HRU shapefile: GRU_area or geometry
           - Alternative delineation with configured name
           - Path and name customizable in config
        3. Fallback to parent class method
           - Checks SUMMA attributes, other shapefiles, defaults

        All methods include geometry area fallback with UTM projection conversion.

        Returns:
            float: Catchment area in m² (square meters)
        """
        # Priority 1: Try river basins shapefile (often used for lumped area)
        try:
            import geopandas as gpd
            basin_dir = self.project_dir / 'shapefiles' / 'river_basins'
            if basin_dir.exists():
                # Find any shapefile in river_basins that starts with domain_name
                import glob
                basin_files = list(basin_dir.glob(f"{self.domain_name}*.shp"))
                if basin_files:
                    basin_path = basin_files[0]
                    gdf = gpd.read_file(basin_path)
                    if 'GRU_area' in gdf.columns:
                        area_m2 = gdf['GRU_area'].sum()
                        if 0 < area_m2 < 1e12:
                            self.logger.debug(f"Catchment area from river_basins GRU_area: {area_m2:.2f} m2")
                            return float(area_m2)
                    
                    # Fallback: calculate from geometry
                    if gdf.crs and not gdf.crs.is_geographic:
                        area_m2 = gdf.geometry.area.sum()
                    else:
                        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                        area_m2 = gdf_utm.geometry.area.sum()
                    self.logger.debug(f"Catchment area from river_basins geometry: {area_m2:.2f} m2")
                    return float(area_m2)
        except Exception as e:
            self.logger.debug(f"Error calculating area from river_basins: {e}")

        # Priority 2: Try catchment/HRU shapefile
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
