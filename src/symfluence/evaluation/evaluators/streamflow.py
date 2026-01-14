#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Streamflow Evaluator.

This module provides streamflow (discharge) evaluation for hydrological calibration.
Handles extraction of simulated streamflow from both SUMMA and mizuRoute outputs,
automatic unit conversion (mass flux to volume flux), spatial aggregation for
distributed models, and observed data matching.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import cast, List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    pass


@EvaluationRegistry.register('STREAMFLOW')
class StreamflowEvaluator(ModelEvaluator):
    """Streamflow evaluator for calibrating hydrological models.

    This evaluator handles extraction and processing of streamflow (discharge)
    from hydrological model outputs for comparison with observations. It supports
    multiple model types (SUMMA, mizuRoute) and handles common data issues like
    unit conversions and spatial aggregation.

    Key Responsibilities:
        1. Locate streamflow output files from model simulations
        2. Detect output format (SUMMA vs mizuRoute) and extract appropriately
        3. Handle unit conversions: mass flux (kg m⁻² s⁻¹) → volume flux (m³ s⁻¹)
        4. Perform spatial aggregation for distributed/semi-distributed models
        5. Convert per-unit-area runoff to basin-scale discharge using catchment area
        6. Match observed streamflow data from calibration targets

    Unit Conversion Details:
        SUMMA outputs runoff in three possible representations:
        - Mass flux: kg m⁻² s⁻¹ (mass per unit area per unit time)
        - Volume flux: m s⁻¹ (depth per unit time)
        - Total runoff: basin-scale totals (mm/day)

        Conversion strategy:
        1. Detect units from NetCDF attributes and data magnitude
        2. If mass flux: divide by water density (1000 kg/m³) → volume flux
        3. Spatial aggregation: sum over HRU/GRU with area weighting
        4. Scale to basin discharge: multiply by catchment area (m²) → m³/s

    Supported Output Formats:
        - SUMMA timestep files (*.nc with variables like averageRoutedRunoff)
        - mizuRoute network output (*.nc with routed runoff by reach)
        - SUMMA+mizuRoute coupled (distributed runoff fed to routing)

    Catchment Area Resolution (Priority):
        1. Manual override: FIXED_CATCHMENT_AREA in config (highest priority)
        2. SUMMA attributes.nc: HRUarea sum (most reliable automated)
        3. Basin shapefile: GRU_area column (user-defined delineation)
        4. Catchment shapefile: geometry area (fallback)
        5. Default: 1 km² (last resort, triggers warning)

    Configuration Parameters:
        FIXED_CATCHMENT_AREA: Manual area override (m²), skips auto-detection
        RIVER_BASIN_SHP_AREA: Column name in river basin shapefile (default: GRU_area)
        CATCHMENT_SHP_AREA: Column name in catchment shapefile (default: HRU_area)
        OBSERVATIONS_PATH: Override to observed streamflow file path

    Common Issues & Solutions:
        Issue: Unit mismatch (extremely high/low values)
            Solution: Automatic detection via magnitude thresholding (>1e-6 m/s = mass flux)

        Issue: Spatial dimension (HRU vs GRU vs reach)
            Solution: Area-weighted aggregation when attributes available; fallback to first unit

        Issue: Missing catchment area
            Solution: Priority fallback system; default 1 km² with warning

        Issue: Different variable names across model versions
            Solution: Tries multiple common names (averageRoutedRunoff, basin__TotalRunoff, etc.)

    Attributes:
        Uses inherited from ModelEvaluator:
        - project_dir: Project root directory
        - config_dict: Configuration dictionary
        - logger: Logger instance
        - calibration_period: (start_date, end_date) tuple for evaluation
    """
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate streamflow output files from simulation directory.

        Searches for SUMMA timestep files or mizuRoute routed output files
        using OutputFileLocator utility. Priority: mizuRoute if available
        (routed values are preferred), otherwise SUMMA direct outputs.

        Args:
            sim_dir: Directory containing simulation outputs

        Returns:
            List[Path]: Paths to streamflow output files
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_streamflow_files(sim_dir)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from simulation files with unit conversion.

        High-level extraction dispatch that:
        1. Detects output format (mizuRoute vs SUMMA)
        2. Calls appropriate extraction method
        3. Returns streamflow as pandas Series (m³/s)

        Data Processing:
        - Detects and corrects units (mass flux → volume flux)
        - Performs spatial aggregation for distributed models
        - Converts to basin-scale discharge

        Args:
            sim_files: List of simulation output NetCDF files
            **kwargs: Additional parameters (unused, for API consistency)

        Returns:
            pd.Series: Time series of streamflow (m³/s)

        Raises:
            Exception: If file cannot be read or no suitable variable found
        """
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
        """Detect if file contains mizuRoute routed output vs SUMMA direct output.

        Uses presence of mizuRoute-specific variables to distinguish formats:
        - mizuRoute has: IRFroutedRunoff, KWTroutedRunoff, reachID dimensions
        - SUMMA has: averageRoutedRunoff, basin__TotalRunoff, scalarTotalRunoff

        Args:
            sim_file: Path to NetCDF file

        Returns:
            bool: True if mizuRoute output format detected, False otherwise
        """
        try:
            with xr.open_dataset(sim_file) as ds:
                mizuroute_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'dlayRunoff', 'reachID', 'seg']
                return any(var in ds.variables or var in ds.dims for var in mizuroute_vars)
        except:
            return False

    def _extract_mizuroute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from mizuRoute routed output.

        mizuRoute provides runoff values for each reach in the river network.
        This method:
        1. Tries multiple routed runoff variables (IRF, KWT, averaged)
        2. Identifies outlet reach/segment (highest mean discharge)
        3. Returns time series at outlet

        mizuRoute Output Variables:
        - IRFroutedRunoff: Impulse Response Function routing
        - KWTroutedRunoff: Kinematic Wave Theory routing
        - averageRoutedRunoff: Combined routing methods
        - dlayRunoff: Delayed runoff (older versions)

        Dimensions:
        - seg: River segment ID (older mizuRoute)
        - reachID: Reach ID (newer versions)

        Args:
            sim_file: Path to mizuRoute output NetCDF

        Returns:
            pd.Series: Time series of routed discharge (m³/s)

        Raises:
            ValueError: If no routed runoff variable found in file
        """
        with xr.open_dataset(sim_file) as ds:
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'dlayRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        segment_means = var.mean(dim='time').values
                        outlet_seg_idx = np.argmax(segment_means)
                        result = cast(pd.Series, var.isel(seg=outlet_seg_idx).to_pandas())
                    elif 'reachID' in var.dims:
                        reach_means = var.mean(dim='time').values
                        outlet_reach_idx = np.argmax(reach_means)
                        result = cast(pd.Series, var.isel(reachID=outlet_reach_idx).to_pandas())
                    else:
                        continue
                    return result
            raise ValueError("No suitable streamflow variable found in mizuRoute output")

    def _extract_summa_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from SUMMA model output with comprehensive processing.

        SUMMA outputs runoff at HRU/GRU level. This method:
        1. Finds streamflow variable (tries multiple common names)
        2. Detects and converts units (mass flux → volume flux)
        3. Performs spatial aggregation (area-weighted if possible)
        4. Scales to basin discharge using catchment area

        Unit Conversion Logic:
            SUMMA may output runoff in mass flux (kg m⁻² s⁻¹) incorrectly
            labeled as volume flux (m s⁻¹). Detection strategy:
            - Check units attribute for 'kg' (explicit mass flux)
            - Check data magnitude: if mean > 1e-6 m/s, likely mislabeled
              (realistic runoff typically 0.01-0.1 mm/day ≈ 10^-10 to 10^-9 m/s)
            - If mass flux detected: divide by 1000 (water density)

        Spatial Aggregation Strategy:
            If SUMMA attributes.nc exists with HRU/GRU areas:
            - Area-weighted sum: discharge = Σ(runoff_i × area_i)
            - Preserves spatial heterogeneity in calibration
            Fallback (if no attributes or error):
            - Select first HRU/GRU (triggers warning)
            - Not ideal for multi-unit basins

        Final Scaling:
            - Multiply per-unit runoff by catchment area (m²)
            - Result: basin-scale discharge (m³/s)

        SUMMA Variable Names (tried in order):
            - averageRoutedRunoff (preferred)
            - basin__TotalRunoff (newer versions)
            - scalarTotalRunoff (older versions)

        Args:
            sim_file: Path to SUMMA output NetCDF

        Returns:
            pd.Series: Basin-scale streamflow time series (m³/s)

        Raises:
            ValueError: If no suitable streamflow variable found
        """
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
                    elif float(var.mean().item()) > 1e-6:
                        self.logger.debug(f"Variable {var_name} mean ({float(var.mean().item()):.2e}) is unreasonably high for m/s. Assuming mislabeled mass flux.")
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
                                            return cast(pd.Series, weighted_runoff.to_pandas())
                                    
                                    # Handle GRU dimension
                                    elif 'gru' in var.dims and 'GRUarea' in attrs:
                                        areas = attrs['GRUarea']
                                        if areas.sizes['gru'] == var.sizes['gru']:
                                            total_area = float(areas.values.sum())
                                            self.logger.debug(f"Performing area-weighted aggregation for {var_name} (GRU). Total area: {total_area:.1f} m²")
                                            weighted_runoff = (var * areas).sum(dim='gru')
                                            return cast(pd.Series, weighted_runoff.to_pandas())
                                            
                                    # Fallback if specific area variable missing but HRUarea available for GRU dim (common in lumped)
                                    elif 'gru' in var.dims and 'HRUarea' in attrs:
                                         # If 1:1 mapping or if we can infer
                                         if attrs.sizes['hru'] == var.sizes['gru']:
                                             areas = attrs['HRUarea'] # Assuming 1:1 mapping for lumped
                                             total_area = float(areas.values.sum())
                                             self.logger.debug(f"Performing area-weighted aggregation for {var_name} (GRU using HRUarea). Total area: {total_area:.1f} m²")
                                             # Use values to avoid dimension mismatch and ensure reduction over 'gru'
                                             weighted_runoff = (var * areas.values).sum(dim='gru')
                                             return cast(pd.Series, weighted_runoff.to_pandas())

                        except Exception as e:
                            self.logger.warning(f"Failed to perform area-weighted aggregation: {e}")

                    # Fallback to selection (original logic)
                    if len(var.shape) > 1:
                        self.logger.warning(f"Using first spatial unit for {var_name} (potential error for multi-unit basins)")
                        if 'hru' in var.dims:
                            sim_data = cast(pd.Series, var.isel(hru=0).to_pandas())
                        elif 'gru' in var.dims:
                            sim_data = cast(pd.Series, var.isel(gru=0).to_pandas())
                        else:
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = cast(pd.Series, var.isel({non_time_dims[0]: 0}).to_pandas())
                            else:
                                sim_data = cast(pd.Series, var.to_pandas())
                    else:
                        sim_data = cast(pd.Series, var.to_pandas())

                    catchment_area = self._get_catchment_area()
                    return sim_data * catchment_area
            raise ValueError("No suitable streamflow variable found in SUMMA output")
    
    def _get_catchment_area(self) -> float:
        """Determine catchment area with multi-source fallback strategy.

        Catchment area is essential for converting per-unit-area runoff (m/s)
        to basin-scale discharge (m³/s): Q = runoff × area

        Resolution Priority (highest to lowest reliability):
        1. FIXED_CATCHMENT_AREA in config
           - User override, skips all detection
           - Use when auto-detection fails or for forcing specific values
        2. SUMMA attributes.nc: HRUarea sum
           - Most reliable: directly from model setup
           - Sum of all HRU areas (distributed or lumped)
           - Requires: settings/SUMMA/attributes.nc exists with HRUarea variable
        3. River basin shapefile: GRU_area column
           - User-defined basin delineation (e.g., from QGIS)
           - Searches shapefiles/river_basins/*.shp for GRU_area column
           - Fallback: calculate from geometry area (projects to UTM)
        4. Catchment shapefile: geometry area
           - Alternative delineation layer
           - Searches shapefiles/catchment/*.shp
           - Attempts HRU_area column first, then geometry
        5. Default: 1 km² (1e6 m²)
           - Last resort, triggers warning
           - Only if all above methods fail or return invalid values

        Validation:
            Checks all areas for reasonable bounds: 0 < area < 1e12 m²
            - Minimum: >0 (negative/zero areas invalid)
            - Maximum: <1e12 m² (extreme check for unit/calculation errors)

        Returns:
            float: Catchment area in square meters (m²)

        Warnings:
            Logs info level when falling back between priority levels
            Logs warning for invalid areas or if using 1 km² default
        """

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
        """Get path to observed streamflow data file.

        Resolves observed streamflow from configuration or default location.

        Path Resolution:
        1. If OBSERVATIONS_PATH configured and not 'default': use it directly
        2. Otherwise: use default pattern:
           observations/streamflow/preprocessed/{domain_name}_streamflow_processed.csv

        The preprocessed CSV should contain:
        - Time index (datetime): observation timestamps
        - Streamflow column: discharge values (typically m³/s)
        - May have multiple columns; column selected via _get_observed_data_column()

        Returns:
            Path: Path to observed streamflow CSV file
        """
        obs_path = self.config_dict.get('OBSERVATIONS_PATH')
        if obs_path == 'default' or not obs_path:
            return self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        return Path(obs_path)

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find streamflow column in observed data CSV.

        Searches CSV columns for streamflow using heuristic pattern matching.
        This accommodates various naming conventions across different data sources.

        Matched Terms (case-insensitive):
        - 'flow': discharge, streamflow, river_flow
        - 'discharge': q_discharge, discharge
        - 'q_': q_mm, q_cms, q_observed (standardized naming)
        - 'streamflow': streamflow, daily_streamflow

        Args:
            columns: List of column names from observed data CSV

        Returns:
            Optional[str]: Matched column name if found, None otherwise

        Note:
            Returns first matching column found. If multiple candidates exist,
            consider manual specification in config or preprocessing step.
        """
        for col in columns:
            if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                return col
        return None

    def needs_routing(self) -> bool:
        """Determine if streamflow calibration requires mizuRoute routing.

        mizuRoute routing is needed when:
        1. Domain is semi-distributed or distributed (not lumped/point)
        2. Domain is lumped but with river network delineation

        Decision Logic:
            - If domain_method in [point, lumped]:
              * If routing_delineation == 'river_network': NEEDS ROUTING
              * Otherwise: DOESN'T NEED ROUTING
            - If domain_method in [semi-distributed, distributed]:
              * ALWAYS NEEDS ROUTING

        Why Routing?:
            - Lumped domains: SUMMA outputs lumped runoff to basin outlet
            - Distributed domains: runoff generated per HRU, must route through network
            - Semi-distributed: mixed; routing needed for proper streamflow timing

        Returns:
            bool: True if mizuRoute routing required, False if direct SUMMA sufficient

        Configuration:
            - domain.definition_method (from config): 'point', 'lumped', etc.
            - ROUTING_DELINEATION (from config): 'lumped' or 'river_network'
        """
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
