#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evapotranspiration (ET) Evaluator.

Evaluates simulated evapotranspiration (ET) and latent heat flux from hydrological
and land-surface models against multiple observation sources.

Supported Observation Sources:
    - MOD16/MODIS: NASA MODIS MOD16A2 8-day ET product (remote sensing)
    - FLUXCOM: FLUXCOM gridded ET estimates (machine learning ensemble)
    - FluxNet: In-situ tower observations of energy and water fluxes
    - GLEAM: Global Land Evaporation Amsterdam Model satellite data

Data Acquisition:
    - MOD16: Cloud-based via NASA AppEEARS (requires authentication)
    - FluxNet: Tower networks (global distribution, varies by region)
    - FLUXCOM/GLEAM: Downloaded via EarthData or Zenodo

Unit Conversions:
    SUMMA Model Output:
    - Evapotranspiration: kg m⁻² s⁻¹ → mm/day (multiply by 86400)
    - Latent heat: W m⁻² (no conversion needed)

    Observations:
    - MOD16: mm/day (8-day composite)
    - FluxNet: mm/day or W m⁻² depending on source
    - FLUXCOM: mm/day
    - GLEAM: mm/day

Configuration:
    ET_OBS_SOURCE: 'mod16', 'fluxcom', 'fluxnet', or 'gleam'
    ET_OBS_PATH: Direct path override for observation file
    ET_TEMPORAL_AGGREGATION: 'daily_mean' or 'daily_sum' for sub-daily data
    ET_USE_QUALITY_CONTROL: Apply QC filtering (True/False)
    ET_MAX_QUALITY_FLAG: Maximum quality flag threshold (FluxNet)
"""

import logging
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import cast, Any, List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('ET')
@EvaluationRegistry.register('MODIS_ET')
@EvaluationRegistry.register('MOD16')
@EvaluationRegistry.register('FLUXNET')
@EvaluationRegistry.register('FLUXNET_ET')
class ETEvaluator(ModelEvaluator):
    """Evapotranspiration and latent heat evaluator with multi-source support.

    Evaluates simulated ET/latent heat flux from land-surface models (SUMMA, NoahMP)
    against observations from four major sources: MOD16/MODIS satellite remote sensing,
    FLUXCOM machine learning gridded estimates, FluxNet tower observations, and GLEAM
    satellite data.

    Multi-Source Support:
        Source selection via ET_OBS_SOURCE configuration (default: 'mod16'):
        - 'mod16', 'modis', 'modis_et': NASA MOD16A2 8-day composites
        - 'fluxcom', 'fluxcom_et': FLUXCOM v3 gridded ET (0.25° × 0.25°)
        - 'fluxnet': Global FluxNet tower network (30+ towers worldwide)
        - 'gleam': GLEAM v3.5a/v3.8a satellite data (0.25° × 0.25°)

    Evaluation Targets:
        - 'et' (default): Total evapotranspiration (mm/day)
        - 'latent_heat': Latent heat flux (W m⁻²)

    Model Compatibility:
        Primary: SUMMA (scalarTotalET, component ET fluxes, scalarLatHeatTotal)
        Secondary: Any model with similar variable naming conventions

    Spatial Aggregation (SUMMA):
        Single HRU/GRU: Selects that unit via isel()
        Multiple HRU/GRU: Averages across spatial dimension via mean()
        Multiple dimensions: Selects first element for non-time dims

    Data Acquisition:
        MOD16: AppEEARS API (requires NASA Earthdata account)
        FluxNet: FLUXNET data portal (site-specific download)
        FLUXCOM/GLEAM: Zenodo/EarthData mirrors (public access)

    Configuration:
        ET_OBS_SOURCE: Observation source identifier (default: 'mod16')
        ET_OBS_PATH: Direct file path override (skips source resolution)
        ET_TEMPORAL_AGGREGATION: 'daily_mean' or 'daily_sum' for sub-daily data
        ET_USE_QUALITY_CONTROL: Enable FluxNet QC filtering (default: True)
        ET_MAX_QUALITY_FLAG: FluxNet QC threshold (0=best, 2=good, default: 2)

    Attributes:
        optimization_target: 'et' or 'latent_heat'
        variable_name: Same as optimization_target
        obs_source: Validated observation source ('mod16', 'fluxcom', 'fluxnet', 'gleam')
        temporal_aggregation: Aggregation method for sub-daily data
        use_quality_control: Enable/disable QC filtering
        max_quality_flag: FluxNet QC threshold
    """

    # Supported observation sources
    SUPPORTED_SOURCES = {
        'mod16', 'modis', 'modis_et', 'mod16a2',
        'fluxcom', 'fluxcom_et',
        'fluxnet',
        'gleam'
    }

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger):
        """Initialize ET evaluator with target and source determination.

        Determines evaluation target (ET vs latent heat) and observation source
        (MOD16, FLUXCOM, FluxNet, GLEAM) via configuration hierarchy.

        Configuration Priority:
            1. config.optimization.target (typed config, if 'et' or 'latent_heat')
            2. EVALUATION_VARIABLE (dict config, if 'et' or 'latent_heat')
            3. Default: 'et'

        Observation Source Priority:
            1. ET_OBS_SOURCE config (validated against SUPPORTED_SOURCES)
            2. Default: 'mod16'
            3. Invalid sources trigger warning + default to 'mod16'

        Temporal Aggregation:
            - 'daily_mean': Average sub-daily data to daily frequency
            - 'daily_sum': Sum sub-daily data to daily frequency
            - Default: 'daily_mean'

        Quality Control (FluxNet only):
            - ET_USE_QUALITY_CONTROL: Enable/disable QC filtering
            - ET_MAX_QUALITY_FLAG: QC threshold (0=best, 2=good, default: 2)

        Args:
            config: Typed configuration object (SymfluenceConfig)
            project_dir: Project root directory
            logger: Logger instance

        Raises:
            None (invalid sources trigger warning + default, not exceptions)
        """
        super().__init__(config, project_dir, logger)

        # Determine ET variable type from config
        self.optimization_target = self._get_config_value(
            lambda: self.config.optimization.target,
            default='streamflow'
        )
        if self.optimization_target not in ['et', 'latent_heat']:
            eval_var = self.config_dict.get('EVALUATION_VARIABLE', '')
            if eval_var in ['et', 'latent_heat']:
                self.optimization_target = eval_var
            else:
                self.optimization_target = 'et'

        self.variable_name = self.optimization_target

        # Observation source configuration
        self.obs_source = str(self.config_dict.get('ET_OBS_SOURCE', 'mod16')).lower()
        if self.obs_source not in self.SUPPORTED_SOURCES:
            self.logger.warning(
                f"Unknown ET_OBS_SOURCE '{self.obs_source}', defaulting to 'mod16'"
            )
            self.obs_source = 'mod16'

        # Temporal aggregation method
        self.temporal_aggregation = self.config_dict.get('ET_TEMPORAL_AGGREGATION', 'daily_mean')

        # Quality control settings
        self.use_quality_control = self.config_dict.get('ET_USE_QUALITY_CONTROL', True)
        self.max_quality_flag = self.config_dict.get('ET_MAX_QUALITY_FLAG', 2)

        self.logger.info(
            f"Initialized ETEvaluator for {self.optimization_target.upper()} "
            f"evaluation using {self.obs_source.upper()} observations"
        )
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate SUMMA daily output files containing ET variables.

        Searches for NetCDF files containing scalarTotalET, ET components
        (canopy/ground evaporation, transpiration, sublimation), or
        scalarLatHeatTotal depending on evaluation target.

        Args:
            sim_dir: Directory containing SUMMA simulation output (NetCDF files)

        Returns:
            List[Path]: Paths to ET output files (typically scalarTotalET*.nc)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_et_files(sim_dir)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract ET or latent heat from SUMMA simulation output.

        Dispatches to _extract_et_data() or _extract_latent_heat_data() based
        on optimization_target determined during initialization.

        Processing Steps:
            1. Open first NetCDF file from sim_files
            2. Check for complete scalarTotalET variable (preferred)
            3. Fallback: Sum individual ET component fluxes
            4. Collapse spatial dimensions (HRU/GRU) to basin scale
            5. Convert units from model native to mm/day
            6. Return time series

        Args:
            sim_files: List of SUMMA simulation output files (NetCDF)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Time series of ET (mm/day) or latent heat (W m⁻²)

        Raises:
            Exception: If file cannot be read or required variables not found
        """
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
        """Extract total evapotranspiration from SUMMA output.

        Attempts to extract scalarTotalET directly; if unavailable, sums individual
        ET component fluxes (canopy transpiration, canopy evaporation, ground
        evaporation, snow sublimation, canopy sublimation).

        SUMMA ET Variables:
            - scalarTotalET: Integrated total ET (preferred, kg m⁻² s⁻¹)
            - scalarCanopyTranspiration: Transpiration through leaves (kg m⁻² s⁻¹)
            - scalarCanopyEvaporation: Evaporation from canopy (kg m⁻² s⁻¹)
            - scalarGroundEvaporation: Evaporation from soil (kg m⁻² s⁻¹)
            - scalarSnowSublimation: Sublimation from snowpack (kg m⁻² s⁻¹)
            - scalarCanopySublimation: Sublimation from canopy (kg m⁻² s⁻¹)

        Spatial Aggregation (HRU/GRU):
            Single HRU/GRU (size=1): Selects that unit via isel()
            Multiple HRU/GRU: Averages across dimension via mean()
            Other dimensions: Selects first element (e.g., layer if present)

        Unit Conversion:
            Input: kg m⁻² s⁻¹ (SUMMA native)
            Output: mm/day (multiply by 86400 seconds/day)

        Args:
            ds: xarray Dataset with ET variables from SUMMA output

        Returns:
            pd.Series: Time series of total ET in mm/day

        Raises:
            ValueError: If neither scalarTotalET nor component variables found
        """
        if 'scalarTotalET' in ds.variables:
            et_var = ds['scalarTotalET']
            
            # Collapse spatial dimensions
            sim_xr = et_var
            for dim in ['hru', 'gru']:
                if dim in sim_xr.dims:
                    if sim_xr.sizes[dim] == 1:
                        sim_xr = sim_xr.isel({dim: 0})
                    else:
                        sim_xr = sim_xr.mean(dim=dim)
            
            # Handle any remaining non-time dimensions
            non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
            if non_time_dims:
                sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})
            
            sim_data = sim_xr.to_pandas()
            
            # Convert units: SUMMA outputs kg m-2 s-1, convert to mm/day
            sim_data = self._convert_et_units(sim_data, from_unit='kg_m2_s', to_unit='mm_day')
            return sim_data
        else:
            return self._sum_et_components(ds)
    
    def _sum_et_components(self, ds: xr.Dataset) -> pd.Series:
        """Sum individual ET component fluxes to compute total ET.

        Fallback method when scalarTotalET is unavailable. Integrates:
        - Canopy transpiration: Water uptake through leaf stomata
        - Canopy evaporation: Direct evaporation from leaf/stem surfaces
        - Ground evaporation: Evaporation from soil surface
        - Snow sublimation: Direct vaporization of snowpack
        - Canopy sublimation: Direct vaporization from canopy

        Component Summation Strategy:
            1. Check availability of each component variable
            2. Collapse spatial dimensions (HRU/GRU) to basin scale
            3. Sum available components sequentially
            4. Convert units to mm/day

        This approach accommodates model variants where scalarTotalET is not
        directly computed but must be derived from components.

        Args:
            ds: xarray Dataset with ET component variables

        Returns:
            pd.Series: Time series of summed ET components in mm/day

        Raises:
            ValueError: If no ET component variables found in dataset
            Exception: If spatial aggregation or unit conversion fails
        """
        try:
            et_components: dict[str, Any] = {}
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
        """Extract latent heat flux from SUMMA output.

        Latent heat flux represents the energy dissipated during phase changes
        (evaporation, sublimation, condensation). Related to ET via:

            Latent Heat = ET (kg m⁻² s⁻¹) × L_v (J kg⁻¹)

        where L_v ≈ 2.5×10⁶ J kg⁻¹ (latent heat of vaporization).

        SUMMA Latent Heat Variables:
            - scalarLatHeatTotal: Total latent heat flux (W m⁻², primary)
            - scalarLatHeatCanopy: Latent heat from canopy processes
            - scalarLatHeatGround: Latent heat from ground/soil

        Spatial Aggregation (HRU/GRU):
            Single HRU/GRU (size=1): Selects that unit via isel()
            Multiple HRU/GRU: Averages across dimension via mean()
            Other dimensions: Selects first element

        No Unit Conversion:
            Input: W m⁻² (SUMMA native)
            Output: W m⁻² (no conversion needed)

        Args:
            ds: xarray Dataset with latent heat variables from SUMMA

        Returns:
            pd.Series: Time series of latent heat flux (W m⁻²)

        Raises:
            ValueError: If scalarLatHeatTotal not found in dataset
        """
        if 'scalarLatHeatTotal' in ds.variables:
            lh_var = ds['scalarLatHeatTotal']
            
            # Collapse spatial dimensions
            sim_xr = lh_var
            for dim in ['hru', 'gru']:
                if dim in sim_xr.dims:
                    if sim_xr.sizes[dim] == 1:
                        sim_xr = sim_xr.isel({dim: 0})
                    else:
                        sim_xr = sim_xr.mean(dim=dim)
            
            # Handle any remaining non-time dimensions
            non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
            if non_time_dims:
                sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})

            sim_data = cast(pd.Series, sim_xr.to_pandas())
            return sim_data
        else:
            raise ValueError("scalarLatHeatTotal not found in SUMMA output")
    
    def _convert_et_units(self, et_data: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert evapotranspiration units.

        Converts between different ET unit systems used by SUMMA and observation sources.

        Supported Conversions:
            - kg m⁻² s⁻¹ → mm/day:
                Formula: ET_mm_day = ET_kg_m2_s × 86400 s/day
                Factor: 86400 (assumes water density = 1 kg/L)
            - Same unit: No conversion (returns unchanged)
            - Other conversions: Returns data unchanged (no-op fallback)

        Physical Basis:
            - SUMMA outputs mass flux (kg m⁻² s⁻¹, specific mass per area per time)
            - mm/day represents equivalent water depth (volume per area per time)
            - Conversion assumes water density = 1000 kg m⁻³ (liquid water)
            - 86400 = seconds per day (365.25 days/year, 24 hours/day, 3600 s/hour)

        Args:
            et_data: Evapotranspiration time series to convert
            from_unit: Source unit ('kg_m2_s', 'mm_day', or other)
            to_unit: Target unit ('kg_m2_s', 'mm_day', or other)

        Returns:
            pd.Series: Converted ET time series with same index as input
        """
        if from_unit == 'kg_m2_s' and to_unit == 'mm_day':
            # Convert kg m-2 s-1 to mm/day
            return et_data * 86400.0
        elif from_unit == to_unit:
            return et_data
        else:
            return et_data
    
    def get_observed_data_path(self) -> Path:
        """Resolve path to observed ET data based on source configuration.

        Implements source-specific path resolution with fallback strategy:
        1. ET_OBS_PATH config override (if specified, skips source resolution)
        2. Source-specific subdirectories (mod16/ vs fluxcom/ vs gleam/ vs fluxnet/)
        3. Domain-specific naming convention (domain_name_SOURCE_et_processed.csv)

        File Location Convention:
            observations/
            ├── et/
            │   ├── preprocessed/
            │   │   ├── {domain}_modis_et_processed.csv (MOD16)
            │   │   ├── {domain}_fluxcom_et_processed.csv (FLUXCOM)
            │   │   ├── {domain}_gleam_et_processed.csv (GLEAM)
            │   │   └── {domain}_fluxnet_et_processed.csv (FluxNet)
            │   └── raw/ (for raw downloads)
            └── energy_fluxes/
                └── processed/
                    └── {domain}_fluxnet_processed.csv (FluxNet energy fluxes)

        Returns Path even if file doesn't exist (triggers acquisition if needed).

        Args:
            None (uses self.project_dir, self.domain_name, self.obs_source)

        Returns:
            Path: Absolute path to observation file (may not exist yet)
        """
        # Direct path override
        et_obs_path = self.config_dict.get('ET_OBS_PATH')
        if et_obs_path:
            return Path(et_obs_path)

        # MOD16/MODIS ET
        if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_modis_et_processed.csv"
            )

        # FLUXCOM ET
        if self.obs_source in {'fluxcom', 'fluxcom_et'}:
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_fluxcom_et_processed.csv"
            )

        # GLEAM ET
        if self.obs_source == 'gleam':
            return (
                self.project_dir
                / "observations"
                / "et"
                / "preprocessed"
                / f"{self.domain_name}_gleam_et_processed.csv"
            )

        # FluxNet
        if self.obs_source == 'fluxnet':
            # Try multiple possible paths for FLUXNET data
            fluxnet_station = self._get_config_value(
                lambda: self.config.evaluation.fluxnet.station,
                default=''
            )
            possible_paths = [
                self.project_dir / "observations" / "et" / "preprocessed" / f"{self.domain_name}_fluxnet_et_processed.csv",
                self.project_dir / "observations" / "energy_fluxes" / "processed" / f"{self.domain_name}_fluxnet_processed.csv",
                self.project_dir / "observations" / "fluxnet" / f"{self.domain_name}_FLUXNET_{fluxnet_station}.csv",
            ]
            for path in possible_paths:
                if path.exists():
                    return path
            # Return default path (will trigger acquisition if not exists)
            return possible_paths[0]

        # Default: FluxNet energy fluxes
        return (
            self.project_dir
            / "observations"
            / "energy_fluxes"
            / "processed"
            / f"{self.domain_name}_fluxnet_processed.csv"
        )
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify ET data column based on observation source and target.

        Different data sources use inconsistent column naming conventions. This method
        attempts to match column names via priority-ordered search to support multiple
        data formats without requiring exact column name specification.

        Column Naming by Source:
            MOD16/MODIS:
                - Primary: 'et_mm_day', 'et_daily_mm' (exact matches)
                - Fallback: 'et' (generic)
            FLUXCOM:
                - Primary: 'et' (common in FLUXCOM exports)
                - Pattern: 'evapotranspiration', 'ET'
            FluxNet (ET):
                - Primary: 'ET_from_LE_mm_per_day' (derived from latent heat)
                - Pattern: 'ET_from_LE_*' or 'ET' (generic)
            FluxNet (Latent Heat):
                - Primary: 'LE_F_MDS' (gap-filled latent heat)
                - Pattern: 'LE_*' (latent heat variants)
            GLEAM:
                - Pattern: 'E', 'Et', 'et' (consistent with ET abbreviation)

        Search Strategy:
            1. Source-specific exact matches (highest priority)
            2. Target-specific patterns (ET vs latent heat)
            3. Generic fallback terms

        Args:
            columns: List of column names from observation CSV

        Returns:
            str: Name of ET/latent heat column, or None if not found
        """
        columns_lower = [c.lower() for c in columns]

        if self.optimization_target == 'et':
            # MOD16 column names (highest priority for MOD16 source)
            if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
                for col in columns:
                    if col.lower() in ['et_mm_day', 'et', 'et_daily_mm']:
                        return col

            # General ET column search
            priority_terms = [
                'et_mm_day',  # MOD16 processed
                'et_from_le',  # FluxNet
                'evapotranspiration',
                'et'  # Generic
            ]

            for term in priority_terms:
                for col in columns:
                    if term in col.lower():
                        return col

            # Specific fallbacks
            if 'ET_from_LE_mm_per_day' in columns:
                return 'ET_from_LE_mm_per_day'
            if 'ET' in columns:
                return 'ET'
            if 'et' in columns:
                return 'et'

        elif self.optimization_target == 'latent_heat':
            for col in columns:
                if any(term in col.lower() for term in ['le_f_mds', 'le_', 'latent']):
                    return col
            if 'LE_F_MDS' in columns:
                return 'LE_F_MDS'

        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed ET with quality control and temporal aggregation.

        Implements robust loading pipeline handling multiple file formats:
        - MOD16: Simple CSV with datetime index and et_mm_day column
        - FluxNet: Complex CSV with metadata, multiple columns, QC flags
        - FLUXCOM/GLEAM: Similar to MOD16 format

        Loading Strategy:
            1. Resolve observation file path (may trigger acquisition if missing)
            2. Attempt direct load with datetime index (MOD16 format)
            3. Fallback to standard loading with date column detection
            4. Extract data column using source-specific naming logic
            5. Apply quality control if enabled and source supports it
            6. Resample sub-daily data to daily frequency if needed
            7. Return clean, daily time series

        Quality Control (FluxNet):
            - Uses LE_F_MDS_QC flags (0-2 scale)
            - Retains only measurements with QC ≤ max_quality_flag
            - Default threshold: 2 (good quality, allows some gap-filling)

        Temporal Aggregation:
            - 'daily_mean': Average sub-daily data (streamflow-like)
            - 'daily_sum': Sum sub-daily data (ET accumulation)
            - Default: 'daily_mean' (most common for ET)

        Data Acquisition:
            - If MOD16 file missing: Attempts AppEEARS acquisition
            - If FluxNet file missing: Attempts portal download
            - Other sources: Returns None if file missing

        Args:
            None (uses self.obs_path, self.obs_source, self.temporal_aggregation)

        Returns:
            Optional[pd.Series]: Daily ET time series (mm/day) or None if load fails

        Raises:
            None (logs errors, returns None for failed loads)
        """
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.warning(f"Observation file not found: {obs_path}")
                # Try to trigger acquisition based on source
                if self.obs_source in {'mod16', 'modis', 'modis_et', 'mod16a2'}:
                    self._try_acquire_mod16_data()
                    if obs_path.exists():
                        self.logger.info(f"MOD16 data acquired: {obs_path}")
                    else:
                        return None
                elif self.obs_source == 'fluxnet':
                    result = self._try_acquire_fluxnet_data()
                    if result and result.exists():
                        obs_path = result
                        self.logger.info(f"FLUXNET data acquired: {obs_path}")
                    else:
                        return None
                else:
                    return None

            # Try loading with date as index first (MOD16 format)
            try:
                obs_df = pd.read_csv(obs_path, index_col=0, parse_dates=True)
                if isinstance(obs_df.index, pd.DatetimeIndex):
                    data_col = self._get_observed_data_column(obs_df.columns)
                    if data_col:
                        obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')
                        obs_data = obs_data.dropna()
                        self.logger.info(f"Loaded {len(obs_data)} ET observations from {obs_path.name}")
                        return obs_data
            except Exception:
                pass

            # Fall back to standard loading
            obs_df = pd.read_csv(obs_path)
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            if not date_col or not data_col:
                self.logger.warning(f"Could not find date or data columns in {obs_path}")
                return None

            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')

            # Apply quality control (mainly for FluxNet)
            if self.use_quality_control and self.obs_source == 'fluxnet':
                obs_data = self._apply_quality_control(obs_df, obs_data, data_col)

            obs_data = obs_data.dropna()

            # Temporal aggregation (for high-frequency data)
            if self.temporal_aggregation == 'daily_mean':
                obs_daily = obs_data.resample('D').mean()
            elif self.temporal_aggregation == 'daily_sum':
                obs_daily = obs_data.resample('D').sum()
            else:
                obs_daily = obs_data

            self.logger.info(f"Loaded {len(obs_daily)} ET observations from {obs_path.name}")
            return obs_daily.dropna()

        except Exception as e:
            self.logger.error(f"Error loading observed ET data: {str(e)}")
            return None

    def _try_acquire_mod16_data(self):
        """Attempt to acquire MOD16 data via NASA AppEEARS if missing.

        MOD16A2 is MODIS Terra satellite land surface ET product:
        - Resolution: 1 km × 1 km
        - Frequency: 8-day composites (46 composites per year)
        - Coverage: Global (80°N - 60°S)
        - Variable: scalarET (kg m⁻² per 8-day period)

        Acquisition Workflow:
            1. Initialize MODISETHandler with project config
            2. Call acquire() to download raw 8-day HDF files from AppEEARS
            3. Call process() to extract, aggregate, and save as daily CSV

        Requirements:
            - NASA Earthdata account credentials in config
            - Internet connection to AppEEARS servers
            - Sufficient disk space (~1-5 GB for full basin-year dataset)

        Output:
            - Preprocessed CSV: {domain}_modis_et_processed.csv
            - Format: datetime index, et_mm_day column (daily aggregated from 8-day)

        Args:
            None (uses self.config, self.project_dir)

        Returns:
            None (logs success/failure, saves to file)

        Raises:
            None (catches exceptions, logs warning, continues)
        """
        try:
            from symfluence.data.observation.handlers.modis_et import MODISETHandler

            handler = MODISETHandler(self.config, self.logger)
            raw_dir = handler.acquire()
            handler.process(raw_dir)
            self.logger.info("MOD16 ET data acquisition completed")
        except Exception as e:
            self.logger.warning(f"Could not acquire MOD16 data: {e}")

    def _try_acquire_fluxnet_data(self):
        """Attempt to acquire FluxNet tower observations if missing.

        FluxNet provides half-hourly eddy-covariance measurements from 800+ towers
        globally. ET and latent heat measured via eddy covariance technique.

        Tower Network Coverage:
            - FLUXNET2015: ~200 tower-years of standardized data
            - FLUXNET Tier 1: Fully processed, quality-controlled (public)
            - FLUXNET Tier 2: Preliminary processing, embargo periods
            - Regional networks: AmeriFlux, AsiaFlux, ICOS, etc.

        Acquisition Workflow:
            1. Initialize FLUXNETETAcquirer with config and output directory
            2. Determine target station from config.evaluation.fluxnet.station
            3. Download half-hourly data from FLUXNET portal
            4. Extract ET or latent heat columns
            5. Gap-fill using standard FLUXNET products (LE_F_MDS, LE_CORR)
            6. Save as daily CSV

        Output Format:
            - Preprocessed CSV: {domain}_fluxnet_et_processed.csv
            - Columns: DateTime, ET_mm_day or LE_W_m2, QC flags
            - Frequency: Daily or sub-daily (depends on processing)

        Args:
            None (uses self.config, self.project_dir)

        Returns:
            Optional[Path]: Path to downloaded file if successful, None if failed

        Raises:
            None (catches exceptions, logs warning, returns None)
        """
        try:
            from symfluence.data.acquisition.handlers.fluxnet import FLUXNETETAcquirer

            output_dir = self.project_dir / "observations" / "et" / "preprocessed"
            output_dir.mkdir(parents=True, exist_ok=True)

            acquirer = FLUXNETETAcquirer(self.config, self.logger)
            result_path = acquirer.download(output_dir)
            self.logger.info(f"FLUXNET ET data acquisition completed: {result_path}")
            return result_path
        except Exception as e:
            self.logger.warning(f"Could not acquire FLUXNET data: {e}")
            return None
    
    def _apply_quality_control(self, obs_df: pd.DataFrame, obs_data: pd.Series, data_col: str) -> pd.Series:
        """Apply FluxNet quality control filtering (source-specific).

        FluxNet provides quality control flags for each measurement indicating
        the reliability/gap-filling status. This method filters observations
        based on configurable quality threshold.

        FluxNet QC Flag Scale:
            0: Measured data (best quality, no gap-filling)
            1: Good gap-filled data (most gaps filled with interpolation/regression)
            2: Moderate gap-filled data (larger gaps or assumptions required)
            3+: Poor quality (rarely used, should be excluded)

        Quality Control Variables:
            - LE_F_MDS_QC: QC flag for latent heat flux (used for both ET and LE)
            - Same flag applies to both optimization targets ('et', 'latent_heat')

        Configuration:
            ET_MAX_QUALITY_FLAG: Retain data with QC ≤ threshold
            - Default: 2 (good + moderate = most data retained)
            - Strict: 0 (measured only = 30-40% data loss)
            - Relaxed: 3 (include all gap-filled = keep suspicious values)

        Filtering Logic:
            1. Locate QC column (LE_F_MDS_QC) in input DataFrame
            2. Convert QC values to numeric (handles errors)
            3. Create boolean mask: qc_flags ≤ max_quality_flag
            4. Return observations passing QC filter
            5. On error: Log warning, return unfiltered data

        Args:
            obs_df: DataFrame with DateTime index and QC columns
            obs_data: Series of ET/latent heat measurements to filter
            data_col: Name of data column (unused, kept for signature compatibility)

        Returns:
            pd.Series: QC-filtered observations (or original if filtering fails)

        Raises:
            None (logs warning, returns unfiltered data on error)
        """
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
        """Determine if observations require routing module for spatial analysis.

        Evapotranspiration is a gridded (or point-based for towers) process that
        does NOT require streamflow routing models. ET observations are compared
        directly to model outputs without upstream/downstream propagation.

        Returns:
            bool: False (ET evaluator never requires routing)
        """
        return False
