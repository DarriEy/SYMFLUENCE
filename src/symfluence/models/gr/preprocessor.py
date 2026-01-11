"""
GR model preprocessor.

Handles data preparation, PET calculation, snow module setup, and file organization.
Supports both lumped and distributed spatial modes.
Uses shared utilities for forcing data processing and data quality handling.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.data.utils.variable_utils import VariableHandler
from symfluence.core.constants import UnitConversion
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin, ObservationLoaderMixin, DatasetBuilderMixin
from ..utilities import ForcingDataProcessor, DataQualityHandler
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

# Optional R/rpy2 support - only needed for GR models
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except (ImportError, ValueError) as e:
    HAS_RPY2 = False


@ModelRegistry.register_preprocessor('GR')
class GRPreProcessor(BaseModelPreProcessor, PETCalculatorMixin, GeospatialUtilsMixin, ObservationLoaderMixin, DatasetBuilderMixin):
    """
    Preprocessor for the GR family of models (GR4J, GR5J, GR6J).

    Handles complete preprocessing workflow for GR models including forcing data processing,
    PET calculation, optional snow module setup, and preparation of input data for R-based
    execution. GR models are parsimonious lumped/semi-distributed rainfall-runoff models
    developed by INRAE (France).

    Key Operations:
        - Process forcing data (precipitation, temperature, PET)
        - Calculate potential evapotranspiration using multiple methods
        - Prepare input data for GR model execution via rpy2/R interface
        - Handle both lumped and distributed (HRU-based) spatial configurations
        - Configure optional snow module (CemaNeige) for snowmelt modeling
        - Quality control and gap-filling of forcing data
        - Load and align observation data for calibration/evaluation

    Workflow Steps:
        1. Initialize paths and validate R/rpy2 installation
        2. Determine spatial mode (lumped vs. distributed)
        3. Process forcing data and calculate PET
        4. Aggregate forcing data by spatial units (if distributed)
        5. Handle data quality issues (gaps, outliers)
        6. Prepare input matrices for R model execution
        7. Configure snow module parameters (if enabled)
        8. Save processed data in R-compatible format

    Supported Spatial Modes:
        - Lumped: Single catchment-averaged inputs
        - Distributed: Multiple HRUs with area-weighted aggregation
        - Auto: Automatically detect based on domain definition method

    GR Model Variants:
        - GR4J: 4-parameter daily model (basic rainfall-runoff)
        - GR5J: 5-parameter daily model (includes groundwater exchange)
        - GR6J: 6-parameter daily model (includes parallel routing)
        - CemaNeige: Optional 2-parameter snow module

    PET Calculation Methods:
        - Oudin: Simple temperature-based method (GR model default)
        - Hamon: Temperature and daylight-based
        - Priestley-Taylor: Radiation-based method

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        PETCalculatorMixin: Potential evapotranspiration calculation methods
        GeospatialUtilsMixin: Spatial operations (area calculation, centroid)
        ObservationLoaderMixin: Observation data loading capabilities
        DatasetBuilderMixin: NetCDF dataset construction utilities

    Attributes:
        config (SymfluenceConfig): Typed configuration object
        logger: Logger object for recording processing information
        project_dir (Path): Directory for the current project
        forcing_gr_path (Path): Directory for GR input files
        catchment_path (Path): Path to catchment shapefile
        spatial_mode (str): Spatial configuration ('lumped' or 'distributed')
        forcing_processor (ForcingDataProcessor): Handles forcing data transformation
        quality_handler (DataQualityHandler): Handles data quality control

    Requirements:
        - R programming language (>= 4.0)
        - rpy2 Python package (R-Python interface)
        - airGR R package (GR model implementations)

    Example:
        >>> from symfluence.models.gr.preprocessor import GRPreProcessor
        >>> preprocessor = GRPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates GR input files in: project_dir/forcing/GR_input/
        # Generates: forcing_data.csv, catchment_attributes.csv

    Raises:
        ImportError: If R or rpy2 is not installed
        ModelExecutionError: If preprocessing fails
    """

    def _get_model_name(self) -> str:
        """Return model name for GR."""
        return "GR"

    def __init__(self, config, logger):
        if not HAS_RPY2:
            raise ImportError(
                "GR models require R and rpy2. "
                "Please install R and rpy2, or use a different model. "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # Initialize base class (handles typed config validation)
        super().__init__(config, logger)

        # GR-specific paths
        self.forcing_gr_path = self.project_dir / 'forcing' / 'GR_input'

        # GR-specific catchment configuration
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')

        # Use typed config accessor
        self.catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_name
        )
        if self.catchment_name == 'default' or self.catchment_name is None:
            discretization = self._get_config_value(
                lambda: self.config.domain.discretization
            )
            self.catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        # Resolve spatial mode
        # 1. Check for explicit configuration
        configured_mode = self._get_config_value(
            lambda: self.config.model.gr.spatial_mode if self.config.model and self.config.model.gr else None
        )

        # 2. If 'auto' or not set, infer from domain definition method
        if configured_mode in (None, 'auto', 'default'):
            # Infer from domain definition method
            if self.domain_definition_method == 'delineate':
                self.spatial_mode = 'distributed'
                self.logger.info(f"GR spatial mode auto-detected as 'distributed' (DOMAIN_DEFINITION_METHOD: {self.domain_definition_method})")
            else:
                self.spatial_mode = 'lumped'
                self.logger.info(f"GR spatial mode auto-detected as 'lumped' (DOMAIN_DEFINITION_METHOD: {self.domain_definition_method})")
        else:
            # 3. Use explicit config
            self.spatial_mode = configured_mode
            self.logger.info(f"GR spatial mode set to '{self.spatial_mode}' from configuration")

    def run_preprocessing(self):
        """
        Run the complete GR preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info(f"Starting GR preprocessing in {self.spatial_mode} mode")
        # GR does not ship base settings; avoid noisy warning.
        self.copy_base_settings = lambda *args, **kwargs: None
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """GR-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data()

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped and distributed modes.

        Uses shared ForcingDataProcessor for loading and subsetting.
        """
        try:
            # Use shared ForcingDataProcessor for loading
            fdp = ForcingDataProcessor(self.config, self.logger)
            ds = fdp.load_forcing_data(self.forcing_basin_path)

            # Subset to simulation window using base class method
            ds = self.subset_to_simulation_time(ds, "Forcing")

            variable_handler = VariableHandler(
                config=self.config_dict,
                logger=self.logger,
                dataset=self.config_dict.get('FORCING_DATASET'),
                model='GR'
            )

            # Process variables
            ds_variable_handler = variable_handler.process_forcing_data(ds)
            ds = ds_variable_handler

            # Handle spatial organization based on mode
            if self.spatial_mode == 'lumped':
                self.logger.info("Preparing lumped forcing data")
                ds = ds.mean(dim='hru') if 'hru' in ds.dims else ds
                return self._prepare_lumped_forcing(ds)
            elif self.spatial_mode == 'distributed':
                self.logger.info("Preparing distributed forcing data")
                return self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown GR spatial mode: {self.spatial_mode}")

        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    def _prepare_lumped_forcing(self, ds):
        """Prepare lumped forcing data using shared utilities."""
        # Check if we have enough dates for inference
        if len(ds.time) < 3:
            self.logger.warning("Fewer than 3 time steps in forcing. Cannot infer frequency for GR.")
            # Skip resampling
        else:
            # Use shared ForcingDataProcessor for resampling
            fdp = ForcingDataProcessor(self.config, self.logger)
            ds = fdp.resample_to_frequency(ds, target_freq='D', method='mean')

        # Unit conversions already handled by VariableHandler.process_forcing_data()

        # Load streamflow observations
        obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"

        # Read observations
        if obs_path.exists():
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'])
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_df.index = obs_df.index.tz_localize(None)
            obs_daily = obs_df.resample('D').mean()
        else:
            self.logger.warning(f"No streamflow observations found at {obs_path}")
            # Create dummy obs for the forcing period
            obs_daily = pd.DataFrame({'discharge_cms': [0.0] * len(ds.time)}, index=pd.to_datetime(ds.time.values))

        # Get area from river basins shapefile
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        basin_name = self.config_dict.get('RIVER_BASINS_NAME')
        if basin_name == 'default' or basin_name is None:
            method_suffix = self._get_method_suffix()
            basin_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_{method_suffix}.shp"
        basin_path = basin_dir / basin_name
        
        if basin_path.exists():
            basin_gdf = gpd.read_file(basin_path)
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        else:
            self.logger.warning(f"Basin shapefile not found at {basin_path}, using default area")
            area_km2 = 1.0

        self.logger.info(f"Total catchment area from GRU_area: {area_km2:.2f} km2")

        # Convert units from cms to mm/day for GR input
        obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * UnitConversion.MM_DAY_TO_CMS

        # Create observation dataset
        obs_ds = xr.Dataset(
            {'q_obs': ('time', obs_daily['discharge_mmday'].values)},
            coords={'time': obs_daily.index.values}
        )

        # Read catchment and get centroid (using inherited GeospatialUtilsMixin)
        catchment_path = self.get_catchment_path()
        if catchment_path.exists():
            catchment = gpd.read_file(catchment_path)
            mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)
        else:
            self.logger.warning(f"Catchment shapefile not found at {catchment_path}, using default latitude")
            mean_lat = 45.0

        # Calculate PET
        pet = self.calculate_pet_oudin(ds['temp'], mean_lat)

        # Find overlapping time period
        start_time = max(ds.time.min().values, obs_ds.time.min().values)
        end_time = min(ds.time.max().values, obs_ds.time.max().values)

        # Create explicit time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')

        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

        # Create GR forcing data
        gr_forcing = pd.DataFrame({
            'time': time_index.strftime('%Y-%m-%d'),
            'pr': ds['pr'].values,
            'temp': ds['temp'].values,
            'pet': pet.values,
            'q_obs': obs_ds['q_obs'].values
        })

        # Save to CSV
        output_file = self.forcing_gr_path / f"{self.domain_name}_input.csv"
        gr_forcing.to_csv(output_file, index=False)

        self.logger.info(f"Lumped forcing data saved to: {output_file}")
        return output_file

    def _prepare_distributed_forcing(self, ds):
        """Prepare distributed forcing data for each HRU using shared utilities."""

        # Load catchment to get HRU information
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)

        # Check if we have HRU dimension in forcing data
        if 'hru' not in ds.dims:
            self.logger.warning("No HRU dimension found in forcing data, creating distributed data from lumped")
            # Replicate lumped data to all HRUs
            n_hrus = len(catchment)
            ds = ds.expand_dims(hru=n_hrus)

        # Use shared ForcingDataProcessor for resampling
        fdp = ForcingDataProcessor(self.config, self.logger)
        ds = fdp.resample_to_frequency(ds, target_freq='D', method='mean')

        # Unit conversions already handled by VariableHandler.process_forcing_data()

        # Load streamflow observations (at outlet)
        obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"

        if obs_path.exists():
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'])
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_df.index = obs_df.index.tz_localize(None)
            obs_daily = obs_df.resample('D').mean()

            # Get area for unit conversion
            basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
            basin_name = self.config_dict.get('RIVER_BASINS_NAME')
            if basin_name == 'default' or basin_name is None:
                method_suffix = self._get_method_suffix()
                basin_name = f"{self.config_dict.get('DOMAIN_NAME')}_riverBasins_{method_suffix}.shp"
            basin_path = basin_dir / basin_name
            basin_gdf = gpd.read_file(basin_path)

            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * UnitConversion.MM_DAY_TO_CMS
        else:
            self.logger.warning("No streamflow observations found")
            obs_daily = None

        # Calculate PET for each HRU using its centroid latitude
        self.logger.info("Calculating PET for each HRU")

        # Ensure catchment has proper CRS
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)
        
        # Get centroids for each HRU avoiding geographic CRS warning
        if catchment.crs.is_geographic:
            hru_centroids = catchment.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
        else:
            hru_centroids = catchment.geometry.centroid.to_crs(epsg=4326)
            
        hru_lats = hru_centroids.y.values

        # Calculate PET for each HRU
        pet_data = []
        for i, lat in enumerate(hru_lats):
            temp_hru = ds['temp'].isel(hru=i)
            pet_hru = self.calculate_pet_oudin(temp_hru, lat)
            pet_data.append(pet_hru.values)

        # Stack PET data
        pet_array = np.stack(pet_data, axis=1)  # shape: (time, hru)
        pet = xr.DataArray(
            pet_array,
            dims=['time', 'hru'],
            coords={'time': ds.time, 'hru': ds.hru},
            attrs={
                'units': 'mm/day',
                'long_name': 'Potential evapotranspiration (Oudin formula)',
                'standard_name': 'water_potential_evaporation_flux'
            }
        )

        # Find overlapping time period
        start_time = ds.time.min().values
        end_time = ds.time.max().values

        if obs_daily is not None:
            start_time = max(start_time, obs_daily.index.min())
            end_time = min(end_time, obs_daily.index.max())

        time_index = pd.date_range(start=start_time, end=end_time, freq='D')

        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

        if obs_daily is not None:
            obs_daily = obs_daily.reindex(time_index)

        # Save distributed forcing as NetCDF (one file with all HRUs)
        output_file = self.forcing_gr_path / f"{self.domain_name}_input_distributed.nc"

        # Create output dataset
        gr_forcing = xr.Dataset({
            'pr': ds['pr'],
            'temp': ds['temp'],
            'pet': pet
        })

        if obs_daily is not None:
            gr_forcing['q_obs'] = xr.DataArray(
                obs_daily['discharge_mmday'].values,
                dims=['time'],
                coords={'time': time_index}
            )

        # Add HRU metadata
        gr_forcing['hru_id'] = xr.DataArray(
            catchment['GRU_ID'].values if 'GRU_ID' in catchment.columns else np.arange(len(catchment)),
            dims=['hru'],
            attrs={'long_name': 'HRU identifier'}
        )

        gr_forcing['hru_lat'] = xr.DataArray(
            hru_lats,
            dims=['hru'],
            attrs={'long_name': 'HRU centroid latitude', 'units': 'degrees_north'}
        )

        # Save to NetCDF
        encoding = {var: {'zlib': True, 'complevel': 4} for var in gr_forcing.data_vars}
        gr_forcing.to_netcdf(output_file, encoding=encoding)

        self.logger.info(f"Distributed forcing data saved to: {output_file}")
        self.logger.info(f"Number of HRUs: {len(ds.hru)}")

        return output_file
