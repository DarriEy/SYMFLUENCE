"""
GR model preprocessor.

Handles data preparation, PET calculation, snow module setup, and file organization.
Supports both lumped and distributed spatial modes.
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.utils.data.utilities.variable_utils import VariableHandler
from symfluence.utils.common.constants import UnitConversion
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin, ObservationLoaderMixin, DatasetBuilderMixin
from symfluence.utils.common.geospatial_utils import GeospatialUtilsMixin
from symfluence.utils.exceptions import ModelExecutionError, symfluence_error_handler

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
    Preprocessor for the GR family of models (initially GR4J).

    Handles data preparation, PET calculation, snow module setup, and file organization.
    Supports both lumped and distributed spatial modes.
    Inherits common functionality from BaseModelPreProcessor, PET calculations from PETCalculatorMixin,
    geospatial utilities from GeospatialUtilsMixin, and observation loading from ObservationLoaderMixin.

    Attributes:
        config: Configuration settings for GR models (inherited)
        logger: Logger object for recording processing information (inherited)
        project_dir: Directory for the current project (inherited)
        setup_dir: Directory for GR setup files (inherited)
        domain_name: Name of the domain being processed (inherited)
        spatial_mode: Spatial mode ('lumped' or 'distributed')
    """

    def _get_model_name(self) -> str:
        """Return model name for GR."""
        return "GR"

    def __init__(self, config: Dict[str, Any], logger: Any):
        if not HAS_RPY2:
            raise ImportError(
                "GR models require R and rpy2. "
                "Please install R and rpy2, or use a different model. "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # Initialize base class
        super().__init__(config, logger)

        # GR-specific paths
        self.forcing_gr_path = self.project_dir / 'forcing' / 'GR_input'

        # GR-specific catchment configuration - maintain compatibility with old code pattern
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')

        # Phase 3: Use typed config when available
        if self.typed_config:
            self.catchment_name = self.typed_config.paths.catchment_shp_name
            if self.catchment_name == 'default' or self.catchment_name is None:
                discretization = self.typed_config.domain.discretization
                self.catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
            self.spatial_mode = self.typed_config.model.gr.spatial_mode if self.typed_config.model.gr else 'lumped'
        else:
            self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
            if self.catchment_name == 'default' or self.catchment_name is None:
                discretization = self.config.get('DOMAIN_DISCRETIZATION')
                self.catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
            self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')

    def run_preprocessing(self):
        """
        Run the complete GR preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info(f"Starting GR preprocessing in {self.spatial_mode} mode")
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """GR-specific forcing data preparation (template hook)."""
        self.prepare_forcing_data()

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped and distributed modes.
        """
        try:
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")

            # Open and concatenate all forcing files
            ds = xr.open_mfdataset(forcing_files)

            # Subset to simulation window using base class method
            ds = self.subset_to_simulation_time(ds, "Forcing")

            variable_handler = VariableHandler(
                config=self.config,
                logger=self.logger,
                dataset=self.config.get('FORCING_DATASET'),
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
        """Prepare lumped forcing data (existing implementation)"""
        # Convert forcing data to daily resolution
        with xr.set_options(use_flox=False):
            ds = ds.resample(time='D').mean()

        try:
            ds['temp'] = ds['airtemp'] - 273.15
            ds['pr'] = ds['pptrate'] * 86400
        except:
            pass

        # Load streamflow observations
        obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"

        # Read observations
        obs_df = pd.read_csv(obs_path)
        obs_df['time'] = pd.to_datetime(obs_df['datetime'])
        obs_df = obs_df.drop('datetime', axis=1)
        obs_df.set_index('time', inplace=True)
        obs_df.index = obs_df.index.tz_localize(None)
        obs_daily = obs_df.resample('D').mean()

        # Get area from river basins shapefile
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default' or basin_name is None:
            basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = basin_dir / basin_name
        basin_gdf = gpd.read_file(basin_path)

        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area from GRU_area: {area_km2:.2f} km2")

        # Convert units from cms to mm/day
        obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * UnitConversion.MM_DAY_TO_CMS

        # Create observation dataset
        obs_ds = xr.Dataset(
            {'q_obs': ('time', obs_daily['discharge_mmday'].values)},
            coords={'time': obs_daily.index.values}
        )

        # Read catchment and get centroid (using inherited GeospatialUtilsMixin)
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self.calculate_catchment_centroid(catchment)

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
        """Prepare distributed forcing data for each HRU"""

        # Load catchment to get HRU information
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)

        # Check if we have HRU dimension in forcing data
        if 'hru' not in ds.dims:
            self.logger.warning("No HRU dimension found in forcing data, creating distributed data from lumped")
            # Replicate lumped data to all HRUs
            n_hrus = len(catchment)
            ds = ds.expand_dims(hru=n_hrus)

        # Convert to daily resolution
        with xr.set_options(use_flox=False):
            ds = ds.resample(time='D').mean()

        try:
            ds['temp'] = ds['airtemp'] - 273.15
            ds['pr'] = ds['pptrate'] * 86400
        except:
            pass

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
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default' or basin_name is None:
                basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
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
        catchment_geo = catchment.to_crs(epsg=4326)

        # Get centroids for each HRU
        hru_centroids = catchment_geo.geometry.centroid
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
