"""
cFUSE Model Preprocessor.

Prepares forcing data (precipitation, temperature, PET) for cFUSE model execution.
Supports both lumped and distributed modes.
"""

from typing import Dict, Any, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.utilities import ForcingDataProcessor
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.constants import UnitConversion


@ModelRegistry.register_preprocessor('CFUSE')
class CFUSEPreprocessor(BaseModelPreProcessor):
    """
    Preprocessor for cFUSE model.

    Prepares forcing data including:
    - Precipitation (mm/day)
    - Temperature (degC)
    - Potential evapotranspiration (mm/day)

    Supports lumped mode (single time series) and distributed mode
    (per-HRU time series for batch processing).
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        logger: logging.Logger,
        params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize cFUSE preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance
            params: Optional parameter overrides
        """
        super().__init__(config, logger)

        self.params = params or {}

        # cFUSE-specific paths
        self.cfuse_setup_dir = self.setup_dir
        self.cfuse_forcing_dir = self.forcing_dir
        self.cfuse_results_dir = self.project_dir / 'simulations' / self.experiment_id / 'CFUSE'

        # Determine spatial mode
        configured_mode = self._get_config_value(
            lambda: self.config.model.cfuse.spatial_mode if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            'auto'
        )

        if configured_mode in (None, 'auto', 'default'):
            if self.domain_definition_method == 'delineate':
                self.spatial_mode = 'distributed'
            else:
                self.spatial_mode = 'lumped'
        else:
            self.spatial_mode = configured_mode

        # Enable routing
        self.enable_routing = self._get_config_value(
            lambda: self.config.model.cfuse.enable_routing if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            False
        )

        # Timestep configuration
        self.timestep_days = self._get_config_value(
            lambda: self.config.model.cfuse.timestep_days if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            1.0
        )

        # Model structure
        self.model_structure = self._get_config_value(
            lambda: self.config.model.cfuse.model_structure if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            'vic'
        )

        # Snow enabled
        self.enable_snow = self._get_config_value(
            lambda: self.config.model.cfuse.enable_snow if self.config.model and hasattr(self.config.model, 'cfuse') and self.config.model.cfuse else None,
            True
        )

    def _get_model_name(self) -> str:
        """Return model name for cFUSE."""
        return "CFUSE"

    def run_preprocessing(self) -> bool:
        """
        Run cFUSE preprocessing workflow.

        Creates forcing data files in the appropriate format for cFUSE model execution.

        Returns:
            True if preprocessing completed successfully.
        """
        self.logger.info(f"Starting cFUSE preprocessing in {self.spatial_mode} mode")

        # Create directories
        self.create_directories()

        # Prepare forcing data
        if self.spatial_mode == 'lumped':
            success = self._prepare_lumped_forcing()
        else:
            success = self._prepare_distributed_forcing()

        if success:
            self.logger.info("cFUSE preprocessing completed successfully")
        else:
            self.logger.error("cFUSE preprocessing failed")

        return success

    def _prepare_lumped_forcing(self) -> bool:
        """
        Prepare forcing data for lumped cFUSE simulation.

        Creates a single time series for the entire catchment.

        Returns:
            True if successful.
        """
        self.logger.info("Preparing lumped forcing data for cFUSE")

        try:
            # Load basin-averaged forcing
            forcing_ds = self._load_basin_averaged_forcing()
            if forcing_ds is None:
                return False

            # Extract variables
            time = pd.to_datetime(forcing_ds.time.values)

            # Precipitation (check various naming conventions)
            precip_vars = ['pr', 'precip', 'pptrate', 'prcp', 'precipitation']
            precip = None
            precip_var_name = None
            for var in precip_vars:
                if var in forcing_ds:
                    precip = forcing_ds[var].values
                    precip_var_name = var
                    self.logger.info(f"Using precipitation variable: {var}")
                    break
            if precip is None:
                self.logger.error(f"Precipitation variable not found. Available: {list(forcing_ds.data_vars)}")
                return False

            # Convert precipitation units if needed
            precip_units = forcing_ds[precip_var_name].attrs.get('units', '').lower()
            if 'mm s' in precip_units or 'mm/s' in precip_units or 's-1' in precip_units:
                precip = precip * UnitConversion.SECONDS_PER_DAY
                self.logger.info("Converted precipitation from mm/s to mm/day")
            elif np.nanmean(precip) < 0.01 and np.nanmax(precip) < 0.1:
                precip = precip * UnitConversion.SECONDS_PER_DAY
                self.logger.info("Precipitation values appear to be in mm/s, converting to mm/day")

            # Temperature (check various naming conventions)
            temp_vars = ['temp', 'tas', 'airtemp', 'tair', 'temperature', 'tmean']
            temp = None
            for var in temp_vars:
                if var in forcing_ds:
                    temp = forcing_ds[var].values
                    self.logger.info(f"Using temperature variable: {var}")
                    break
            if temp is None:
                self.logger.error(f"Temperature variable not found. Available: {list(forcing_ds.data_vars)}")
                return False

            # Convert temperature from K to C if needed
            if np.nanmean(temp) > 100:  # Likely Kelvin
                temp = temp - 273.15

            # Potential evapotranspiration
            pet = self._get_pet(forcing_ds, temp, time)
            if pet is None:
                return False

            # Handle temporal resolution
            timestep_config = self.get_timestep_config()
            if timestep_config['time_label'] == 'hourly':
                # Resample hourly data to daily for cFUSE
                self.logger.info("Resampling hourly data to daily for cFUSE")
                forcing_df = pd.DataFrame({
                    'time': time,
                    'precip': precip.flatten(),
                    'temp': temp.flatten(),
                    'pet': pet.flatten()
                }).set_index('time')

                forcing_df = forcing_df.resample('D').agg({
                    'precip': 'sum',
                    'temp': 'mean',
                    'pet': 'mean'
                })
                forcing_df = forcing_df.reset_index()
            else:
                # Daily input data
                forcing_df = pd.DataFrame({
                    'time': time,
                    'precip': precip.flatten(),
                    'temp': temp.flatten(),
                    'pet': pet.flatten()
                })

            # Subset to simulation time window
            time_window = self.get_simulation_time_window()
            if time_window:
                start_time, end_time = time_window
                forcing_df['time'] = pd.to_datetime(forcing_df['time'])
                forcing_df = forcing_df[
                    (forcing_df['time'] >= start_time) &
                    (forcing_df['time'] <= end_time)
                ]

            # Save forcing file
            output_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing.csv"
            forcing_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved lumped forcing to: {output_file}")

            # Also save as NetCDF for consistency
            self._save_forcing_netcdf(forcing_df, 'lumped')

            # Load and save observations if available
            self._prepare_observations()

            return True

        except Exception as e:
            self.logger.error(f"Error preparing lumped forcing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _prepare_distributed_forcing(self) -> bool:
        """
        Prepare forcing data for distributed cFUSE simulation.

        Creates per-HRU time series for batch processing.

        Returns:
            True if successful.
        """
        self.logger.info("Preparing distributed forcing data for cFUSE")

        try:
            # Load gridded forcing
            forcing_ds = self._load_merged_forcing()
            if forcing_ds is None:
                self.logger.warning("Merged forcing not found, falling back to basin-averaged")
                return self._prepare_lumped_forcing()

            # Get HRU information
            catchment_path = self.get_catchment_path()
            if not catchment_path.exists():
                self.logger.error(f"Catchment shapefile not found: {catchment_path}")
                return False

            import geopandas as gpd
            catchment = gpd.read_file(catchment_path)
            n_hrus = len(catchment)
            self.logger.info(f"Processing forcing for {n_hrus} HRUs")

            # Extract time coordinate
            time = pd.to_datetime(forcing_ds.time.values)

            # Initialize arrays for HRU data
            hru_ids = catchment[self.hru_id_col].values if self.hru_id_col in catchment.columns else np.arange(n_hrus) + 1

            # For distributed mode, extract forcing for each HRU
            if 'hru' in forcing_ds.dims:
                # Forcing already per-HRU
                precip = forcing_ds['pr'].values if 'pr' in forcing_ds else forcing_ds['precip'].values
                temp = self._get_temperature_variable(forcing_ds)
                pet = self._get_pet_distributed(forcing_ds, temp, time)
            else:
                # Need to spatially average forcing to HRUs
                self.logger.info("Spatially averaging gridded forcing to HRUs")
                precip, temp, pet = self._spatially_average_to_hrus(forcing_ds, catchment)

            # Convert temperature from K to C if needed
            if np.nanmean(temp) > 100:
                temp = temp - 273.15

            # Handle temporal resolution (resample to daily if hourly)
            timestep_config = self.get_timestep_config()
            if timestep_config['time_label'] == 'hourly':
                self.logger.info("Resampling hourly data to daily for cFUSE")
                precip, temp, pet, time = self._resample_to_daily(precip, temp, pet, time)

            # Create xarray Dataset for distributed forcing
            ds = xr.Dataset(
                data_vars={
                    'precip': (['time', 'hru'], precip),
                    'temp': (['time', 'hru'], temp),
                    'pet': (['time', 'hru'], pet),
                    'hru_id': (['hru'], hru_ids.astype(np.int32)),
                },
                coords={
                    'time': time,
                    'hru': np.arange(n_hrus),
                },
                attrs={
                    'model': 'cFUSE',
                    'spatial_mode': 'distributed',
                    'domain': self.domain_name,
                    'n_hrus': n_hrus,
                    'model_structure': self.model_structure,
                    'enable_routing': self.enable_routing,
                    'units_precip': 'mm/day',
                    'units_temp': 'degC',
                    'units_pet': 'mm/day',
                }
            )

            # Subset to simulation time window
            time_window = self.get_simulation_time_window()
            if time_window:
                start_time, end_time = time_window
                ds = ds.sel(time=slice(start_time, end_time))

            # Save distributed forcing
            output_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing_distributed.nc"
            encoding = create_netcdf_encoding(ds, compression=True)
            ds.to_netcdf(output_file, encoding=encoding)
            self.logger.info(f"Saved distributed forcing to: {output_file}")

            # Also save observations
            self._prepare_observations()

            return True

        except Exception as e:
            self.logger.error(f"Error preparing distributed forcing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_basin_averaged_forcing(self) -> Optional[xr.Dataset]:
        """Load basin-averaged forcing data using ForcingDataProcessor."""
        try:
            fdp = ForcingDataProcessor(self.config, self.logger)

            if hasattr(self, 'forcing_basin_path') and self.forcing_basin_path.exists():
                self.logger.info(f"Loading basin-averaged forcing from: {self.forcing_basin_path}")
                ds = fdp.load_forcing_data(self.forcing_basin_path)
                if ds is not None:
                    ds = self.subset_to_simulation_time(ds, "Forcing")
                    return ds

            return self._load_merged_forcing()

        except Exception as e:
            self.logger.error(f"Error loading forcing data: {e}")
            return None

    def _load_merged_forcing(self) -> Optional[xr.Dataset]:
        """Load merged forcing data."""
        merged_file = self.merged_forcing_path / f"{self.domain_name}_merged_forcing.nc"
        if merged_file.exists():
            self.logger.info(f"Loading merged forcing from: {merged_file}")
            ds = xr.open_dataset(merged_file)
            ds = self.subset_to_simulation_time(ds, "Forcing")
            return ds

        self.logger.error(f"No forcing data found at: {merged_file}")
        return None

    def _get_temperature_variable(self, ds: xr.Dataset) -> np.ndarray:
        """Extract temperature variable from dataset."""
        for var in ['temp', 'tas', 'airtemp', 'tair', 'temperature']:
            if var in ds:
                return ds[var].values
        raise ValueError("Temperature variable not found in forcing dataset")

    def _get_pet(
        self,
        ds: xr.Dataset,
        temp: np.ndarray,
        time: pd.DatetimeIndex
    ) -> Optional[np.ndarray]:
        """
        Get or calculate PET (Potential Evapotranspiration).

        Args:
            ds: Forcing dataset
            temp: Temperature array (degC)
            time: Time index

        Returns:
            PET array (mm/day) or None if calculation fails.
        """
        # Check for PET in forcing data
        for var in ['pet', 'pET', 'potEvap', 'evap', 'evspsbl']:
            if var in ds:
                self.logger.info(f"Using PET from forcing data (variable: {var})")
                pet = ds[var].values
                if np.nanmean(np.abs(pet)) < 0.01:  # Likely mm/s
                    pet = pet * UnitConversion.SECONDS_PER_DAY
                return pet.flatten() if pet.ndim > 1 else pet

        # Calculate PET using Hamon method
        return self._calculate_hamon_pet(temp.flatten(), time)

    def _get_pet_distributed(
        self,
        ds: xr.Dataset,
        temp: np.ndarray,
        time: pd.DatetimeIndex
    ) -> np.ndarray:
        """Get or calculate PET for distributed mode."""
        for var in ['pet', 'pET', 'potEvap', 'evap', 'evspsbl']:
            if var in ds:
                pet = ds[var].values
                if np.nanmean(np.abs(pet)) < 0.01:
                    pet = pet * UnitConversion.SECONDS_PER_DAY
                return pet

        # Calculate PET for each HRU using catchment-mean temperature
        mean_temp = np.nanmean(temp, axis=1)
        pet_1d = self._calculate_hamon_pet(mean_temp, time)
        return np.broadcast_to(pet_1d[:, np.newaxis], temp.shape)

    def _calculate_hamon_pet(
        self,
        temp: np.ndarray,
        time: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Calculate PET using Hamon method.

        Args:
            temp: Daily mean temperature (degC)
            time: Time index for day length calculation

        Returns:
            PET array (mm/day)
        """
        self.logger.info("Calculating PET using Hamon method")

        # Get latitude for day length calculation
        try:
            import geopandas as gpd
            catchment = gpd.read_file(self.get_catchment_path())
            centroid = catchment.to_crs(epsg=4326).union_all().centroid
            lat = centroid.y
        except Exception:
            lat = 45.0  # Default mid-latitude
            self.logger.warning(f"Using default latitude {lat} for PET calculation")

        # Calculate day length
        day_of_year = np.asarray(time.dayofyear)
        lat_rad = np.deg2rad(lat)

        # Solar declination
        decl = 0.409 * np.sin(2 * np.pi / 365 * day_of_year - 1.39)

        # Sunset hour angle
        sunset_angle = np.arccos(-np.tan(lat_rad) * np.tan(decl))
        sunset_angle = np.clip(sunset_angle, 0, np.pi)

        # Day length in hours
        day_length = 24 / np.pi * sunset_angle

        # Saturated vapor pressure (kPa)
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))

        # Hamon PET (mm/day)
        pet = 0.55 * (day_length / 12) ** 2 * es
        pet = np.maximum(pet, 0.0)

        return pet

    def _spatially_average_to_hrus(
        self,
        ds: xr.Dataset,
        catchment
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spatially average gridded forcing to HRUs.

        Args:
            ds: Gridded forcing dataset
            catchment: GeoDataFrame with HRU geometries

        Returns:
            Tuple of (precip, temp, pet) arrays with shape (time, n_hrus)
        """
        import rasterio
        from rasterio.features import geometry_mask

        n_hrus = len(catchment)
        n_times = len(ds.time)

        # Get spatial coordinates
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lats = ds.lat.values
            lons = ds.lon.values
        elif 'y' in ds.coords and 'x' in ds.coords:
            lats = ds.y.values
            lons = ds.x.values
        else:
            raise ValueError("Cannot find spatial coordinates in forcing dataset")

        # Create affine transform
        res_lat = np.abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
        res_lon = np.abs(lons[1] - lons[0]) if len(lons) > 1 else 0.1
        transform = rasterio.transform.from_origin(
            lons.min() - res_lon/2,
            lats.max() + res_lat/2,
            res_lon,
            res_lat
        )

        # Extract variables
        precip_var = 'pr' if 'pr' in ds else 'precip'
        temp_var = self._find_var(ds, ['temp', 'tas', 'airtemp', 'tair'])

        precip_grid = ds[precip_var].values
        temp_grid = ds[temp_var].values

        # Initialize output arrays
        precip_hru = np.zeros((n_times, n_hrus))
        temp_hru = np.zeros((n_times, n_hrus))

        # Ensure CRS match
        catchment_reproj = catchment.to_crs(epsg=4326)

        # Spatial average for each HRU
        for i, geom in enumerate(catchment_reproj.geometry):
            try:
                mask = geometry_mask(
                    [geom],
                    out_shape=(len(lats), len(lons)),
                    transform=transform,
                    invert=True
                )

                for t in range(n_times):
                    precip_masked = np.ma.masked_array(precip_grid[t], ~mask)
                    temp_masked = np.ma.masked_array(temp_grid[t], ~mask)

                    precip_hru[t, i] = np.ma.mean(precip_masked)
                    temp_hru[t, i] = np.ma.mean(temp_masked)

            except Exception as e:
                self.logger.warning(f"Error averaging HRU {i}: {e}. Using catchment mean.")
                precip_hru[:, i] = np.nanmean(precip_grid, axis=(1, 2))
                temp_hru[:, i] = np.nanmean(temp_grid, axis=(1, 2))

        # Calculate PET (broadcast across HRUs)
        mean_temp = np.nanmean(temp_hru, axis=1)
        time_idx = pd.to_datetime(ds.time.values)
        pet_1d = self._calculate_hamon_pet(mean_temp, time_idx)
        pet_hru = np.broadcast_to(pet_1d[:, np.newaxis], (n_times, n_hrus))

        return precip_hru, temp_hru, pet_hru.copy()

    def _find_var(self, ds: xr.Dataset, candidates: list) -> str:
        """Find first matching variable name."""
        for var in candidates:
            if var in ds:
                return var
        raise ValueError(f"None of {candidates} found in dataset")

    def _resample_to_daily(
        self,
        precip: np.ndarray,
        temp: np.ndarray,
        pet: np.ndarray,
        time: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Resample hourly data to daily."""
        n_hrus = precip.shape[1] if precip.ndim > 1 else 1

        if n_hrus == 1:
            df = pd.DataFrame({
                'precip': precip.flatten(),
                'temp': temp.flatten(),
                'pet': pet.flatten()
            }, index=time)

            df_daily = df.resample('D').agg({
                'precip': 'sum',
                'temp': 'mean',
                'pet': 'mean'
            })

            return (
                df_daily['precip'].values[:, np.newaxis],
                df_daily['temp'].values[:, np.newaxis],
                df_daily['pet'].values[:, np.newaxis],
                df_daily.index
            )
        else:
            # Multi-HRU case
            precip_daily = []
            temp_daily = []
            pet_daily = []

            for i in range(n_hrus):
                df = pd.DataFrame({
                    'precip': precip[:, i],
                    'temp': temp[:, i],
                    'pet': pet[:, i]
                }, index=time)

                df_daily = df.resample('D').agg({
                    'precip': 'sum',
                    'temp': 'mean',
                    'pet': 'mean'
                })

                precip_daily.append(df_daily['precip'].values)
                temp_daily.append(df_daily['temp'].values)
                pet_daily.append(df_daily['pet'].values)

            return (
                np.column_stack(precip_daily),
                np.column_stack(temp_daily),
                np.column_stack(pet_daily),
                df_daily.index
            )

    def _save_forcing_netcdf(self, forcing_df: pd.DataFrame, mode: str) -> None:
        """Save forcing as NetCDF file."""
        ds = xr.Dataset(
            data_vars={
                'precip': (['time'], forcing_df['precip'].values),
                'temp': (['time'], forcing_df['temp'].values),
                'pet': (['time'], forcing_df['pet'].values),
            },
            coords={
                'time': pd.to_datetime(forcing_df['time']) if 'time' in forcing_df.columns else forcing_df.index,
            },
            attrs={
                'model': 'cFUSE',
                'spatial_mode': mode,
                'domain': self.domain_name,
                'model_structure': self.model_structure,
                'enable_snow': self.enable_snow,
                'units_precip': 'mm/day',
                'units_temp': 'degC',
                'units_pet': 'mm/day',
            }
        )

        output_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(output_file, encoding=encoding)
        self.logger.info(f"Saved forcing NetCDF to: {output_file}")

    def _prepare_observations(self) -> None:
        """Prepare observation data for validation/calibration."""
        obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        obs_file = obs_dir / f"{self.domain_name}_streamflow_processed.csv"

        if obs_file.exists():
            self.logger.info(f"Observations available at: {obs_file}")

            obs_df = pd.read_csv(obs_file)
            output_obs = self.cfuse_forcing_dir / f"{self.domain_name}_observations.csv"
            obs_df.to_csv(output_obs, index=False)
        else:
            self.logger.warning(f"No observation file found at: {obs_file}")

    def load_forcing_and_obs(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Load prepared forcing and observation data.

        Returns:
            Tuple of (forcing_dict, observations)
            forcing_dict contains 'precip', 'temp', 'pet' arrays
        """
        if self.spatial_mode == 'distributed':
            forcing_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing_distributed.nc"
        else:
            forcing_file = self.cfuse_forcing_dir / f"{self.domain_name}_cfuse_forcing.nc"

        if not forcing_file.exists():
            raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

        ds = xr.open_dataset(forcing_file)

        forcing = {
            'precip': ds['precip'].values,
            'temp': ds['temp'].values,
            'pet': ds['pet'].values,
            'time': pd.to_datetime(ds.time.values),
        }

        # Load observations if available
        obs_file = self.cfuse_forcing_dir / f"{self.domain_name}_observations.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            observations = obs_df.iloc[:, 0].values
        else:
            observations = None

        return forcing, observations
