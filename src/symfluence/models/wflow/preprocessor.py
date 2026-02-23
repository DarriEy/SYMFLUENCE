"""Wflow Model Preprocessor."""
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("WFLOW")
class WflowPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Prepares inputs for a Wflow model run."""

    MODEL_NAME = "WFLOW"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.settings_dir = self.setup_dir
        self.forcing_out_dir = self.setup_dir / "forcing"

        configured_mode = self._get_config_value(
            lambda: self.config.model.wflow.spatial_mode,
            default=None, dict_key='WFLOW_SPATIAL_MODE'
        )
        if configured_mode and configured_mode not in (None, 'auto', 'default'):
            self.spatial_mode = configured_mode
        else:
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped', dict_key='DOMAIN_DEFINITION_METHOD'
            )
            self.spatial_mode = 'distributed' if domain_method == 'delineate' else 'lumped'
        logger.info(f"Wflow spatial mode: {self.spatial_mode}")

    def run_preprocessing(self) -> bool:
        try:
            logger.info("Starting Wflow preprocessing...")
            self._create_directory_structure()
            self._generate_staticmaps()
            self._generate_forcing()
            self._generate_toml_config()
            logger.info("Wflow preprocessing complete.")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Wflow preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        for d in [self.settings_dir, self.forcing_out_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)
        return pd.to_datetime(start_str).to_pydatetime(), pd.to_datetime(end_str).to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        try:
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()
                elev = float(gdf.get('elev_mean', [1500])[0]) if 'elev_mean' in gdf.columns else 1500.0
                return {'lat': lat, 'lon': lon, 'area_m2': area_m2, 'elev': elev}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read catchment properties: {e}")
        return {'lat': 51.17, 'lon': -115.57, 'area_m2': 2.21e9, 'elev': 1500.0}

    def _generate_staticmaps(self) -> None:
        logger.info("Generating Wflow static maps...")
        props = self._get_catchment_properties()
        staticmaps_name = self._get_config_value(
            lambda: self.config.model.wflow.staticmaps_file, default='wflow_staticmaps.nc',
        )
        lat = np.array([props['lat']])
        lon = np.array([props['lon']])
        ds = xr.Dataset(coords={
            'lat': (['y'], lat, {'units': 'degrees_north'}),
            'lon': (['x'], lon, {'units': 'degrees_east'}),
        })
        # Topography
        ds['wflow_dem'] = xr.DataArray(np.array([[props['elev']]]), dims=['y', 'x'])
        ds['wflow_subcatch'] = xr.DataArray(np.array([[1]]), dims=['y', 'x'])
        ds['wflow_river'] = xr.DataArray(np.array([[1]]), dims=['y', 'x'])
        ds['wflow_gauges'] = xr.DataArray(np.array([[1]]), dims=['y', 'x'])
        ds['wflow_riverwidth'] = xr.DataArray(np.array([[10.0]]), dims=['y', 'x'])
        ds['wflow_riverlength'] = xr.DataArray(np.array([[1000.0]]), dims=['y', 'x'])
        # Soil params
        ds['KsatVer'] = xr.DataArray(np.array([[250.0]]), dims=['y', 'x'])
        ds['f'] = xr.DataArray(np.array([[3.0]]), dims=['y', 'x'])
        ds['SoilThickness'] = xr.DataArray(np.array([[2000.0]]), dims=['y', 'x'])
        ds['InfiltCapPath'] = xr.DataArray(np.array([[50.0]]), dims=['y', 'x'])
        ds['RootingDepth'] = xr.DataArray(np.array([[500.0]]), dims=['y', 'x'])
        ds['KsatHorFrac'] = xr.DataArray(np.array([[50.0]]), dims=['y', 'x'])
        ds['PathFrac'] = xr.DataArray(np.array([[0.01]]), dims=['y', 'x'])
        ds['thetaS'] = xr.DataArray(np.array([[0.45]]), dims=['y', 'x'])
        ds['thetaR'] = xr.DataArray(np.array([[0.05]]), dims=['y', 'x'])
        ds['c'] = xr.DataArray(np.array([[10.0]]), dims=['y', 'x'])
        # Snow parameters
        ds['TT'] = xr.DataArray(np.array([[0.0]]), dims=['y', 'x'])
        ds['TTI'] = xr.DataArray(np.array([[1.0]]), dims=['y', 'x'])
        ds['TTM'] = xr.DataArray(np.array([[0.0]]), dims=['y', 'x'])
        ds['Cfmax'] = xr.DataArray(np.array([[3.75]]), dims=['y', 'x'])
        # Infiltration / leakage
        ds['cf_soil'] = xr.DataArray(np.array([[0.038]]), dims=['y', 'x'])
        ds['MaxLeakage'] = xr.DataArray(np.array([[0.0]]), dims=['y', 'x'])
        # Routing
        ds['N_River'] = xr.DataArray(np.array([[0.036]]), dims=['y', 'x'])
        ds['N'] = xr.DataArray(np.array([[0.072]]), dims=['y', 'x'])
        ds['RiverSlope'] = xr.DataArray(np.array([[0.01]]), dims=['y', 'x'])
        ds['Slope'] = xr.DataArray(np.array([[0.05]]), dims=['y', 'x'])
        # LDD (5 = pit/outflow for lumped catchment)
        ds['wflow_ldd'] = xr.DataArray(np.array([[5]], dtype=np.int32), dims=['y', 'x'])
        ds['wflow_cellarea'] = xr.DataArray(np.array([[props['area_m2']]]), dims=['y', 'x'])
        # Vegetation (static LAI, also generate cyclic file)
        ds['LAI'] = xr.DataArray(np.array([[3.0]]), dims=['y', 'x'])
        ds.attrs['title'] = 'Wflow static maps for SYMFLUENCE'
        output_path = self.settings_dir / staticmaps_name
        ds.to_netcdf(output_path)
        logger.info(f"Written static maps to {output_path}")

    def _generate_forcing(self) -> None:
        logger.info("Generating Wflow forcing data...")
        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()
        forcing_path = self._get_forcing_path()
        forcing_files = sorted(forcing_path.glob('*.nc'))
        if not forcing_files:
            forcing_files = sorted(forcing_path.glob('**/*.nc'))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")
        ds_forcing = xr.open_mfdataset(forcing_files, combine='by_coords')
        ds_forcing = ds_forcing.sel(time=slice(str(start_date), str(end_date)))

        lat = np.array([props['lat']])
        lon = np.array([props['lon']])

        precip = self._extract_forcing_var(
            ds_forcing, ['pptrate', 'mtpr', 'tp', 'precipitation', 'PREC', 'precip'],
            props['lat'], props['lon']
        )
        if precip.max() < 0.1:
            precip = precip * 3600.0
        precip = np.maximum(precip, 0.0)

        temp = self._extract_forcing_var(
            ds_forcing, ['t2m', 'temperature', 'TEMP', 'airtemp', 'tas'],
            props['lat'], props['lon']
        )
        if temp.mean() > 100:
            temp = temp - 273.15

        times = pd.DatetimeIndex(ds_forcing.time.values)
        pet = self._estimate_pet_hamon(temp, times, props['lat'])

        ds_out = xr.Dataset(coords={
            'time': times,
            'lat': (['y'], lat),
            'lon': (['x'], lon),
        })
        ds_out['precip'] = xr.DataArray(precip.reshape(-1, 1, 1), dims=['time', 'y', 'x'],
                                        attrs={'units': 'mm/timestep'})
        ds_out['temp'] = xr.DataArray(temp.reshape(-1, 1, 1), dims=['time', 'y', 'x'],
                                      attrs={'units': 'degC'})
        ds_out['pet'] = xr.DataArray(pet.reshape(-1, 1, 1), dims=['time', 'y', 'x'],
                                     attrs={'units': 'mm/timestep'})
        ds_forcing.close()
        output_path = self.forcing_out_dir / 'forcing.nc'
        ds_out.to_netcdf(output_path)
        logger.info(f"Written forcing to {output_path}")

    def _extract_forcing_var(self, ds, var_names, lat, lon):
        for var in var_names:
            if var in ds.data_vars:
                data = ds[var]
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if spatial_dims:
                    try:
                        data = data.sel(
                            **{d: lat if 'lat' in d else lon for d in spatial_dims},
                            method='nearest'
                        )
                    except Exception:  # noqa: BLE001
                        data = data.isel(**{d: 0 for d in spatial_dims})
                return data.values.flatten()
        raise ValueError(f"None of {var_names} found in forcing. Available: {list(ds.data_vars)}")

    def _estimate_pet_hamon(self, temp_c, times, lat_deg):
        doy = np.array([t.timetuple().tm_yday for t in times])
        lat_rad = math.radians(lat_deg)
        decl = 0.4093 * np.sin(2 * np.pi / 365 * doy - 1.405)
        cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1)
        day_length = 24 / np.pi * np.arccos(cos_omega)
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        pet_daily = np.maximum(0.1651 * (day_length / 12.0) * es * 216.7 / (temp_c + 273.3), 0.0)
        timestep_hours = self._get_config_value(
            lambda: self.config.data.forcing_time_step_size,
            default=3600, dict_key='FORCING_TIME_STEP_SIZE'
        )
        if isinstance(timestep_hours, (int, float)) and timestep_hours > 100:
            timestep_hours = timestep_hours / 3600
        return pet_daily * (timestep_hours / 24.0)

    def _get_forcing_path(self) -> Path:
        forcing_path = self._get_config_value(
            lambda: self.config.data.forcing_path, default=None, dict_key='FORCING_PATH'
        )
        if forcing_path and forcing_path != 'default':
            return Path(forcing_path)
        domain_name = self._get_config_value(
            lambda: self.config.domain.name, default='Bow_at_Banff', dict_key='DOMAIN_NAME'
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR'
        )
        return Path(data_dir) / f'domain_{domain_name}' / 'forcing' / 'basin_averaged_data'

    def _generate_toml_config(self) -> None:
        logger.info("Generating Wflow TOML configuration...")
        start_date, end_date = self._get_simulation_dates()
        config_file = self._get_config_value(lambda: self.config.model.wflow.config_file, default='wflow_sbm.toml')
        staticmaps_file = self._get_config_value(lambda: self.config.model.wflow.staticmaps_file, default='wflow_staticmaps.nc')
        _output_file = self._get_config_value(lambda: self.config.model.wflow.output_file, default='output.nc')  # noqa: F841

        domain_name = self._get_config_value(lambda: self.config.domain.name, default='', dict_key='DOMAIN_NAME')
        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1', dict_key='EXPERIMENT_ID')
        data_dir = self._get_config_value(lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR')
        output_dir = Path(data_dir) / f'domain_{domain_name}' / 'simulations' / experiment_id / 'WFLOW'
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dir = str(output_dir / 'states')
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        timestep = self._get_config_value(lambda: self.config.data.forcing_time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')

        toml_content = f'''# Wflow SBM configuration - generated by SYMFLUENCE
# Uses Wflow.jl v1.0+ CSDMS standard names

dir_output = "{str(output_dir)}"

[time]
starttime = {start_date.strftime('%Y-%m-%dT%H:%M:%S')}
endtime = {end_date.strftime('%Y-%m-%dT%H:%M:%S')}
timestepsecs = {timestep}

[logging]
loglevel = "info"

[input]
path_forcing = "{str(self.forcing_out_dir / 'forcing.nc')}"
path_static = "{str(self.settings_dir / staticmaps_file)}"

# Basin topology
basin__local_drain_direction = "wflow_ldd"
river_location__mask = "wflow_river"
subbasin_location__count = "wflow_subcatch"

[input.forcing]
atmosphere_water__precipitation_volume_flux = "precip"
land_surface_water__potential_evaporation_volume_flux = "pet"
atmosphere_air__temperature = "temp"

[input.static]
# Snow parameters
atmosphere_air__snowfall_temperature_threshold = "TT"
atmosphere_air__snowfall_temperature_interval = "TTI"
snowpack__melting_temperature_threshold = "TTM"
snowpack__degree_day_coefficient = "Cfmax"

# Soil parameters
soil_layer_water__brooks_corey_exponent = "c"
soil_surface_water__vertical_saturated_hydraulic_conductivity = "KsatVer"
soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter = "f"
compacted_soil_surface_water__infiltration_capacity = "InfiltCapPath"
soil_water__residual_volume_fraction = "thetaR"
soil_water__saturated_volume_fraction = "thetaS"
compacted_soil__area_fraction = "PathFrac"
soil__thickness = "SoilThickness"
subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio = "KsatHorFrac"
soil_surface_water__infiltration_reduction_parameter = "cf_soil"
soil_water_saturated_zone_bottom__max_leakage_volume_flux = "MaxLeakage"

# Vegetation parameters
vegetation_root__depth = "RootingDepth"

# River parameters
river__length = "wflow_riverlength"
river_water_flow__manning_n_parameter = "N_River"
river__slope = "RiverSlope"
river__width = "wflow_riverwidth"

# Land surface parameters
land_surface_water_flow__manning_n_parameter = "N"
land_surface__slope = "Slope"

[model]
soil_layer__thickness = [100, 300, 800]
type = "sbm"

[state]
path_input = "{state_dir}/instates.nc"
path_output = "{state_dir}/outstates.nc"

[output.csv]
path = "{str(output_dir / 'output.csv')}"

[[output.csv.column]]
header = "Q"
parameter = "river_water__volume_flow_rate"
reducer = "mean"
'''
        (self.settings_dir / config_file).write_text(toml_content)
        logger.info(f"Written TOML config to {self.settings_dir / config_file}")
