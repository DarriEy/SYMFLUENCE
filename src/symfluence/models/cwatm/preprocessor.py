# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM Model Preprocessor.

CWatM (Community Water Model) is a pure-Python global hydrological model
from IIASA. Unlike PCR-GLOBWB, it has no PCRaster dependency — it uses
NumPy reimplementations of PCRaster operations. It reads its own global
parameter dataset and can delineate catchments from a coordinate pair.

The preprocessor generates:
- Daily forcing files (precipitation m/day, temperature °C, reference ET m/day)
- A settings.ini with CWatM's custom $(SECTION:OPTION) interpolation
- CWatM reads its own static parameter files from the CWatM-Earth dataset
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_preprocessor("CWATM")
class CWatMPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """Prepares inputs for a CWatM model run.

    CWatM can delineate its own catchment from a coordinate pair +
    global LDD, so the preprocessor primarily generates forcing data
    and the settings INI file. Static parameters (soil, land cover,
    routing) are read from CWatM's bundled dataset.
    """

    MODEL_NAME = "CWATM"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.settings_dir = self.setup_dir
        self.forcing_out_dir = self.setup_dir / "forcing"
        self.output_dir = self.setup_dir / "output"

        self.resolution = self._get_config_value(
            lambda: self.config.model.cwatm.resolution,
            default='30min', dict_key='CWATM_RESOLUTION',
        )

    def run_preprocessing(self) -> bool:
        try:
            self.logger.info("Starting CWatM preprocessing...")
            self._create_directory_structure()
            self._generate_forcing()
            self._generate_settings_ini()
            self.logger.info("CWatM preprocessing complete.")
            return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as e:
            self.logger.error(f"CWatM preprocessing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        for d in [self.settings_dir, self.forcing_out_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)
        return (
            pd.to_datetime(start_str).to_pydatetime(),
            pd.to_datetime(end_str).to_pydatetime(),
        )

    def _get_catchment_properties(self) -> Dict:
        props = {'lat': 51.17, 'lon': -115.57, 'area_m2': 2.21e9}
        try:
            import geopandas as gpd
            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)
                centroid = gdf.geometry.centroid.iloc[0]
                props['lat'] = centroid.y
                props['lon'] = centroid.x
                utm_zone = int((centroid.x + 180) / 6) + 1
                hemisphere = 'north' if centroid.y >= 0 else 'south'
                epsg = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
                props['area_m2'] = gdf.to_crs(f"EPSG:{epsg}").geometry.area.sum()
        except (OSError, ValueError, IndexError, KeyError, ImportError) as e:
            self.logger.warning(f"Could not read catchment properties: {e}")
        return props

    def _generate_forcing(self) -> None:
        """Convert SYMFLUENCE forcing to CWatM format.

        CWatM expects daily NetCDF forcing:
        - precipitation in m/day (or kg/m2/s with conversion factor)
        - temperature in °C (or K)
        - reference ET in m/day
        """
        self.logger.info("Generating CWatM forcing data...")
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

        time_vals = pd.DatetimeIndex(ds_forcing.time.values)
        if len(time_vals) > 1:
            dt_hours = (time_vals[1] - time_vals[0]).total_seconds() / 3600.0
        else:
            dt_hours = 24.0
        is_subdaily = dt_hours < 24.0
        if is_subdaily:
            self.logger.info(
                f"Sub-daily forcing ({dt_hours:.0f}h) — resampling to daily"
            )

        # Extract precipitation
        precip_raw = self._extract_forcing_var(
            ds_forcing,
            ['pptrate', 'precipitation_flux', 'mtpr', 'tp', 'precipitation', 'PREC', 'precip'],
            props['lat'], props['lon'],
        )
        precip_raw = np.maximum(precip_raw, 0.0)
        if precip_raw.max() < 0.1:
            precip_m_per_step = precip_raw * (dt_hours * 3600.0) / 1000.0
        elif precip_raw.max() < 200:
            precip_m_per_step = precip_raw / 1000.0
        else:
            precip_m_per_step = precip_raw

        # Extract temperature
        temp_raw = self._extract_forcing_var(
            ds_forcing,
            ['airtemp', 't2m', 'temperature', 'TEMP', 'air_temperature', 'tas'],
            props['lat'], props['lon'],
        )
        if temp_raw.mean() > 100:
            temp_raw = temp_raw - 273.15

        ds_forcing.close()

        precip_series = pd.Series(precip_m_per_step, index=time_vals)
        temp_series = pd.Series(temp_raw, index=time_vals)

        if is_subdaily:
            precip_daily = precip_series.resample('D').sum()
            temp_daily = temp_series.resample('D').mean()
        else:
            precip_daily = precip_series
            temp_daily = temp_series

        daily_times = precip_daily.index
        precip = precip_daily.values
        temp = temp_daily.values

        pet = self._estimate_pet(temp, daily_times, props['lat'])

        self.logger.info(
            f"Forcing: {len(daily_times)} days, "
            f"P={precip.mean()*1000:.1f} mm/day, "
            f"T={temp.mean():.1f}°C, "
            f"PET={pet.mean()*1000:.2f} mm/day"
        )

        # Write forcing as single-cell NetCDFs (CWatM reads nearest grid cell)
        lat, lon = props['lat'], props['lon']
        coords = {'time': daily_times, 'lat': [lat], 'lon': [lon]}

        for varname, data, units in [
            ('precipitation', precip, 'm/day'),
            ('tavg', temp, 'C'),
            ('EWRef', pet, 'm/day'),
            ('ETRef', pet, 'm/day'),
        ]:
            da = xr.DataArray(
                data.reshape(len(daily_times), 1, 1),
                dims=['time', 'lat', 'lon'],
                coords=coords,
                attrs={'units': units},
                name=varname,
            )
            out_path = self.forcing_out_dir / f'{varname}.nc'
            da.to_dataset().to_netcdf(out_path)

        self.logger.info(f"Written forcing to {self.forcing_out_dir}")

    def _extract_forcing_var(self, ds, var_names, lat, lon):
        for var in var_names:
            if var in ds.data_vars:
                data = ds[var]
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if spatial_dims:
                    try:
                        data = data.sel(
                            **{d: lat if 'lat' in d else lon for d in spatial_dims},
                            method='nearest',
                        )
                    except (KeyError, ValueError, TypeError, IndexError):
                        data = data.isel(**{d: 0 for d in spatial_dims})
                return data.values.flatten()
        raise ValueError(
            f"None of {var_names} found in forcing. Available: {list(ds.data_vars)}"
        )

    def _estimate_pet(self, temp_c, times, lat_deg):
        """Estimate PET in m/day using Hamon method."""
        from symfluence.models.mixins.pet_calculator import PETCalculatorMixin

        pet_method = self._get_config_value(
            lambda: self.config.model.cwatm.pet_method,
            default='hamon', dict_key='CWATM_PET_METHOD',
        )
        doy = np.array([t.timetuple().tm_yday for t in times])

        if pet_method == 'oudin':
            pet_mm_day = PETCalculatorMixin.oudin_pet_numpy(temp_c, doy, lat_deg)
            return pet_mm_day / 1000.0

        # Hamon
        lat_rad = math.radians(lat_deg)
        decl = 0.4093 * np.sin(2 * np.pi / 365 * doy - 1.405)
        cos_omega = np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1)
        day_length = 24 / np.pi * np.arccos(cos_omega)
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        pet_mm_day = np.maximum(
            0.1651 * (day_length / 12.0) * es * 216.7 / (temp_c + 273.3), 0.0
        )
        return pet_mm_day / 1000.0

    def _get_forcing_path(self) -> Path:
        forcing_path = self._get_config_value(
            lambda: self.config.data.forcing_path,
            default=None, dict_key='FORCING_PATH',
        )
        if forcing_path and forcing_path != 'default':
            return Path(forcing_path)
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='Bow_at_Banff', dict_key='DOMAIN_NAME',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )
        return Path(data_dir) / f'domain_{domain_name}' / 'forcing' / 'basin_averaged_data'

    def _generate_settings_ini(self) -> None:
        """Generate the CWatM settings.ini file.

        Uses CWatM's coordinate-based MaskMap to let the model delineate
        the catchment from its own LDD. Static parameters are read from
        CWatM's bundled dataset (CWatM-Earth-30min or local).
        """
        self.logger.info("Generating CWatM settings.ini...")
        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()

        config_file = self._get_config_value(
            lambda: self.config.model.cwatm.config_file,
            default='settings.ini',
        )
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='', dict_key='DOMAIN_NAME',
        )
        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1', dict_key='EXPERIMENT_ID',
        )
        data_dir = self._get_config_value(
            lambda: self.config.system.data_dir,
            default='.', dict_key='SYMFLUENCE_DATA_DIR',
        )

        sim_output_dir = (
            Path(data_dir) / f'domain_{domain_name}'
            / 'simulations' / experiment_id / 'CWATM'
        )
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        # CWatM install dir (contains cwatm_input/ with global parameter data)
        install_path = self._get_config_value(
            lambda: self.config.model.cwatm.install_path,
            default='default',
        )
        if install_path == 'default' or install_path is None:
            cwatm_dir = Path(data_dir) / 'installs' / 'cwatm'
        else:
            cwatm_dir = Path(install_path)

        # Spinup
        spinup_years = self._get_config_value(
            lambda: self.config.model.cwatm.spinup_years,
            default=0, dict_key='CWATM_SPINUP_YEARS',
        )

        # Determine paths for CWatM's global parameter dataset
        # CWatM-Earth-30min is typically at cwatm_dir/cwatm_input or a sibling
        cwatm_input = cwatm_dir / 'cwatm_input'
        if not cwatm_input.exists():
            cwatm_input = Path(data_dir) / 'cwatm_input'

        # Use CWatM-Earth template if available — it has all required keys
        template_path = cwatm_input / 'settings_CWatM_template_30min.ini'
        if template_path.exists():
            ini_content = self._patch_template(
                template_path.read_text(), props, start_date, end_date,
                cwatm_input, cwatm_dir, sim_output_dir, spinup_years, domain_name,
            )
        else:
            self.logger.warning("CWatM-Earth template not found — using minimal INI")
            ini_content = ""

        ini_path = self.settings_dir / config_file
        ini_path.write_text(ini_content)
        self.logger.info(f"Written settings to {ini_path}")

    def _patch_template(
        self, content: str, props: Dict, start_date, end_date,
        cwatm_input: Path, cwatm_dir: Path, sim_output_dir: Path,
        spinup_years: int, domain_name: str,
    ) -> str:
        """Patch the CWatM-Earth template with domain-specific values."""
        import re

        def _sub(key: str, val: str) -> None:
            nonlocal content
            content = re.sub(rf'(?m)^{re.escape(key)}\s*=.*$', f'{key} = {val}', content)

        _sub('PathRoot', str(cwatm_input))
        _sub('PathOut', str(sim_output_dir))
        _sub('metaNetcdfFile', f'{cwatm_dir}/cwatm/metaNetcdf.xml')
        _sub('MaskMap', f"{props['lon']} {props['lat']}")
        _sub('Gauges', f"{props['lon']} {props['lat']}")
        _sub('StepStart', start_date.strftime('%d/%m/%Y'))
        _sub('StepEnd', end_date.strftime('%d/%m/%Y'))
        _sub('SpinUp', 'None' if not spinup_years else str(spinup_years * 365))
        _sub('PrecipitationMaps', f'{self.forcing_out_dir}/precipitation*')
        _sub('TavgMaps', f'{self.forcing_out_dir}/tavg*')
        _sub('E0Maps', f'{self.forcing_out_dir}/EWRef*')
        _sub('ETMaps', f'{self.forcing_out_dir}/ETRef*')
        _sub('load_initial', 'False')
        _sub('save_initial', 'False')
        _sub('OUT_Dir', str(sim_output_dir))
        _sub('title', f'CWatM output - {domain_name} (SYMFLUENCE)')

        return content
