# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Noah-MP (noah-owp-modular) Preprocessor."""
from pathlib import Path
from typing import Tuple

import numpy as np

from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_preprocessor('NOAHMP')
class NoahMPPreProcessor(BaseModelPreProcessor):
    MODEL_NAME = "NOAHMP"

    def run_preprocessing(self) -> bool:
        self.noahmp_forcing_dir = self.project_forcing_dir / 'NOAHMP_input'
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        import pandas as pd
        import xarray as xr
        self.noahmp_forcing_dir.mkdir(parents=True, exist_ok=True)
        time_window = self.get_simulation_time_window()
        sim_start, sim_end = time_window if time_window else (None, None)
        forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing in {self.forcing_basin_path}")
        all_data = []
        for f in forcing_files:
            ds = xr.open_dataset(f)
            if sim_start and sim_end:
                ds = ds.sel(time=slice(sim_start, sim_end))
            if ds.sizes.get('time', 0) == 0:
                ds.close(); continue
            df = pd.DataFrame({
                'time': ds['time'].values,
                'windspd': ds['windspd'].values.flatten(),
                'airtemp': ds['airtemp'].values.flatten(),
                'spechum': ds['spechum'].values.flatten(),
                'airpres': ds['airpres'].values.flatten(),
                'SWRadAtm': ds['SWRadAtm'].values.flatten(),
                'LWRadAtm': ds['LWRadAtm'].values.flatten(),
                'pptrate': ds['pptrate'].values.flatten(),
            })
            all_data.append(df); ds.close()
        if not all_data:
            raise ValueError("No forcing data in simulation window")
        forcing = pd.concat(all_data, ignore_index=True).sort_values('time').reset_index(drop=True)
        forcing['rh'] = self._spechum_to_rh(forcing['spechum'].values, forcing['airtemp'].values, forcing['airpres'].values)
        forcing_file = self.noahmp_forcing_dir / 'forcing.txt'
        with open(forcing_file, 'w') as fout:
            fout.write("<FORCING>\n")
            for _, row in forcing.iterrows():
                dt = pd.Timestamp(row['time'])
                fout.write(f"{dt.year:04d} {dt.month:02d} {dt.day:02d} {dt.hour:02d} {dt.minute:02d}"
                           f"{row['windspd']:17.10f}{0.0:17.10f}{row['airtemp']:17.10f}"
                           f"{row['rh']:17.10f}{row['airpres']/100.0:17.10f}"
                           f"{row['SWRadAtm']:17.10f}{row['LWRadAtm']:17.10f}{row['pptrate']:17.10f}\n")
        self.logger.info(f"Wrote Noah-MP forcing: {len(forcing)} timesteps to {forcing_file}")
        self._forcing_start = pd.Timestamp(forcing['time'].iloc[0])
        self._forcing_end = pd.Timestamp(forcing['time'].iloc[-1])

    def _create_model_configs(self) -> None:
        self.noahmp_settings_dir.mkdir(parents=True, exist_ok=True)
        self._copy_parameter_tables()
        self._write_namelist()

    def _copy_parameter_tables(self) -> None:
        param_dir = self.noahmp_settings_dir / 'parameters'
        param_dir.mkdir(parents=True, exist_ok=True)
        install_path = self._get_config_value(lambda: self.config.model.noahmp.install_path, default='default')
        if install_path == 'default' or install_path is None:
            data_dir = self._get_config_value(lambda: self.config.system.data_dir, default='.', dict_key='SYMFLUENCE_DATA_DIR')
            noahmp_dir = Path(data_dir) / 'installs' / 'noah-owp-modular'
        else:
            noahmp_dir = Path(install_path)
        source = noahmp_dir / 'parameters'
        if not source.exists():
            source = noahmp_dir / 'run' / 'parameters'
        from symfluence.core.file_utils import copy_file
        for tbl in ['GENPARM.TBL', 'SOILPARM.TBL', 'MPTABLE.TBL']:
            src = source / tbl
            if src.exists():
                copy_file(src, param_dir / tbl)

    def _write_namelist(self) -> None:
        forcing_file = self.noahmp_forcing_dir / 'forcing.txt'
        exp_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1')
        output_file = self.project_dir / 'simulations' / exp_id / 'NOAHMP' / 'output.nc'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        param_dir = self.noahmp_settings_dir / 'parameters'
        dt = self._get_config_value(lambda: self.config.model.noahmp.timestep, default=3600)
        nsoil = self._get_config_value(lambda: self.config.model.noahmp.nsoil, default=4)
        nsnow = self._get_config_value(lambda: self.config.model.noahmp.nsnow, default=3)
        start_str = self._forcing_start.strftime("%Y%m%d%H%M")
        end_str = self._forcing_end.strftime("%Y%m%d%H%M")
        lat, lon = self._get_domain_coords()
        dv = self._get_config_value(lambda: self.config.model.noahmp.dynamic_veg_option, default=1)
        st = self._get_config_value(lambda: self.config.model.noahmp.canopy_stomatal_option, default=1)
        sd = self._get_config_value(lambda: self.config.model.noahmp.sfc_drag_option, default=1)
        sc = self._get_config_value(lambda: self.config.model.noahmp.supercooled_water_option, default=1)
        fz = self._get_config_value(lambda: self.config.model.noahmp.frozen_soil_option, default=1)
        rt = self._get_config_value(lambda: self.config.model.noahmp.radiative_transfer_option, default=3)
        sa = self._get_config_value(lambda: self.config.model.noahmp.snow_albedo_option, default=2)
        pp = self._get_config_value(lambda: self.config.model.noahmp.precip_phase_option, default=1)
        nml = (self.noahmp_settings_dir / 'namelist.input')
        nml.write_text(f"""\
&timing
 dt                 = {dt}.0
 startdate          = "{start_str}"
 enddate            = "{end_str}"
 forcing_filename   = "{forcing_file}"
 output_filename    = "{output_file}"
/

&parameters
 parameter_dir      = "{param_dir}/"
 general_table      = "GENPARM.TBL"
 soil_table         = "SOILPARM.TBL"
 noahowp_table      = "MPTABLE.TBL"
 soil_class_name    = "STAS"
 veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
 lat                = {lat:.4f}
 lon                = {lon:.4f}
 terrain_slope      = 0.01
 azimuth            = 0.0
/

&forcing
 ZREF               = 2.0
 rain_snow_thresh   = 2.0
/

&model_options
 precip_phase_option               = {pp}
 snow_albedo_option                = {sa}
 dynamic_veg_option                = {dv}
 runoff_option                     = 8
 drainage_option                   = 8
 frozen_soil_option                = {fz}
 dynamic_vic_option                = 1
 radiative_transfer_option         = {rt}
 sfc_drag_coeff_option             = {sd}
 canopy_stom_resist_option         = {st}
 crop_model_option                 = 0
 snowsoil_temp_time_option         = 1
 soil_temp_boundary_option         = 2
 supercooled_water_option          = {sc}
 stomatal_resistance_option        = 1
 evap_srfc_resistance_option       = 1
 subsurface_option                 = 1
/

&structure
 isltyp             = 4
 nsoil              = {nsoil}
 nsnow              = {nsnow}
 nveg               = 20
 vegtyp             = 1
 croptype           = 0
 sfctyp             = 1
 soilcolor          = 4
/

&initial_values
 dzsnso    =  0.0,  0.0,  0.0,  0.1,  0.3,  0.6,  1.0
 sice      =  0.0,  0.0,  0.0,  0.0
 sh2o      =  0.3,  0.3,  0.3,  0.3
 zwt       =  -2.0
/
""")
        self.logger.info(f"Wrote namelist.input: {nml}")

    def _get_domain_coords(self) -> Tuple[float, float]:
        coords_str = self._get_config_value(lambda: self.config.domain.pour_point_coords, default=None, dict_key='POUR_POINT_COORDS')
        if coords_str:
            parts = str(coords_str).replace(',', '/').split('/')
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 51.17, -115.57

    @staticmethod
    def _spechum_to_rh(q: np.ndarray, t_k: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
        t_c = t_k - 273.15
        es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
        e = q * p_pa / (0.622 + 0.378 * q)
        return np.clip(100.0 * e / es, 0.0, 100.0)
