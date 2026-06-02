# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Noah-MP Model Preprocessor.

Prepares forcing data and configuration files for the standalone
noah-owp-modular model (NOAA-OWP).  Forcing is converted from NetCDF
(basin-averaged) to the ASCII format expected by the executable, and
parameter tables (GENPARM.TBL, SOILPARM.TBL, MPTABLE.TBL) plus
a Fortran namelist (namelist.input) are written to the settings directory.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.registries import R
from symfluence.models.base.base_preprocessor import BaseModelPreProcessor

logger = logging.getLogger(__name__)


@R.preprocessors.add('NOAHMP')
class NoahMPPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Preprocessor for the standalone Noah-MP land surface model.

    Converts basin-averaged forcing (NetCDF) into the ASCII format
    expected by noah-owp-modular and generates the Fortran namelist
    plus parameter lookup tables.
    """

    MODEL_NAME = "NOAHMP"

    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.noahmp_forcing_dir = self.project_forcing_dir / 'NOAHMP_input'
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'

        self._forcing_start: Optional[datetime] = None
        self._forcing_end: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_preprocessing(self) -> bool:
        """Run the complete Noah-MP preprocessing workflow."""
        self.noahmp_forcing_dir = self.project_forcing_dir / 'NOAHMP_input'
        self.noahmp_settings_dir = self.project_dir / 'settings' / 'NOAHMP'
        return self.run_preprocessing_template()

    # ------------------------------------------------------------------
    # Template-method hooks
    # ------------------------------------------------------------------

    def _prepare_forcing(self) -> None:
        """Convert basin-averaged NetCDF forcing to Noah-MP ASCII format.

        noah-owp-modular expects a single ASCII file with a ``<FORCING>``
        header line followed by rows of
        ``YYYY MM DD HH MM  windspd airtemp spechum airpres SWRad LWRad pptrate RH``
        using a Fortran format of ``(I4.4, 4(1x,I2.2), 8(F17.10))``.
        """
        try:
            self.noahmp_forcing_dir.mkdir(parents=True, exist_ok=True)

            # Locate forcing NetCDF
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                logger.error(f"No forcing NetCDF found in {self.forcing_basin_path}")
                return

            ds = xr.open_mfdataset(forcing_files, combine='by_coords')

            # Variable mapping (SYMFLUENCE standard -> Noah-MP names)
            var_map = {
                'windspd': ['windspd', 'WIND', 'wind_speed', 'u10'],
                'airtemp': ['airtemp', 'TAIR', 'air_temperature', 't2m', 'TMP'],
                'spechum': ['spechum', 'SPFH', 'specific_humidity', 'q2'],
                'airpres': ['airpres', 'PRES', 'air_pressure', 'sp', 'PSFC'],
                'SWRadAtm': ['SWRadAtm', 'DSWRF', 'SWDOWN', 'ssrd'],
                'LWRadAtm': ['LWRadAtm', 'DLWRF', 'LWDOWN', 'strd'],
                'pptrate': ['pptrate', 'APCP', 'RAINRATE', 'tp', 'PRATE'],
            }

            def _find_var(ds, aliases):
                for alias in aliases:
                    if alias in ds:
                        return ds[alias].values.flatten()
                return None

            windspd = _find_var(ds, var_map['windspd'])
            airtemp = _find_var(ds, var_map['airtemp'])
            spechum = _find_var(ds, var_map['spechum'])
            airpres = _find_var(ds, var_map['airpres'])
            swrad = _find_var(ds, var_map['SWRadAtm'])
            lwrad = _find_var(ds, var_map['LWRadAtm'])
            pptrate = _find_var(ds, var_map['pptrate'])

            times = pd.to_datetime(ds['time'].values)
            ds.close()

            for name, arr in [('windspd', windspd), ('airtemp', airtemp),
                              ('spechum', spechum), ('airpres', airpres),
                              ('SWRadAtm', swrad), ('LWRadAtm', lwrad),
                              ('pptrate', pptrate)]:
                if arr is None:
                    logger.error(f"Required forcing variable '{name}' not found")
                    return

            # Convert specific humidity -> relative humidity
            rh = self._spechum_to_rh(spechum, airtemp, airpres)

            # Store time bounds
            self._forcing_start = times[0].to_pydatetime()
            self._forcing_end = times[-1].to_pydatetime()

            # Write ASCII forcing file
            forcing_path = self.noahmp_forcing_dir / 'forcing.txt'
            with open(forcing_path, 'w') as f:
                f.write('<FORCING>\n')
                for i, t in enumerate(times):
                    line = (
                        f"{t.year:04d} {t.month:02d} {t.day:02d} "
                        f"{t.hour:02d} {t.minute:02d}"
                        f"{windspd[i]:17.10f}{airtemp[i]:17.10f}"
                        f"{spechum[i]:17.10f}{airpres[i]:17.10f}"
                        f"{swrad[i]:17.10f}{lwrad[i]:17.10f}"
                        f"{pptrate[i]:17.10f}{rh[i]:17.10f}"
                    )
                    f.write(line + '\n')

            logger.info(
                f"Wrote Noah-MP forcing: {len(times)} timesteps "
                f"({self._forcing_start} to {self._forcing_end})"
            )
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error preparing Noah-MP forcing: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _create_model_configs(self) -> None:
        """Copy parameter tables and write namelist.input."""
        try:
            self.noahmp_settings_dir.mkdir(parents=True, exist_ok=True)

            # Copy parameter lookup tables from install directory
            install_path = self._get_config_value(
                lambda: self.config.model.noahmp.install_path,
                default='default',
                dict_key='NOAHMP_INSTALL_PATH',
            )
            if install_path == 'default':
                data_dir = Path(self._get_config_value(
                    lambda: self.config.system.data_dir,
                    dict_key='SYMFLUENCE_DATA_DIR',
                ))
                install_dir = data_dir / 'installs' / 'noah-owp-modular'
            else:
                install_dir = Path(install_path)

            params_dir = install_dir / 'parameters'
            for tbl in ['GENPARM.TBL', 'SOILPARM.TBL', 'MPTABLE.TBL']:
                src = params_dir / tbl
                if src.exists():
                    shutil.copy2(src, self.noahmp_settings_dir / tbl)
                    logger.debug(f"Copied {tbl}")
                else:
                    logger.warning(f"Parameter table not found: {src}")

            # Write namelist.input
            self._write_namelist()

            logger.info("Noah-MP configuration files created")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error creating Noah-MP configs: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # Namelist generation
    # ------------------------------------------------------------------

    def _write_namelist(self) -> None:
        """Write the Fortran namelist (namelist.input) for noah-owp-modular."""
        # Determine time settings
        if self._forcing_start and self._forcing_end:
            start_str = self._forcing_start.strftime('%Y%m%d%H%M')
            end_str = self._forcing_end.strftime('%Y%m%d%H%M')
        else:
            start_str = self._get_config_value(
                lambda: self.config.domain.time_start,
                default='200001010000',
                dict_key='EXPERIMENT_TIME_START',
            ).replace('-', '').replace(' ', '').replace(':', '')
            end_str = self._get_config_value(
                lambda: self.config.domain.time_end,
                default='200112310000',
                dict_key='EXPERIMENT_TIME_END',
            ).replace('-', '').replace(' ', '').replace(':', '')

        # Dynamic vegetation option from config (default 1)
        dynamic_veg = int(self._get_config_value(
            lambda: self.config.model.noahmp.dynamic_veg_option,
            default=1,
            dict_key='NOAHMP_DYNAMIC_VEG_OPTION',
        ))

        forcing_path = self.noahmp_forcing_dir / 'forcing.txt'
        output_path = self.noahmp_settings_dir / 'output.nc'

        namelist = f"""\
&timing
  dt                 = 3600
  startdate          = "{start_str}"
  enddate            = "{end_str}"
  forcing_filename   = "{forcing_path}"
  output_filename    = "{output_path}"
/

&parameters
  parameter_dir      = "{self.noahmp_settings_dir}/"
  soil_class_name    = "STAS"
  veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
  lat                =  {self._get_lat():.6f}
  lon                =  {self._get_lon():.6f}
  terrain_slope      =  0.0
  azimuth            =  0.0
/

&forcing
  ZREF               =  10.0
/

&model_options
  dynamic_veg_option         = {dynamic_veg}
  runoff_option              = 8
  drainage_option            = 8
  snowsoil_temp_time_option  = 1
/

&structure
  isltyp             = 4
  nsoil              = 4
  nsnow              = 3
  nveg               = 20
  vegtyp             = 1
  croptype           = 0
  dzsnso             = 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 1.0
/
"""
        nl_path = self.noahmp_settings_dir / 'namelist.input'
        nl_path.write_text(namelist)
        logger.info(f"Wrote namelist.input to {nl_path}")

    # ------------------------------------------------------------------
    # Helper: lat/lon from domain configuration
    # ------------------------------------------------------------------

    def _get_lat(self) -> float:
        """Get latitude from configuration or default."""
        return float(self._get_config_value(
            lambda: self.config.domain.latitude,
            default=51.0,
            dict_key='DOMAIN_LATITUDE',
        ))

    def _get_lon(self) -> float:
        """Get longitude from configuration or default."""
        return float(self._get_config_value(
            lambda: self.config.domain.longitude,
            default=-116.0,
            dict_key='DOMAIN_LONGITUDE',
        ))

    # ------------------------------------------------------------------
    # Static utility: specific humidity -> relative humidity
    # ------------------------------------------------------------------

    @staticmethod
    def _spechum_to_rh(
        q: np.ndarray,
        t_k: np.ndarray,
        p_pa: np.ndarray,
    ) -> np.ndarray:
        """Convert specific humidity to relative humidity.

        Uses the Tetens approximation for saturation vapour pressure.

        Args:
            q: Specific humidity (kg/kg).
            t_k: Air temperature (K).
            p_pa: Air pressure (Pa).

        Returns:
            Relative humidity (%), clipped to [0, 100].
        """
        tc = t_k - 273.15
        es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
        e = q * p_pa / (0.622 + 0.378 * q)
        rh = 100.0 * e / es
        return np.clip(rh, 0.0, 100.0)
