# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for WRFHYDRO.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class WRFHydroConfig(BaseModel):
    """WRF-Hydro (NCAR) coupled atmosphere-hydrology model configuration.

    WRF-Hydro is NCAR's community hydrological modeling system and forms
    the backbone of the US National Water Model. It couples the Noah-MP
    land surface model with terrain-following routing.

    Reference:
        Gochis, D.J., et al. (2020): The WRF-Hydro modeling system technical
        description, (Version 5.1.1). NCAR Technical Note.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WRFHYDRO_INSTALL_PATH')
    exe: str = Field(default='wrf_hydro.exe', alias='WRFHYDRO_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WRFHYDRO_PATH')
    namelist_file: str = Field(default='namelist.hrldas', alias='WRFHYDRO_NAMELIST_FILE')
    hydro_namelist: str = Field(default='hydro.namelist', alias='WRFHYDRO_HYDRO_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='WRFHYDRO_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WRFHYDRO')
    params_to_calibrate: str = Field(
        default='REFKDT,SLOPE,OVROUGHRTFAC,RETDEPRTFAC,LKSATFAC,BEXP,DKSAT,SMCMAX',
        alias='WRFHYDRO_PARAMS_TO_CALIBRATE'
    )
    lsm: str = Field(default='noahmp', alias='WRFHYDRO_LSM')
    routing_option: str = Field(default='gridded', alias='WRFHYDRO_ROUTING_OPTION')
    channel_routing: str = Field(default='diffusive_wave', alias='WRFHYDRO_CHANNEL_ROUTING')
    restart_frequency: str = Field(default='monthly', alias='WRFHYDRO_RESTART_FREQUENCY')
    timeout: int = Field(default=7200, alias='WRFHYDRO_TIMEOUT', ge=60, le=86400)
