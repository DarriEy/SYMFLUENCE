# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for PRMS.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class PRMSConfig(BaseModel):
    """PRMS (Precipitation-Runoff Modeling System) configuration.

    PRMS is a deterministic, distributed-parameter, physical-process
    watershed model developed by the USGS for simulating the effects
    of precipitation, climate, and land use on streamflow.

    Reference:
        Markstrom, S.L., et al. (2015): PRMS-IV, the Precipitation-Runoff
        Modeling System, Version 4. USGS Techniques and Methods 6-B7.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='PRMS_INSTALL_PATH')
    exe: str = Field(default='prms', alias='PRMS_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_PRMS_PATH')
    control_file: str = Field(default='control.dat', alias='PRMS_CONTROL_FILE')
    parameter_file: str = Field(default='params.dat', alias='PRMS_PARAMETER_FILE')
    data_file: str = Field(default='data.dat', alias='PRMS_DATA_FILE')
    spatial_mode: SpatialModeType = Field(default='semi_distributed', alias='PRMS_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PRMS')
    params_to_calibrate: str = Field(
        default='soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,hru_percent_imperv,carea_max,smidx_coef,slowcoef_lin,gwflow_coef,ssr2gw_rate',
        alias='PRMS_PARAMS_TO_CALIBRATE'
    )
    model_mode: str = Field(default='DAILY', alias='PRMS_MODEL_MODE')
    timeout: int = Field(default=3600, alias='PRMS_TIMEOUT', ge=60, le=86400)
    use_obs_solar: bool = Field(default=False, alias='PRMS_USE_OBS_SOLAR')
