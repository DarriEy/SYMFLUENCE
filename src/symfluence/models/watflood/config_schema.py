# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for WATFLOOD.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class WATFLOODConfig(BaseModel):
    """WATFLOOD (Kouwen) distributed flood forecasting model configuration.

    WATFLOOD is a physically-based, distributed hydrological model using
    Grouped Response Units (GRUs) on a regular grid with internal channel
    routing. It is optimized for flood forecasting with simplified energy
    balance requiring only precipitation and temperature forcing.

    Reference:
        Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
        & Flood Forecasting System. University of Waterloo.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WATFLOOD_INSTALL_PATH')
    exe: str = Field(default='watflood', alias='WATFLOOD_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WATFLOOD_PATH')
    shed_file: str = Field(default='bow_shd.r2c', alias='WATFLOOD_SHED_FILE')
    par_file: str = Field(default='bow.par', alias='WATFLOOD_PAR_FILE')
    event_file: str = Field(default='event.evt', alias='WATFLOOD_EVENT_FILE')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='WATFLOOD_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WATFLOOD')
    params_to_calibrate: str = Field(
        default='FLZCOEF,PWR,R2N,AK,AKF,REESSION,RETN,AK2,AK2FS,R3,DS,FPET,FTALL,FM,BASE,SUBLIM_FACTOR',
        alias='WATFLOOD_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='WATFLOOD_TIMEOUT', ge=60, le=86400)


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
WATFLOODConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_WATFLOOD_PATH': 'settings_path',
}
