# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for LISFLOOD.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class LisfloodConfig(BaseModel):
    """LISFLOOD distributed hydrological model configuration.

    LISFLOOD is a spatially distributed water resources model developed by
    the Joint Research Centre (JRC) of the European Commission. It simulates
    hydrological processes including snowmelt, soil moisture, groundwater,
    and channel routing using PCRaster.

    Reference:
        De Roo et al. (2000): LISFLOOD: a GIS-based distributed model for
        river basin scale water balance and flood simulation. Int. J.
        Geographical Information Science, 14(4), 443-455.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='LISFLOOD_INSTALL_PATH')
    exe: str = Field(default='lisflood', alias='LISFLOOD_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_LISFLOOD_PATH')
    settings_file: str = Field(default='settings.xml', alias='LISFLOOD_SETTINGS_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='LISFLOOD_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_LISFLOOD')
    pcraster_path: str = Field(default='default', alias='LISFLOOD_PCRASTER_PATH')
    num_threads: int = Field(default=1, alias='LISFLOOD_NUM_THREADS', ge=1, le=64)
    pet_method: str = Field(default='oudin', alias='LISFLOOD_PET_METHOD')
    forest_fraction: float = Field(default=0.3, alias='LISFLOOD_FOREST_FRACTION', ge=0.0, le=1.0)
    forcing_timestep: int = Field(default=86400, alias='LISFLOOD_FORCING_TIMESTEP', ge=3600, le=86400)
    params_to_calibrate: str = Field(
        default='ksat1,ksat2,ksat3,lambda1,lambda2,lambda3,thetas1,thetas2,thetas3,chanman,mannings,cropcoef,snow_melt_coef,temp_melt,temp_snow,GwPercValue',
        alias='LISFLOOD_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=7200, alias='LISFLOOD_TIMEOUT', ge=60, le=86400)
