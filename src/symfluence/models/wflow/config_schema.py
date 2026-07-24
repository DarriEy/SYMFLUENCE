# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for WFLOW.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class WflowConfig(BaseModel):
    """Wflow (wflow_sbm) distributed hydrological model configuration.

    Wflow is a distributed hydrological model developed by Deltares,
    implemented in Julia. The wflow_sbm concept uses topography-driven
    subsurface flow with a kinematic wave for overland and river routing.

    Reference:
        van Verseveld et al. (2024): Wflow_sbm v0.7.3, a spatially
        distributed hydrological model. Geosci. Model Dev., 17, 3021-3043.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='WFLOW_INSTALL_PATH')
    exe: str = Field(default='wflow_cli', alias='WFLOW_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_WFLOW_PATH')
    config_file: str = Field(default='wflow_sbm.toml', alias='WFLOW_CONFIG_FILE')
    staticmaps_file: str = Field(default='wflow_staticmaps.nc', alias='WFLOW_STATICMAPS_FILE')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='WFLOW_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_WFLOW')
    output_file: str = Field(default='output.nc', alias='WFLOW_OUTPUT_FILE')
    pet_method: str = Field(default='oudin', alias='WFLOW_PET_METHOD')
    params_to_calibrate: str = Field(
        default='KsatVer,f,SoilThickness,InfiltCapPath,RootingDepth,KsatHorFrac,n_river,PathFrac,thetaS,thetaR,Cfmax,TT,TTI,TTM,WHC',
        alias='WFLOW_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=7200, alias='WFLOW_TIMEOUT', ge=60, le=86400)
