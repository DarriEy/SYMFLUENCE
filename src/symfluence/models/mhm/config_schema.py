# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MHM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class MHMConfig(BaseModel):
    """mHM (mesoscale Hydrological Model) configuration.

    mHM is a spatially distributed hydrological model developed at the
    Helmholtz Centre for Environmental Research (UFZ). It uses multiscale
    parameter regionalization (MPR) for parameter transfer.

    Reference:
        Samaniego, L., et al. (2010): Multiscale parameter regionalization
        of a grid-based hydrologic model at the mesoscale. Water Resources
        Research, 46, W05523.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MHM_INSTALL_PATH')
    exe: str = Field(default='mhm', alias='MHM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MHM_PATH')
    namelist_file: str = Field(default='mhm.nml', alias='MHM_NAMELIST_FILE')
    routing_namelist: str = Field(default='mrm.nml', alias='MHM_ROUTING_NAMELIST')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MHM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MHM')
    params_to_calibrate: str = Field(
        default='canopyInterceptionFactor,snowTreshholdTemperature,degreeDayFactor_forest,degreeDayFactor_pervious,PTF_Ks_constant,interflowRecession_slope,rechargeCoefficient,GeoParam(1,:),infiltrationShapeFactor,rootFractionCoefficient_pervious,interflowStorageCapacityFactor,slowInterflowRecession_Ks,muskingumTravelTime_constant,orgMatterContent_forest',
        alias='MHM_PARAMS_TO_CALIBRATE'
    )
    timeout: int = Field(default=3600, alias='MHM_TIMEOUT', ge=60, le=86400)
    distributed_morph: bool = Field(default=False, alias='MHM_DISTRIBUTED')
    grid_res: float = Field(default=0.02, alias='MHM_GRID_RES', gt=0.0)
