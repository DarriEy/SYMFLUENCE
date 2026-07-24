# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for SWAT.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class SWATConfig(BaseModel):
    """SWAT (Soil and Water Assessment Tool) model configuration.

    SWAT is a river basin scale model developed by USDA-ARS to predict
    the impact of land management on water, sediment, and agricultural
    chemical yields.

    Reference:
        Arnold, J.G., et al. (1998): Large area hydrologic modeling and
        assessment Part I: Model development. JAWRA, 34(1), 73-89.
    """
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='SWAT_INSTALL_PATH')
    exe: str = Field(default='swat_rel.exe', alias='SWAT_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_SWAT_PATH')
    txtinout_dir: str = Field(default='TxtInOut', alias='SWAT_TXTINOUT_DIR')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='SWAT_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_SWAT')
    params_to_calibrate: str = Field(
        default='CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP',
        alias='SWAT_PARAMS_TO_CALIBRATE'
    )
    warmup_years: int = Field(default=2, alias='SWAT_WARMUP_YEARS', ge=0, le=10)
    timeout: int = Field(default=3600, alias='SWAT_TIMEOUT', ge=60, le=86400)
    plaps: float = Field(default=0.0, alias='SWAT_PLAPS')
    tlaps: float = Field(default=0.0, alias='SWAT_TLAPS')
