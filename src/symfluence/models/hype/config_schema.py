# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for HYPE.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class HYPEConfig(BaseModel):
    """HYPE hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='HYPE_INSTALL_PATH')
    exe: str = Field(default='hype', alias='HYPE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_HYPE_PATH')
    info_file: str = Field(default='info.txt', alias='SETTINGS_HYPE_INFO')
    params_to_calibrate: str = Field(
        default='ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp,wcwp,wcfc,wcep,srrcs',
        alias='HYPE_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='HYPE_SPINUP_DAYS')
    # Process options written to info.txt (None = config_manager defaults apply)
    infiltration_model: Optional[int] = Field(
        default=None, alias='HYPE_INFILTRATION_MODEL',
        description='HYPE infiltration model option (info.txt modeloption)'
    )
    pet_model: Optional[int] = Field(
        default=None, alias='HYPE_PET_MODEL',
        description='HYPE potential evapotranspiration model option'
    )
    frozen_soil_model: Optional[int] = Field(
        default=None, alias='HYPE_FROZEN_SOIL_MODEL',
        description='HYPE frozen soil model option'
    )
    snow_evaporation: Optional[int] = Field(
        default=None, alias='HYPE_SNOW_EVAPORATION',
        description='HYPE snow evaporation option'
    )
    deep_ground: Optional[int] = Field(
        default=None, alias='HYPE_DEEP_GROUND',
        description='HYPE deep groundwater option'
    )
    surface_runoff: Optional[int] = Field(
        default=None, alias='HYPE_SURFACE_RUNOFF',
        description='HYPE surface runoff option'
    )
    soil_init_wet: Optional[bool] = Field(
        default=None, alias='HYPE_SOIL_INIT_WET',
        description='Initialize soil moisture at field capacity'
    )
    soil_layer_depths: Optional[List[float]] = Field(
        default=None, alias='HYPE_SOIL_LAYER_DEPTHS',
        description='Soil layer depths (m) for GeoData generation'
    )
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='HYPE_PARAM_BOUNDS')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
HYPEConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_HYPE_PATH': 'settings_path',
    'SETTINGS_HYPE_INFO': 'info_file',
}
