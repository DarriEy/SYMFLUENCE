# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for GR.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class GRConfig(BaseModel):
    """GR (GR4J/GR5J) hydrological model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='GR_INSTALL_PATH')
    exe: str = Field(default='GR.r', alias='GR_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='GR_SPATIAL_MODE')
    routing_integration: str = Field(default='none', alias='GR_ROUTING_INTEGRATION')
    settings_path: str = Field(default='default', alias='SETTINGS_GR_PATH')
    control: str = Field(default='default', alias='SETTINGS_GR_CONTROL')
    params_to_calibrate: str = Field(
        default='X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
        alias='GR_PARAMS_TO_CALIBRATE'
    )
    # Fallback behavior control - default to False to prevent silent data corruption
    allow_dummy_observations: bool = Field(
        default=False,
        alias='GR_ALLOW_DUMMY_OBSERVATIONS',
        description='If True, use zero-filled dummy observations when no streamflow data found'
    )
    allow_default_area: bool = Field(
        default=False,
        alias='GR_ALLOW_DEFAULT_AREA',
        description='If True, use 1.0 km² default area when basin shapefile not found'
    )
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR_PARAM_BOUNDS')
    gr4j_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='GR4J_PARAM_BOUNDS')
    initial_params: str = Field(default='default', alias='GR_INITIAL_PARAMS')
    default_params: Optional[List[float]] = Field(default=None, alias='GR_DEFAULT_PARAMS')
    model_type: Optional[str] = Field(
        default=None, alias='GR_MODEL_TYPE',
        description="GR model variant (e.g. 'GR4J', 'GR5J'); default GR4J"
    )


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
GRConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_GR_PATH': 'settings_path',
    'SETTINGS_GR_CONTROL': 'control',
}
