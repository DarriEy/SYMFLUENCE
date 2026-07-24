# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for TROUTE.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class TRouteConfig(BaseModel):
    """T-Route (NOAA OWP) channel routing configuration.

    T-Route is NOAA's Office of Water Prediction channel routing model
    supporting Muskingum-Cunge and diffusive wave routing methods for
    large-scale river network simulations.
    """
    model_config = FROZEN_CONFIG

    # Installation and paths
    install_path: str = Field(default='default', alias='TROUTE_INSTALL_PATH')
    pkg_path: str = Field(
        default='troute/network/__init__.py',
        alias='TROUTE_PKG_PATH',
    )
    settings_path: str = Field(default='default', alias='SETTINGS_TROUTE_PATH')

    # Topology and config files
    topology_file: str = Field(
        default='troute_topology.nc',
        alias='SETTINGS_TROUTE_TOPOLOGY',
    )
    config_file: str = Field(
        default='troute_config.yml',
        alias='SETTINGS_TROUTE_CONFIG_FILE',
    )

    # Routing configuration
    dt_seconds: int = Field(
        default=3600,
        alias='SETTINGS_TROUTE_DT_SECONDS',
        ge=60,
        le=86400,
        description='Routing timestep in seconds',
    )
    routing_method: Literal['muskingum_cunge', 'diffusive_wave'] = Field(
        default='muskingum_cunge',
        alias='TROUTE_ROUTING_METHOD',
        description='Routing scheme: muskingum_cunge or diffusive_wave',
    )

    # Integration settings
    from_model: str = Field(
        default='SUMMA',
        alias='TROUTE_FROM_MODEL',
        description='Source model for runoff input (SUMMA, FUSE, etc.)',
    )
    mannings_n: float = Field(
        default=0.035,
        alias='TROUTE_MANNINGS_N',
        gt=0,
        description="Manning's roughness coefficient",
    )

    # Output settings
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_TROUTE')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_TROUTE')

    # Hydraulic geometry for channel width estimation (W = a * A^b)
    hg_width_coeff: float = Field(
        default=2.71,
        alias='TROUTE_HG_WIDTH_COEFF',
        gt=0,
        description='Hydraulic geometry width coefficient (a in W=a*A^b)',
    )
    hg_width_exp: float = Field(
        default=0.557,
        alias='TROUTE_HG_WIDTH_EXP',
        gt=0,
        le=1.0,
        description='Hydraulic geometry width exponent (b in W=a*A^b)',
    )

    # Sub-timestep for Courant stability
    qts_subdivisions: int = Field(
        default=0,
        alias='TROUTE_QTS_SUBDIVISIONS',
        ge=0,
        le=20,
        description='Sub-timestep divisions (0=auto from Courant)',
    )

    # Calibration settings
    params_to_calibrate: str = Field(
        default='mannings_n',
        alias='TROUTE_PARAMS_TO_CALIBRATE',
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_TROUTE')
    timeout: int = Field(default=3600, alias='TROUTE_TIMEOUT', ge=60, le=86400)


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
TRouteConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_TROUTE_PATH': 'settings_path',
    'SETTINGS_TROUTE_TOPOLOGY': 'topology_file',
    'SETTINGS_TROUTE_CONFIG_FILE': 'config_file',
    'SETTINGS_TROUTE_DT_SECONDS': 'dt_seconds',
}
