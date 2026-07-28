# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Routing model configuration classes."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class MizuRouteConfig(BaseModel):
    """mizuRoute routing model configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MIZUROUTE_INSTALL_PATH')
    exe: str = Field(default='mizuRoute.exe', alias='MIZUROUTE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_MIZU_PATH')
    within_basin: int = Field(default=0, alias='SETTINGS_MIZU_WITHIN_BASIN')
    routing_dt: int = Field(default=3600, alias='SETTINGS_MIZU_ROUTING_DT')
    routing_units: str = Field(default='m/s', alias='SETTINGS_MIZU_ROUTING_UNITS')
    routing_var: str = Field(default='averageRoutedRunoff', alias='SETTINGS_MIZU_ROUTING_VAR')
    output_freq: str = Field(default='single', alias='SETTINGS_MIZU_OUTPUT_FREQ')
    output_vars: str = Field(default='1', alias='SETTINGS_MIZU_OUTPUT_VARS')
    make_outlet: str = Field(default='n/a', alias='SETTINGS_MIZU_MAKE_OUTLET')
    needs_remap: bool = Field(default=False, alias='SETTINGS_MIZU_NEEDS_REMAP')
    topology: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY')
    parameters: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETERS')
    control_file: str = Field(default='mizuroute.control', alias='SETTINGS_MIZU_CONTROL_FILE')
    remap: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP')
    from_model: str = Field(default='default', alias='MIZU_FROM_MODEL')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_MIZUROUTE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MIZUROUTE')
    # Additional mizuRoute settings
    output_var: str = Field(default='IRFroutedRunoff', alias='SETTINGS_MIZU_OUTPUT_VAR')
    parameter_file: str = Field(default='param.nml.default', alias='SETTINGS_MIZU_PARAMETER_FILE')
    remap_file: str = Field(default='routing_remap.nc', alias='SETTINGS_MIZU_REMAP_FILE')
    topology_file: str = Field(default='topology.nc', alias='SETTINGS_MIZU_TOPOLOGY_FILE')
    params_to_calibrate: str = Field(
        default='velo,diff',
        alias='MIZUROUTE_PARAMS_TO_CALIBRATE'
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_MIZUROUTE')
    timeout: int = Field(default=3600, alias='MIZUROUTE_TIMEOUT', ge=60, le=86400)  # seconds (1min to 24hr)
    num_threads: Optional[int] = Field(
        default=None, alias='MIZUROUTE_NUM_THREADS', ge=1,
        description='OpenMP thread count for the mizuRoute subprocess'
    )
    time_rounding_freq: str = Field(
        default='h',
        alias='MIZUROUTE_TIME_ROUNDING_FREQ',
        description='Frequency for rounding time values (e.g., "h" for hour, "min" for minute, "none" to disable)'
    )
    topology_staleness: str = Field(
        default='warn',
        alias='MIZUROUTE_TOPOLOGY_STALENESS',
        description=(
            'Action when topology.nc no longer matches the geofabric it was built '
            'from (segment/HRU counts differ, or a source shapefile is newer): '
            '"warn" (default), "error" to refuse to run, "regenerate" to rebuild '
            'it from the current shapefiles, or "ignore" to skip the check'
        )
    )

    @field_validator('output_vars', mode='before')
    @classmethod
    def normalize_output_vars(cls, v):
        """Convert list or other types to string for output_vars"""
        if isinstance(v, list):
            return ' '.join(str(item).strip() for item in v)
        return str(v)


# NOTE: DRouteConfig moved to the external ``droute`` package (python/droute/config.py) and is
# registered with SYMFLUENCE via ``model_manifest(config_schema=DRouteConfig)`` in
# ``droute.register()`` — matching the JAX-model plugin pattern where the model package owns its
# own typed config schema.


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



__all__ = [
    'MizuRouteConfig',
    'TRouteConfig',
]
