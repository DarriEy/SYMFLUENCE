# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Routing model configuration classes."""

from typing import Literal

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
    time_rounding_freq: str = Field(
        default='h',
        alias='MIZUROUTE_TIME_ROUNDING_FREQ',
        description='Frequency for rounding time values (e.g., "h" for hour, "min" for minute, "none" to disable)'
    )

    @field_validator('output_vars', mode='before')
    @classmethod
    def normalize_output_vars(cls, v):
        """Convert list or other types to string for output_vars"""
        if isinstance(v, list):
            return ' '.join(str(item).strip() for item in v)
        return str(v)


class DRouteConfig(BaseModel):
    """dRoute routing model configuration (EXPERIMENTAL)

    dRoute is a C++ river routing library with Python bindings that offers:
    - Multiple routing methods (Muskingum-Cunge, IRF, Lag, Diffusive Wave, KWT)
    - Native automatic differentiation for gradient-based calibration
    - mizuRoute-compatible network topology format
    """
    model_config = FROZEN_CONFIG

    # Execution settings
    execution_mode: Literal['python', 'subprocess'] = Field(
        default='python',
        alias='DROUTE_EXECUTION_MODE',
        description='Execution mode: python API (preferred) or subprocess fallback'
    )
    install_path: str = Field(default='default', alias='DROUTE_INSTALL_PATH')
    exe: str = Field(default='droute', alias='DROUTE_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_DROUTE_PATH')

    # Routing configuration
    routing_method: Literal['muskingum_cunge', 'irf', 'lag', 'diffusive_wave', 'kwt'] = Field(
        default='muskingum_cunge',
        alias='DROUTE_ROUTING_METHOD',
        description='Routing scheme to use'
    )
    routing_dt: int = Field(
        default=3600,
        alias='DROUTE_ROUTING_DT',
        ge=60,
        le=86400,
        description='Routing timestep in seconds'
    )

    # Gradient/AD settings
    enable_gradients: bool = Field(
        default=False,
        alias='DROUTE_ENABLE_GRADIENTS',
        description='Enable automatic differentiation for gradient-based calibration'
    )
    ad_backend: Literal['codipack', 'enzyme'] = Field(
        default='codipack',
        alias='DROUTE_AD_BACKEND',
        description='AD backend (requires dRoute compiled with AD support)'
    )

    # Topology configuration
    topology_file: str = Field(default='topology.nc', alias='DROUTE_TOPOLOGY_FILE')
    topology_format: Literal['netcdf', 'geojson', 'csv'] = Field(
        default='netcdf',
        alias='DROUTE_TOPOLOGY_FORMAT'
    )
    config_file: str = Field(default='droute_config.yaml', alias='DROUTE_CONFIG_FILE')

    # Integration settings
    from_model: str = Field(
        default='default',
        alias='DROUTE_FROM_MODEL',
        description='Source model for runoff input (SUMMA, FUSE, GR, etc.)'
    )

    # Output settings
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_DROUTE')
    experiment_log: str = Field(default='default', alias='EXPERIMENT_LOG_DROUTE')

    # Calibration settings
    params_to_calibrate: str = Field(
        default='velocity,diffusivity',
        alias='DROUTE_PARAMS_TO_CALIBRATE'
    )
    calibrate: bool = Field(default=False, alias='CALIBRATE_DROUTE')
    timeout: int = Field(default=3600, alias='DROUTE_TIMEOUT', ge=60, le=86400)


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
    'DRouteConfig',
    'TRouteConfig',
]
