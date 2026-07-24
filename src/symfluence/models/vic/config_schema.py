# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for VIC.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class VICConfig(BaseModel):
    """VIC (Variable Infiltration Capacity) model configuration.

    VIC is a large-scale, semi-distributed hydrological model that solves
    full water and energy balances. It uses a grid-based structure and is
    typically applied to large river basins.

    Reference:
        Liang, X., D. P. Lettenmaier, E. F. Wood, and S. J. Burges, 1994:
        A simple hydrologically based model of land surface water and energy
        fluxes for general circulation models. J. Geophys. Res., 99(D7), 14415-14428.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='VIC_INSTALL_PATH')
    exe: str = Field(default='vic_image.exe', alias='VIC_EXE')
    driver: Literal['image', 'classic'] = Field(default='image', alias='VIC_DRIVER')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_VIC_PATH')
    global_param_file: str = Field(default='vic_global.txt', alias='VIC_GLOBAL_PARAM_FILE')
    domain_file: str = Field(default='vic_domain.nc', alias='VIC_DOMAIN_FILE')
    params_file: str = Field(default='vic_params.nc', alias='VIC_PARAMS_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='auto', alias='VIC_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_VIC')
    output_prefix: str = Field(default='vic_output', alias='VIC_OUTPUT_PREFIX')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default='infilt,Ds,Dsmax,Ws,c,depth1,depth2,depth3,expt,expt_increase,Ksat,Ksat_decay,Wcr_FRACT,Wpwp_ratio,snow_rough,max_snow_albedo,min_rain_temp,max_snow_temp,elev_offset',
        alias='VIC_PARAMS_TO_CALIBRATE'
    )

    # Model options
    full_energy: bool = Field(default=True, alias='VIC_FULL_ENERGY')
    frozen_soil: bool = Field(default=True, alias='VIC_FROZEN_SOIL')
    snow_band: bool = Field(default=False, alias='VIC_SNOW_BAND')
    n_snow_bands: int = Field(default=10, alias='VIC_N_SNOW_BANDS', ge=1, le=25)
    pfactor_per_km: float = Field(default=0.0005, alias='VIC_PFACTOR_PER_KM', ge=0.0, le=0.01)

    # Timing
    model_steps_per_day: int = Field(default=24, alias='VIC_STEPS_PER_DAY', ge=1, le=48)

    # Execution
    timeout: int = Field(default=7200, alias='VIC_TIMEOUT', ge=60, le=86400)
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='VIC_PARAM_BOUNDS')
