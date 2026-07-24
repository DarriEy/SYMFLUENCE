# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for NOAHMP.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class NoahMPConfig(BaseModel):
    """Noah-MP standalone land surface model configuration.

    Drives noah-owp-modular (NOAA-OWP), a 1-D column Fortran model.
    Repository: https://github.com/NOAA-OWP/noah-owp-modular
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='NOAHMP_INSTALL_PATH')
    exe: str = Field(default='noah_owp_modular.exe', alias='NOAHMP_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_NOAHMP_PATH')
    namelist_file: str = Field(default='namelist.input', alias='NOAHMP_NAMELIST_FILE')
    forcing_file: str = Field(default='forcing.txt', alias='NOAHMP_FORCING_FILE')
    output_file: str = Field(default='output.nc', alias='NOAHMP_OUTPUT_FILE')
    parameter_dir: str = Field(default='parameters', alias='NOAHMP_PARAMETER_DIR')
    timestep: int = Field(default=3600, alias='NOAHMP_TIMESTEP', ge=900, le=3600)
    nsoil: int = Field(default=4, alias='NOAHMP_NSOIL', ge=1, le=10)
    nsnow: int = Field(default=3, alias='NOAHMP_NSNOW', ge=1, le=5)
    dynamic_veg_option: int = Field(default=1, alias='NOAHMP_DYNAMIC_VEG_OPTION', ge=1, le=10)
    canopy_stomatal_option: int = Field(default=1, alias='NOAHMP_CANOPY_STOMATAL_OPTION', ge=1, le=2)
    soil_moisture_option: int = Field(default=1, alias='NOAHMP_SOIL_MOISTURE_OPTION', ge=1, le=3)
    runoff_option: int = Field(default=1, alias='NOAHMP_RUNOFF_OPTION', ge=1, le=8)
    sfc_drag_option: int = Field(default=1, alias='NOAHMP_SFC_DRAG_OPTION', ge=1, le=2)
    supercooled_water_option: int = Field(default=1, alias='NOAHMP_SUPERCOOLED_WATER_OPTION', ge=1, le=2)
    frozen_soil_option: int = Field(default=1, alias='NOAHMP_FROZEN_SOIL_OPTION', ge=1, le=2)
    radiative_transfer_option: int = Field(default=3, alias='NOAHMP_RADIATIVE_TRANSFER_OPTION', ge=1, le=3)
    snow_albedo_option: int = Field(default=2, alias='NOAHMP_SNOW_ALBEDO_OPTION', ge=1, le=2)
    precip_phase_option: int = Field(default=1, alias='NOAHMP_PRECIP_PHASE_OPTION', ge=1, le=2)
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_NOAHMP')
    output_dir: str = Field(default='default', alias='NOAHMP_OUTPUT_DIR')
    spinup_loops: int = Field(default=0, alias='NOAHMP_SPINUP_LOOPS', ge=0, le=100)
    params_to_calibrate: str = Field(
        default='refkdt,dksat,bexp,smcmax,slope,noah_czil',
        alias='NOAHMP_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=7200, alias='NOAHMP_TIMEOUT', ge=60, le=172800)
