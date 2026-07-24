# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for CWATM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class CWatMConfig(BaseModel):
    """CWatM (Community Water Model) configuration.

    CWatM is a global-scale distributed hydrological model developed
    at IIASA. Pure Python with NumPy-based grid operations (no PCRaster
    dependency). Supports 5/30 arcmin and 30 arcsec resolution.

    Reference:
        Burek, P., et al. (2020): Development of the Community Water
        Model (CWatM v1.04). Geosci. Model Dev., 13, 3267-3298.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='CWATM_INSTALL_PATH')
    exe: str = Field(default='run_cwatm.py', alias='CWATM_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_CWATM_PATH')
    config_file: str = Field(default='settings.ini', alias='CWATM_CONFIG_FILE')
    resolution: str = Field(default='30min', alias='CWATM_RESOLUTION')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='CWATM_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CWATM')
    output_dir: str = Field(default='default', alias='CWATM_OUTPUT_DIR')
    spinup_years: int = Field(default=0, alias='CWATM_SPINUP_YEARS', ge=0, le=50)
    calc_evaporation: bool = Field(default=False, alias='CWATM_CALC_EVAPORATION')
    pet_method: str = Field(default='hamon', alias='CWATM_PET_METHOD')
    params_to_calibrate: str = Field(
        default='SnowMeltCoef,crop_correct,soildepth_factor,arnoBeta_add,recessionCoeff_factor,manningsN',
        alias='CWATM_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=14400, alias='CWATM_TIMEOUT', ge=60, le=172800)
