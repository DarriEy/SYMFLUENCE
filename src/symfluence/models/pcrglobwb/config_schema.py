# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for PCRGLOBWB.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class PCRGLOBWBConfig(BaseModel):
    """PCR-GLOBWB 2.0 global distributed hydrological model configuration.

    PCR-GLOBWB is a global-scale distributed hydrological model developed
    at Utrecht University. It simulates water storage and fluxes across
    the terrestrial water cycle at 5 or 30 arcminute resolution using
    PCRaster for grid operations.

    Reference:
        Sutanudjaja, E.H., et al. (2018): PCR-GLOBWB 2: a 5 arcmin global
        hydrological and water resources model. Geosci. Model Dev., 11, 2429-2453.
    """

    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='PCRGLOBWB_INSTALL_PATH')
    exe: str = Field(default='deterministic_runner.py', alias='PCRGLOBWB_EXE')
    python_exe: str = Field(default='python', alias='PCRGLOBWB_PYTHON_EXE')
    settings_path: str = Field(default='default', alias='SETTINGS_PCRGLOBWB_PATH')
    config_file: str = Field(default='setup.ini', alias='PCRGLOBWB_CONFIG_FILE')
    clone_map: str = Field(default='clone.map', alias='PCRGLOBWB_CLONE_MAP')
    resolution: str = Field(default='05min', alias='PCRGLOBWB_RESOLUTION')
    spatial_mode: SpatialModeType = Field(default='distributed', alias='PCRGLOBWB_SPATIAL_MODE')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PCRGLOBWB')
    output_dir: str = Field(default='default', alias='PCRGLOBWB_OUTPUT_DIR')
    spinup_years: int = Field(default=0, alias='PCRGLOBWB_SPINUP_YEARS', ge=0, le=50)
    spinup_convergence: bool = Field(default=False, alias='PCRGLOBWB_SPINUP_CONVERGENCE')
    use_opendap: bool = Field(default=False, alias='PCRGLOBWB_USE_OPENDAP')
    input_dir: str = Field(default='default', alias='PCRGLOBWB_INPUT_DIR')
    met_forcing_dir: str = Field(default='default', alias='PCRGLOBWB_MET_FORCING_DIR')
    pet_method: str = Field(default='hamon', alias='PCRGLOBWB_PET_METHOD')
    params_to_calibrate: str = Field(
        default='KSat1,KSat2,recessionCoeff,degreeDayFactor,freezingT,manningsN,ROUTE_ALPHA,ROUTE_BETA,ROUTE_SPLIT,ROUTE_BASEFLOW',
        alias='PCRGLOBWB_PARAMS_TO_CALIBRATE',
    )
    timeout: int = Field(default=14400, alias='PCRGLOBWB_TIMEOUT', ge=60, le=172800)
