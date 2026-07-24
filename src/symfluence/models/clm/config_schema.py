# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for CLM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class CLMConfig(BaseModel):
    """CLM (Community Land Model / CTSM 5.x) configuration.

    CLM5 is the land component of CESM, providing comprehensive
    biogeophysics, biogeochemistry, hydrology, snow, and vegetation
    dynamics. It is the most physics-heavy LSM in the ensemble.

    Reference:
        Lawrence, D. M., et al. (2019): The Community Land Model version 5.
        JAMES, 11, 4245-4287.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='CLM_INSTALL_PATH')
    exe: str = Field(default='cesm.exe', alias='CLM_EXE')
    # Root for CESM inputdata downloaded on demand. 'default' resolves to
    # <data_dir>/installs/cesm-inputdata (shared, writable scratch — same base
    # as CLM_INSTALL_PATH). Override to point elsewhere if needed.
    cesm_inputdata_path: str = Field(default='default', alias='CLM_CESM_INPUTDATA_PATH')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_CLM_PATH')
    compset: str = Field(default='I2000Clm50SpGs', alias='CLM_COMPSET')
    params_file: str = Field(default='clm5_params.nc', alias='CLM_PARAMS_FILE')
    surfdata_file: str = Field(default='surfdata_clm.nc', alias='CLM_SURFDATA_FILE')
    domain_file: str = Field(default='domain.nc', alias='CLM_DOMAIN_FILE')

    # Spatial mode
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CLM_SPATIAL_MODE')

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CLM')
    hist_nhtfrq: int = Field(default=-24, alias='CLM_HIST_NHTFRQ')
    hist_mfilt: int = Field(default=365, alias='CLM_HIST_MFILT')

    # Calibration
    params_to_calibrate: Optional[str] = Field(
        default=None,
        alias='CLM_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='CLM_TIMEOUT', ge=60, le=86400)
    warmup_days: int = Field(default=365, alias='CLM_WARMUP_DAYS', ge=0, le=3650)
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='CLM_PARAM_BOUNDS')
