# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for PARFLOW.

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


class ParFlowConfig(BaseModel):
    """ParFlow integrated hydrologic model configuration.

    ParFlow solves variably-saturated flow (Richards equation) and
    overland flow. In SYMFLUENCE it is used as an alternative to MODFLOW
    for coupled land surface + groundwater simulations with full vadose
    zone support.

    Reference:
        Kollet, S.J. & Maxwell, R.M. (2006): Integrated surface-groundwater
        flow modeling. Advances in Water Resources 29(7).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='PARFLOW_INSTALL_PATH')
    exe: str = Field(default='parflow', alias='PARFLOW_EXE')
    parflow_dir: str = Field(default='default', alias='PARFLOW_DIR')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_PARFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='PARFLOW_SPATIAL_MODE')

    # Grid discretization
    nx: int = Field(default=1, alias='PARFLOW_NX', ge=1, le=10000)
    ny: int = Field(default=1, alias='PARFLOW_NY', ge=1, le=10000)
    nz: int = Field(default=1, alias='PARFLOW_NZ', ge=1, le=100)
    dx: float = Field(default=1000.0, alias='PARFLOW_DX', gt=0)
    dy: float = Field(default=1000.0, alias='PARFLOW_DY', gt=0)
    dz: float = Field(default=100.0, alias='PARFLOW_DZ', gt=0)

    # Domain geometry
    top: float = Field(default=1500.0, alias='PARFLOW_TOP')
    bot: float = Field(default=1400.0, alias='PARFLOW_BOT')

    # Subsurface properties
    k_sat: float = Field(default=5.0, alias='PARFLOW_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='PARFLOW_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='PARFLOW_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='PARFLOW_VG_N', gt=1.0)
    s_res: float = Field(default=0.1, alias='PARFLOW_S_RES', ge=0, lt=1.0)
    s_sat: float = Field(default=1.0, alias='PARFLOW_S_SAT', gt=0, le=1.0)
    specific_storage: float = Field(default=1e-5, alias='PARFLOW_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='PARFLOW_MANNINGS_N', gt=0)

    # Initial conditions
    initial_pressure: Optional[float] = Field(default=None, alias='PARFLOW_INITIAL_PRESSURE')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='PARFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='PARFLOW_RECHARGE_VARIABLE')

    # Solver
    solver: str = Field(default='Richards', alias='PARFLOW_SOLVER')
    timestep_hours: float = Field(default=1.0, alias='PARFLOW_TIMESTEP_HOURS', gt=0)

    # Parallel execution
    num_procs: int = Field(default=1, alias='PARFLOW_NUM_PROCS', ge=1, le=1024)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PARFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,MANNINGS_N',
        alias='PARFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='PARFLOW_TIMEOUT', ge=60, le=86400)
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='PARFLOW_PARAM_BOUNDS')
