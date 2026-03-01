# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Groundwater (MODFLOW) Calibration Module.

Provides calibration infrastructure for coupling any land surface model
(SUMMA, CLM, MESH, etc.) with MODFLOW 6 for joint surface-groundwater
calibration. Uses dCoupler for graph-based coupling when available,
with sequential file-based coupling as fallback.

Components:
    optimizer: CoupledGWModelOptimizer — sets up parallel dirs for both models
    parameter_manager: CoupledGWParameterManager — joint land-surface + MODFLOW params
    worker: CoupledGWWorker — sequential/dCoupler-based coupled execution
    targets: CoupledGWStreamflowTarget — combined surface runoff + drain discharge

Configuration:
    HYDROLOGICAL_MODEL: COUPLED_GW
    LAND_SURFACE_MODEL: SUMMA (or CLM, MESH, etc.)
    GROUNDWATER_MODEL: MODFLOW
    MODFLOW_PARAMS_TO_CALIBRATE: K,SY,DRAIN_CONDUCTANCE
    COUPLING_MODE: auto (default), dcoupler, or sequential

Calibration loop (per DDS iteration, summary):
    1. DDS proposes parameters → split into land-surface + MODFLOW subsets
    2. Update land-surface files (trialParams.nc for SUMMA, etc.)
    3. Rewrite gwf.npf/sto/drn for MODFLOW params
    4. Run land surface model → extract soil drainage → write recharge.ts
    5. Run MODFLOW 6 → extract drain discharge from gwf.bud
    6. Combine surface runoff + drain discharge → total streamflow
    7. KGE against observed → return to DDS
"""
from .targets import CoupledGWStreamflowTarget  # noqa: F401 - triggers target registration
