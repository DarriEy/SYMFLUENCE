# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow Model Calibration Module.

Provides calibration infrastructure for the ParFlow integrated hydrologic model,
which solves the 3D Richards equation for variably-saturated subsurface flow
coupled with kinematic overland flow via a Newton-Krylov solver.

Components:
    optimizer: ParFlow-specific calibration optimizer
    parameter_manager: Manages van Genuchten, hydraulic, and overland flow parameters
    worker: Executes ParFlow runs with modified .pfidb parameter databases
    targets: Defines calibration targets for streamflow from overland + subsurface flow

Calibration parameters (8 default):
    K_SAT: Saturated hydraulic conductivity (m/hr) [log-space]
    POROSITY: Total porosity (-)
    VG_ALPHA: van Genuchten alpha parameter (1/m) [log-space]
    VG_N: van Genuchten n shape parameter (-)
    S_RES: Residual saturation (-)
    MANNINGS_N: Manning's roughness coefficient (s/m^1/3) [log-space]
    TOP: Domain surface elevation (m)
    BOT: Domain bottom elevation (m)
"""
