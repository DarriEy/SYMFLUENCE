# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Model Calibration Module.

Provides calibration infrastructure for the FUSE (Framework for Understanding
Structural Errors) model, supporting both lumped and distributed configurations.

Components:
    optimizer: FUSE-specific calibration optimizer using Ostrich or DDS algorithms
    parameter_manager: Manages FUSE parameter bounds, scaling, and trial file generation
    targets: Defines calibration targets (NSE, KGE, RMSE) for FUSE simulations
    worker: Executes FUSE model runs within the calibration optimization loop

The calibration system supports:
- Multiple FUSE model decision combinations
- Snow module (CemaNeige) parameter calibration
- Multi-site calibration for distributed domains
- Parameter sensitivity analysis
"""
