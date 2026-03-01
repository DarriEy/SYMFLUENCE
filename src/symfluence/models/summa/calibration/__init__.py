# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Model Calibration Module.

Provides calibration infrastructure for the SUMMA (Structure for Unifying
Multiple Modeling Alternatives) model, supporting its flexible physics
decision framework.

Components:
    optimizer: SUMMA-specific calibration optimizer with multi-objective support
    optimizer_mixin: Shared optimization functionality for SUMMA variants
    parameter_manager: Manages localParamInfo.txt and trialParamFile parameters
    targets: Defines calibration targets for streamflow, ET, SWE, and soil moisture
    worker: Executes SUMMA model runs with file manager configuration

The calibration system supports:
- HRU-level and basin-level parameter calibration
- Physics decision sensitivity analysis
- Layered soil parameter calibration
- Multi-variable objective functions (streamflow, snow, soil moisture)
- Integration with mizuRoute for routed streamflow calibration
"""
