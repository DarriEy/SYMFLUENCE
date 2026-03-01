# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Model Calibration Module.

Provides calibration infrastructure for the HYPE (HYdrological Predictions
for the Environment) semi-distributed model developed by SMHI.

Components:
    optimizer: HYPE-specific calibration optimizer supporting DDS and Ostrich
    parameter_manager: Manages HYPE par.txt parameters including SLC-specific values
    targets: Defines calibration targets for streamflow, evaporation, and snow
    worker: Executes HYPE model runs and handles output parsing

The calibration system supports:
- Land use class (SLC) specific parameters
- Soil-dependent parameters
- Routing parameters (velocity, dispersion)
- Multi-site calibration with outlet prioritization
- Spinup period handling
"""
