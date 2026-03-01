# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR Model Calibration Module.

Provides calibration infrastructure for the GR (Génie Rural) family of models,
including GR4J, GR5J, and GR6J variants with optional CemaNeige snow module.

Components:
    optimizer: GR-specific calibration optimizer with airGR integration
    parameter_manager: Manages GR model parameters (X1-X6) and CemaNeige snow parameters
    targets: Defines calibration targets for streamflow and snow-related metrics
    worker: Executes GR model runs using the airGR R interface

The calibration system supports:
- GR4J (4 parameters), GR5J (5 parameters), GR6J (6 parameters) variants
- CemaNeige snow module calibration (CTG, Kf parameters)
- Warm-up period handling for stable state initialization
- NSE, KGE, and log-transformed objective functions
"""
