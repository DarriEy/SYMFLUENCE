# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MizuRoute Model Calibration Module.

Provides calibration infrastructure for the mizuRoute river routing model,
supporting routing parameter optimization.

Components:
    optimizer: MizuRoute-specific calibration optimizer for routing parameters

The calibration system supports:
- Impulse Response Function (IRF) parameters
- Kinematic Wave Tracking (KWT) parameters
- Diffusive Wave routing parameters
- Channel geometry parameters (width, depth coefficients)
- Manning's roughness coefficient optimization
"""
