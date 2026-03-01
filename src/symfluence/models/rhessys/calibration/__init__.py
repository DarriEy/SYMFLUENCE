# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Model Calibration Module.

Provides calibration infrastructure for the RHESSys (Regional Hydro-Ecologic
Simulation System) ecohydrology model, supporting landscape-scale simulations.

Components:
    optimizer: RHESSys-specific calibration optimizer with spatial output support
    parameter_manager: Manages worldfile parameters and def file overrides
    targets: Defines calibration targets for streamflow, ET, and vegetation metrics
    worker: Executes RHESSys model runs with TEC file management

The calibration system supports:
- Soil hydraulic parameters (Ksat, porosity, pore size)
- Vegetation parameters (stomatal conductance, LAI)
- Carbon and nitrogen cycling parameters (optional)
- Fire module (WMFire) parameters when enabled
- Multi-patch calibration for distributed domains
"""
