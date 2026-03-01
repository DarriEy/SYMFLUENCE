# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow Model Calibration Module.

Provides calibration infrastructure for the ParFlow-CLM integrated hydrologic model,
which couples ParFlow's 3D Richards equation solver with CLM's land surface energy
balance, evapotranspiration, snow dynamics, and vegetation processes.

Components:
    optimizer: CLMParFlow-specific calibration optimizer
    parameter_manager: Manages van Genuchten, hydraulic, overland flow, and CLM parameters
    worker: Executes CLMParFlow runs with modified .pfidb parameter databases
    targets: Defines calibration targets for streamflow from overland + subsurface flow

Calibration parameters (14 default):
    K_SAT: Saturated hydraulic conductivity (m/hr) [log-space]
    POROSITY: Total porosity (-)
    VG_ALPHA: van Genuchten alpha parameter (1/m) [log-space]
    VG_N: van Genuchten n shape parameter (-)
    S_RES: Residual saturation (-)
    MANNINGS_N: Manning's roughness coefficient (s/m^1/3) [log-space]
    SNOW17_SCF: Snowfall correction factor (-)
    SNOW17_MFMAX: Max melt factor Jun 21 (mm/C/6hr)
    SNOW17_MFMIN: Min melt factor Dec 21 (mm/C/6hr)
    SNOW17_PXTEMP: Rain/snow threshold temperature (C)
    SNOW_LAPSE_RATE: Temperature lapse rate for elevation bands (C/m)
    ROUTE_ALPHA: Quick flow fraction of overland flow (-)
    ROUTE_K_SLOW: Slow reservoir time constant (days)
    ROUTE_BASEFLOW: Constant baseflow component (m3/s)
"""
