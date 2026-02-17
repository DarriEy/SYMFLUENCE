"""
WRF-Hydro Parameter Definitions.

This module provides parameter bounds and default values for the WRF-Hydro model,
used during calibration with the SYMFLUENCE optimization framework.

WRF-Hydro couples the Noah-MP land surface model with terrain-following routing.
Calibration parameters control infiltration, overland flow, subsurface lateral
flow, and soil hydraulic properties.

Parameters are applied to WRF-Hydro via:
  1. Hydro namelist (hydro.namelist) for routing parameters
  2. HRLDAS namelist (namelist.hrldas) for LSM parameters
  3. Geogrid/parameter netCDF files for spatially distributed parameters

References:
    Gochis, D.J., et al. (2020): The WRF-Hydro modeling system technical
    description, (Version 5.1.1). NCAR Technical Note.

    Yucel, I., et al. (2015): Calibration and evaluation of a flood
    forecasting system: Utility of numerical weather prediction model,
    data assimilation and satellite-based rainfall. J. Hydrology.
"""

from typing import Dict

# =============================================================================
# CALIBRATION PARAMETER BOUNDS  (used by the optimizer / parameter manager)
# =============================================================================

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'REFKDT': {
        'min': 0.1, 'max': 10.0,                   # surface infiltration parameter
    },
    'SLOPE': {
        'min': 0.0, 'max': 1.0,                    # linear drainage coefficient for subsurface
    },
    'OVROUGHRTFAC': {
        'min': 0.1, 'max': 5.0,                    # overland roughness scaling factor
    },
    'RETDEPRTFAC': {
        'min': 0.0, 'max': 10.0,                   # retention depth scaling factor
    },
    'LKSATFAC': {
        'min': 1.0, 'max': 10000.0,                # lateral Ksat scaling factor
    },
    'BEXP': {
        'min': 1.0, 'max': 10.0,                   # Clapp-Hornberger B parameter exponent
    },
    'DKSAT': {
        'min': 1e-7, 'max': 1e-5,                  # saturated hydraulic conductivity [m/s]
    },
    'SMCMAX': {
        'min': 0.3, 'max': 0.6,                    # porosity (max soil moisture content)
    },
}


# =============================================================================
# DEFAULT PARAMETERS  (simple name -> value mapping for quick access)
# =============================================================================

DEFAULT_PARAMS: Dict[str, float] = {
    'REFKDT': 3.0,
    'SLOPE': 0.1,
    'OVROUGHRTFAC': 1.0,
    'RETDEPRTFAC': 1.0,
    'LKSATFAC': 1000.0,
    'BEXP': 4.0,
    'DKSAT': 1e-6,
    'SMCMAX': 0.45,
}


# =============================================================================
# PARAMETER TARGET FILE MAPPING
#
# Maps parameter names to where they live in WRF-Hydro configuration.
# target: 'hydro_namelist', 'hrldas_namelist', or 'spatial_file'
# =============================================================================

WRFHYDRO_PARAM_TARGETS: Dict[str, Dict] = {
    'REFKDT':        {'target': 'genparm_tbl', 'key': 'REFKDT_DATA'},
    'SLOPE':         {'target': 'genparm_tbl', 'key': 'SLOPE_DATA'},
    'OVROUGHRTFAC':  {'target': 'hydro_namelist', 'section': 'HYDRO_nlist'},
    'RETDEPRTFAC':   {'target': 'hydro_namelist', 'section': 'HYDRO_nlist'},
    'LKSATFAC':      {'target': 'hydro_namelist', 'section': 'HYDRO_nlist'},
    'BEXP':          {'target': 'soilparm_tbl', 'column': 'BB', 'column_index': 1},
    'DKSAT':         {'target': 'soilparm_tbl', 'column': 'SATDK', 'column_index': 7},
    'SMCMAX':        {'target': 'soilparm_tbl', 'column': 'MAXSMC', 'column_index': 4},
}
