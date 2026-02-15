"""
SWAT Parameter Definitions.

This module provides parameter bounds, defaults, change methods, and file
mappings for the SWAT (Soil and Water Assessment Tool) model.

SWAT uses text-based input files (.bsn, .gw, .hru, .sol, .mgt) and supports
three methods for modifying parameters during calibration:
    - r__ (relative): new_value = original_value * (1 + change)
    - v__ (value replacement): new_value = change
    - a__ (absolute): new_value = original_value + change

References:
    Arnold, J.G., et al. (1998): Large area hydrologic modeling and
    assessment Part I: Model development. JAWRA, 34(1), 73-89.

    Abbaspour, K.C. (2015): SWAT-CUP: SWAT Calibration and Uncertainty
    Programs. Eawag, Swiss Federal Institute of Aquatic Science and Technology.
"""

from typing import Dict, Tuple

# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'CN2': {'min': -0.25, 'max': 0.25},           # relative change to curve number
    'ALPHA_BF': {'min': 0.0, 'max': 1.0},          # baseflow alpha factor [days]
    'GW_DELAY': {'min': 0.0, 'max': 500.0},        # groundwater delay time [days]
    'GWQMN': {'min': 0.0, 'max': 5000.0},          # threshold depth for return flow [mm H2O]
    'GW_REVAP': {'min': 0.02, 'max': 0.2},         # groundwater "revap" coefficient
    'ESCO': {'min': 0.0, 'max': 1.0},              # soil evaporation compensation factor
    'SOL_AWC': {'min': -0.25, 'max': 0.25},        # relative change to available water capacity
    'SOL_K': {'min': -0.25, 'max': 0.25},          # relative change to sat. hyd. conductivity
    'SURLAG': {'min': 0.05, 'max': 24.0},          # surface runoff lag coefficient [days]
    'SFTMP': {'min': -5.0, 'max': 5.0},            # snowfall temperature [deg C]
    'SMTMP': {'min': -5.0, 'max': 5.0},            # snowmelt base temperature [deg C]
    'SMFMX': {'min': 0.0, 'max': 10.0},            # max melt rate for snow [mm/deg C/day]
    'SMFMN': {'min': 0.0, 'max': 10.0},            # min melt rate for snow [mm/deg C/day]
    'TIMP': {'min': 0.01, 'max': 1.0},             # snow pack temperature lag factor
}

# Default parameter values (midpoint of bounds or typical values)
DEFAULT_PARAMS: Dict[str, float] = {
    'CN2': 0.0,          # No relative change
    'ALPHA_BF': 0.048,   # Typical baseflow recession constant
    'GW_DELAY': 31.0,    # Typical groundwater delay [days]
    'GWQMN': 1000.0,     # Threshold depth for return flow [mm]
    'GW_REVAP': 0.02,    # Groundwater revap coefficient
    'ESCO': 0.95,        # Soil evaporation compensation factor
    'SOL_AWC': 0.0,      # No relative change
    'SOL_K': 0.0,        # No relative change
    'SURLAG': 4.0,       # Surface runoff lag coefficient
    'SFTMP': 1.0,        # Snowfall temperature [deg C]
    'SMTMP': 0.5,        # Snowmelt base temperature [deg C]
    'SMFMX': 4.5,        # Max snowmelt rate [mm/deg C/day]
    'SMFMN': 4.5,        # Min snowmelt rate [mm/deg C/day]
    'TIMP': 1.0,         # Snow pack temperature lag factor
}

# =============================================================================
# PARAMETER CHANGE METHODS
# =============================================================================

# How each parameter is modified:
#   r__ : relative change  -> new = original * (1 + value)
#   v__ : value replace    -> new = value
#   a__ : absolute change  -> new = original + value
PARAM_CHANGE_METHOD: Dict[str, str] = {
    'CN2': 'r__',       # relative (.mgt)
    'ALPHA_BF': 'v__',  # value (.gw)
    'GW_DELAY': 'v__',  # value (.gw)
    'GWQMN': 'v__',     # value (.gw)
    'GW_REVAP': 'v__',  # value (.gw)
    'ESCO': 'v__',      # value (.bsn or .hru)
    'SOL_AWC': 'r__',   # relative (.sol)
    'SOL_K': 'r__',     # relative (.sol)
    'SURLAG': 'v__',    # value (.bsn)
    'SFTMP': 'v__',     # value (.bsn)
    'SMTMP': 'v__',     # value (.bsn)
    'SMFMX': 'v__',     # value (.bsn)
    'SMFMN': 'v__',     # value (.bsn)
    'TIMP': 'v__',      # value (.bsn)
}

# =============================================================================
# PARAMETER FILE MAP
# =============================================================================

# Which SWAT input file extension contains each parameter
PARAM_FILE_MAP: Dict[str, str] = {
    'CN2': '.mgt',
    'ALPHA_BF': '.gw',
    'GW_DELAY': '.gw',
    'GWQMN': '.gw',
    'GW_REVAP': '.gw',
    'ESCO': '.hru',
    'SOL_AWC': '.sol',
    'SOL_K': '.sol',
    'SURLAG': '.bsn',
    'SFTMP': '.bsn',
    'SMTMP': '.bsn',
    'SMFMX': '.bsn',
    'SMFMN': '.bsn',
    'TIMP': '.bsn',
}


def get_param_bounds_as_tuples() -> Dict[str, Tuple[float, float]]:
    """Return parameter bounds as (min, max) tuples for optimizer compatibility.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples.
    """
    return {
        name: (bounds['min'], bounds['max'])
        for name, bounds in PARAM_BOUNDS.items()
    }
