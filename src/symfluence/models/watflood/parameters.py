"""
WATFLOOD Parameter Definitions.

WATFLOOD uses Grouped Response Units (GRUs) on a regular grid. Parameters
control surface and subsurface runoff generation, channel routing, and
snow processes. Parameters are stored in `.par` files with per-land-class blocks.

References:
    Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
    & Flood Forecasting System. University of Waterloo.
"""

from typing import Dict

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'R2N': {'min': 0.01, 'max': 5.0},          # channel Manning roughness multiplier
    'R1N': {'min': 0.01, 'max': 5.0},          # overland flow roughness multiplier
    'AK': {'min': 0.001, 'max': 1.0},           # upper zone interflow coefficient (mm/h)
    'AKF': {'min': 0.001, 'max': 1.0},          # interflow recession coefficient (mm/h)
    'REESSION': {'min': 0.0, 'max': 1.0},       # baseflow recession coefficient
    'FLZCOEF': {'min': 0.0, 'max': 0.1},        # lower zone function coefficient
    'PWR': {'min': 1.0, 'max': 3.0},            # power on lower zone function
    'THETA': {'min': 0.0, 'max': 1.0},          # soil moisture content parameter
    'DS': {'min': 0.0, 'max': 1.0},             # surface depression storage fraction
    'MANNING_N': {'min': 0.01, 'max': 0.5},     # channel Manning roughness (s/m^1/3)
}

DEFAULT_PARAMS: Dict[str, float] = {
    'R2N': 1.0,
    'R1N': 1.0,
    'AK': 0.1,
    'AKF': 0.05,
    'REESSION': 0.3,
    'FLZCOEF': 0.01,
    'PWR': 2.0,
    'THETA': 0.5,
    'DS': 0.1,
    'MANNING_N': 0.035,
}

# Parameter file specification for .par format
# Maps parameter names to their section identifiers in WATFLOOD .par files
WATFLOOD_PARAM_SPEC: Dict[str, Dict] = {
    'R2N': {'section': 'river_roughness', 'per_class': False},
    'R1N': {'section': 'overland_roughness', 'per_class': True},
    'AK': {'section': 'interflow', 'per_class': True},
    'AKF': {'section': 'interflow_recession', 'per_class': True},
    'REESSION': {'section': 'baseflow', 'per_class': True},
    'FLZCOEF': {'section': 'lower_zone', 'per_class': True},
    'PWR': {'section': 'lower_zone_power', 'per_class': True},
    'THETA': {'section': 'soil_moisture', 'per_class': True},
    'DS': {'section': 'depression_storage', 'per_class': True},
    'MANNING_N': {'section': 'manning', 'per_class': False},
}
