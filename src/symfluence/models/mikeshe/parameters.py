"""
MIKE-SHE Parameter Definitions.

This module provides parameter bounds, defaults, and XML element mappings
for the MIKE-SHE (DHI) integrated physics-based hydrological model.

MIKE-SHE uses XML-based `.she` setup files. During calibration, parameters
are modified by parsing the XML tree with `xml.etree.ElementTree` and
updating the corresponding element values in-place.

Parameter Categories:
    - Overland Flow: Manning's M, detention storage
    - Unsaturated Zone: Ks_uz, theta_sat, theta_fc, theta_wp
    - Saturated Zone: Ks_sz_h, specific_yield
    - Snow: degree-day factor, snow threshold temperature
    - Vegetation: maximum canopy interception storage

References:
    Graham, D.N. & Butts, M.B. (2005): Flexible, integrated watershed
    modelling with MIKE SHE. Watershed Models, 245-272.

    DHI (2017): MIKE SHE User Manual. Danish Hydraulic Institute.
"""

from typing import Dict, Tuple


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    'manning_m': {'min': 1.0, 'max': 100.0},            # Manning's M (1/n) [m^(1/3)/s]
    'detention_storage': {'min': 0.0, 'max': 10.0},      # detention storage [mm]
    'Ks_uz': {'min': 1e-8, 'max': 1e-3},                 # unsaturated zone Ks [m/s]
    'theta_sat': {'min': 0.3, 'max': 0.6},               # saturated water content [-]
    'theta_fc': {'min': 0.15, 'max': 0.45},              # field capacity [-]
    'theta_wp': {'min': 0.05, 'max': 0.25},              # wilting point [-]
    'Ks_sz_h': {'min': 1e-7, 'max': 1e-2},               # saturated zone horizontal Ks [m/s]
    'specific_yield': {'min': 0.01, 'max': 0.35},        # specific yield [-]
    'ddf': {'min': 0.5, 'max': 8.0},                     # degree-day factor [mm/deg_C/day]
    'snow_threshold': {'min': -2.0, 'max': 2.0},         # snow threshold temp [deg_C]
    'max_canopy_storage': {'min': 0.1, 'max': 5.0},      # max canopy interception [mm]
}

# Default parameter values (physically reasonable starting points)
DEFAULT_PARAMS: Dict[str, float] = {
    'manning_m': 20.0,
    'detention_storage': 2.0,
    'Ks_uz': 1e-5,
    'theta_sat': 0.45,
    'theta_fc': 0.3,
    'theta_wp': 0.15,
    'Ks_sz_h': 1e-4,
    'specific_yield': 0.1,
    'ddf': 3.0,
    'snow_threshold': 0.0,
    'max_canopy_storage': 1.5,
}


# =============================================================================
# XML ELEMENT MAPPINGS
# =============================================================================

# Maps calibration parameter names to their XPath locations within the .she XML.
# MIKE-SHE .she files use a hierarchical XML structure where each physical
# component (overland flow, unsaturated zone, saturated zone, snow, vegetation)
# has its own section. These paths represent typical default structures;
# actual paths may vary depending on the MIKE-SHE version and project setup.
PARAM_XML_PATHS: Dict[str, str] = {
    'manning_m': './/OverlandFlow/ManningM',
    'detention_storage': './/OverlandFlow/DetentionStorage',
    'Ks_uz': './/UnsaturatedFlow/HydraulicConductivity',
    'theta_sat': './/UnsaturatedFlow/SaturatedMoistureContent',
    'theta_fc': './/UnsaturatedFlow/FieldCapacity',
    'theta_wp': './/UnsaturatedFlow/WiltingPoint',
    'Ks_sz_h': './/SaturatedFlow/HorizontalConductivity',
    'specific_yield': './/SaturatedFlow/SpecificYield',
    'ddf': './/SnowMelt/DegreeDayFactor',
    'snow_threshold': './/SnowMelt/ThresholdTemperature',
    'max_canopy_storage': './/Vegetation/MaxCanopyStorage',
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
