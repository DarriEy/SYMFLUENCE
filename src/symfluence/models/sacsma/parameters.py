"""
SAC-SMA + Snow-17 Parameter Definitions.

Defines all 26 parameters (10 Snow-17 + 16 SAC-SMA) with bounds, defaults,
and NamedTuple structures for the coupled model.

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.

    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
    Computer Models of Watershed Hydrology, 311-366.
"""

from typing import Dict, NamedTuple, Tuple



# =============================================================================
# SNOW-17 PARAMETERS
# =============================================================================

SNOW17_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'SCF': (0.7, 1.4),       # Snowfall correction factor (-)
    'PXTEMP': (-2.0, 2.0),   # Rain/snow threshold temperature (°C)
    'MFMAX': (0.5, 2.0),     # Max melt factor Jun 21 (mm/°C/6hr)
    'MFMIN': (0.05, 0.6),    # Min melt factor Dec 21 (mm/°C/6hr)
    'NMF': (0.05, 0.5),      # Negative melt factor (mm/°C/6hr)
    'MBASE': (0.0, 1.0),     # Base melt temperature (°C)
    'TIPM': (0.01, 1.0),     # Antecedent temperature index weight (-)
    'UADJ': (0.01, 0.2),     # Rain-on-snow wind function (mm/mb/6hr)
    'PLWHC': (0.01, 0.3),    # Liquid water holding capacity (fraction)
    'DAYGM': (0.0, 0.3),     # Daily ground melt (mm/day)
}

SNOW17_DEFAULTS: Dict[str, float] = {
    'SCF': 1.0,
    'PXTEMP': 1.0,
    'MFMAX': 1.0,
    'MFMIN': 0.3,
    'NMF': 0.15,
    'MBASE': 0.0,
    'TIPM': 0.1,
    'UADJ': 0.04,
    'PLWHC': 0.04,
    'DAYGM': 0.0,
}


class Snow17Parameters(NamedTuple):
    """Snow-17 model parameters."""
    SCF: float      # Snowfall correction factor
    PXTEMP: float   # Rain/snow threshold temperature (°C)
    MFMAX: float    # Max melt factor (mm/°C/6hr)
    MFMIN: float    # Min melt factor (mm/°C/6hr)
    NMF: float      # Negative melt factor (mm/°C/6hr)
    MBASE: float    # Base melt temperature (°C)
    TIPM: float     # Antecedent temperature index weight
    UADJ: float     # Wind function for rain-on-snow (mm/mb/6hr)
    PLWHC: float    # Liquid water holding capacity (fraction)
    DAYGM: float    # Daily ground melt (mm/day)


# =============================================================================
# SAC-SMA PARAMETERS
# =============================================================================

SACSMA_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'UZTWM': (1.0, 150.0),    # Upper zone tension water max (mm)
    'UZFWM': (1.0, 150.0),    # Upper zone free water max (mm)
    'UZK': (0.1, 0.75),       # Upper zone lateral depletion rate (1/day)
    'LZTWM': (1.0, 500.0),    # Lower zone tension water max (mm)
    'LZFPM': (1.0, 1000.0),   # Lower zone primary free water max (mm) - LOG
    'LZFSM': (1.0, 1000.0),   # Lower zone supplemental free water max (mm) - LOG
    'LZPK': (0.001, 0.05),    # Primary baseflow depletion rate (1/day) - LOG
    'LZSK': (0.01, 0.25),     # Supplemental baseflow depletion rate (1/day) - LOG
    'ZPERC': (1.0, 350.0),    # Max percolation rate scaling (-) - LOG
    'REXP': (1.0, 5.0),       # Percolation curve exponent (-)
    'PFREE': (0.0, 0.8),      # Fraction percolation to free water (-)
    'PCTIM': (0.0, 0.1),      # Permanent impervious area fraction (-)
    'ADIMP': (0.0, 0.4),      # Additional impervious area fraction (-)
    'RIVA': (0.0, 0.2),       # Riparian vegetation ET fraction (-)
    'SIDE': (0.0, 0.5),       # Deep recharge fraction (-)
    'RSERV': (0.0, 0.4),      # Lower zone free water reserve fraction (-)
}

SACSMA_DEFAULTS: Dict[str, float] = {
    'UZTWM': 50.0,
    'UZFWM': 40.0,
    'UZK': 0.3,
    'LZTWM': 130.0,
    'LZFPM': 60.0,
    'LZFSM': 25.0,
    'LZPK': 0.01,
    'LZSK': 0.05,
    'ZPERC': 40.0,
    'REXP': 2.0,
    'PFREE': 0.3,
    'PCTIM': 0.01,
    'ADIMP': 0.05,
    'RIVA': 0.0,
    'SIDE': 0.0,
    'RSERV': 0.3,
}

# Parameters requiring log transform for calibration (span 1.5-3 orders of magnitude)
LOG_TRANSFORM_PARAMS = {'ZPERC', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK'}


class SacSmaParameters(NamedTuple):
    """SAC-SMA model parameters."""
    UZTWM: float   # Upper zone tension water maximum (mm)
    UZFWM: float   # Upper zone free water maximum (mm)
    UZK: float     # Upper zone lateral depletion rate (1/day)
    LZTWM: float   # Lower zone tension water maximum (mm)
    LZFPM: float   # Lower zone primary free water maximum (mm)
    LZFSM: float   # Lower zone supplemental free water maximum (mm)
    LZPK: float    # Primary baseflow depletion rate (1/day)
    LZSK: float    # Supplemental baseflow depletion rate (1/day)
    ZPERC: float   # Maximum percolation rate scaling (-)
    REXP: float    # Percolation curve exponent (-)
    PFREE: float   # Fraction percolation to free water (-)
    PCTIM: float   # Permanent impervious area fraction (-)
    ADIMP: float   # Additional impervious area fraction (-)
    RIVA: float    # Riparian vegetation ET fraction (-)
    SIDE: float    # Deep recharge fraction (-)
    RSERV: float   # Lower zone free water reserve fraction (-)


# =============================================================================
# COMBINED PARAMETERS
# =============================================================================

# All parameter bounds
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    **SNOW17_PARAM_BOUNDS,
    **SACSMA_PARAM_BOUNDS,
}

# All default values
DEFAULT_PARAMS: Dict[str, float] = {
    **SNOW17_DEFAULTS,
    **SACSMA_DEFAULTS,
}


# =============================================================================
# PARAMETER UTILITIES
# =============================================================================

def create_snow17_params(params_dict: Dict[str, float]) -> Snow17Parameters:
    """Create Snow17Parameters from a dictionary, filling missing with defaults."""
    merged = {**SNOW17_DEFAULTS, **{k: v for k, v in params_dict.items() if k in SNOW17_DEFAULTS}}
    return Snow17Parameters(**merged)


def create_sacsma_params(params_dict: Dict[str, float]) -> SacSmaParameters:
    """Create SacSmaParameters from a dictionary, filling missing with defaults."""
    merged = {**SACSMA_DEFAULTS, **{k: v for k, v in params_dict.items() if k in SACSMA_DEFAULTS}}
    return SacSmaParameters(**merged)


def split_params(params_dict: Dict[str, float]) -> Tuple[Snow17Parameters, SacSmaParameters]:
    """Split combined parameter dict into Snow-17 and SAC-SMA parameter tuples."""
    return create_snow17_params(params_dict), create_sacsma_params(params_dict)


def get_param_transform(param_name: str) -> str:
    """Get the transform type for a parameter ('linear' or 'log')."""
    return 'log' if param_name in LOG_TRANSFORM_PARAMS else 'linear'
