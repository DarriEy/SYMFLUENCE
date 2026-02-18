"""
GSFLOW Parameter Definitions.

GSFLOW couples PRMS surface processes with MODFLOW-NWT groundwater.
Calibration parameters span both PRMS (soil, runoff) and MODFLOW-NWT (K, SY).

References:
    Markstrom, S.L., et al. (2008): GSFLOW—Coupled Ground-Water and
    Surface-Water Flow Model. USGS Techniques and Methods 6-D1.
"""

from typing import Dict

PARAM_BOUNDS: Dict[str, Dict[str, float]] = {
    # PRMS soil zone
    'soil_moist_max': {'min': 1.0, 'max': 15.0},       # inches, max soil moisture storage
    'soil_rechr_max': {'min': 0.5, 'max': 5.0},         # inches, max recharge zone storage
    'ssr2gw_rate': {'min': 0.001, 'max': 0.5},           # 1/day, gravity reservoir to GW
    'gwflow_coef': {'min': 0.001, 'max': 0.5},           # 1/day, GW outflow coefficient
    'gw_seep_coef': {'min': 0.001, 'max': 0.2},          # 1/day, GW seepage coefficient
    # MODFLOW-NWT
    'K': {'min': 0.001, 'max': 100.0},                   # m/d, hydraulic conductivity
    'SY': {'min': 0.01, 'max': 0.4},                     # -, specific yield
    # PRMS runoff
    'slowcoef_lin': {'min': 0.001, 'max': 0.5},          # 1/day, linear gravity drainage
    'carea_max': {'min': 0.1, 'max': 1.0},               # -, max contributing area fraction
    'smidx_coef': {'min': 0.001, 'max': 0.10},           # -, surface runoff equation coeff
    # Snow / climate
    'jh_coef': {'min': 0.005, 'max': 0.030},             # -, Jensen-Haise PET coefficient
    'tmax_allrain': {'min': 1.0, 'max': 7.0},            # °C, all-rain temperature threshold
    'tmax_allsnow': {'min': -3.0, 'max': 2.0},           # °C, all-snow temperature threshold
    'rain_adj': {'min': 0.5, 'max': 2.0},                # -, rainfall adjustment multiplier
    'snow_adj': {'min': 0.5, 'max': 2.0},                # -, snowfall adjustment multiplier
}

DEFAULT_PARAMS: Dict[str, float] = {
    'soil_moist_max': 6.0,
    'soil_rechr_max': 2.0,
    'ssr2gw_rate': 0.1,
    'gwflow_coef': 0.015,
    'gw_seep_coef': 0.01,
    'K': 1.0,
    'SY': 0.15,
    'slowcoef_lin': 0.015,
    'carea_max': 0.6,
    'smidx_coef': 0.01,
    'jh_coef': 0.014,
    'tmax_allrain': 3.3,
    'tmax_allsnow': 0.0,
    'rain_adj': 1.0,
    'snow_adj': 1.0,
}

# PRMS parameter file specification (for ####-delimited blocks)
PRMS_PARAM_SPEC: Dict[str, Dict] = {
    'soil_moist_max': {'dimension': 'nhru', 'type_code': 2},
    'soil_rechr_max': {'dimension': 'nhru', 'type_code': 2},
    'ssr2gw_rate': {'dimension': 'nhru', 'type_code': 2},
    'gwflow_coef': {'dimension': 'nhru', 'type_code': 2},
    'gw_seep_coef': {'dimension': 'nhru', 'type_code': 2},
    'slowcoef_lin': {'dimension': 'nhru', 'type_code': 2},
    'carea_max': {'dimension': 'nhru', 'type_code': 2},
    'smidx_coef': {'dimension': 'nhru', 'type_code': 2},
    'jh_coef': {'dimension': 'nmonths', 'type_code': 2},
    'tmax_allrain': {'dimension': 'nmonths', 'type_code': 2},
    'tmax_allsnow': {'dimension': 'nmonths', 'type_code': 2},
    'rain_adj': {'dimension': 'nmonths', 'type_code': 2},
    'snow_adj': {'dimension': 'nmonths', 'type_code': 2},
}

# MODFLOW-NWT UPW package parameter specification
MODFLOW_PARAM_SPEC: Dict[str, Dict] = {
    'K': {'package': 'UPW', 'layer': 1},
    'SY': {'package': 'UPW', 'layer': 1},
}
