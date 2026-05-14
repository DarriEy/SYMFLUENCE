# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE-specific parameter regionalization defaults.

The framework classes have moved to:
    symfluence.optimization.regionalization.strategies
"""

from typing import Any, Dict

FUSE_DEFAULT_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
    'MAXWATR_2': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
    'FRACTEN':   {'attribute': 'aridity',      'calibrate_b': False},
    'BASERTE':   {'attribute': 'aridity',      'calibrate_b': True},
    'QB_POWR':   {'attribute': 'aridity',      'calibrate_b': True},
    'PERCRTE':   {'attribute': 'aridity',      'calibrate_b': True},
    'TIMEDELAY': {'attribute': 'precip_mm_yr', 'calibrate_b': False},
    'RTFRAC1':   {'attribute': 'aridity',      'calibrate_b': False},
    'MBASE':     {'attribute': 'elev_m',       'calibrate_b': True},
    'MFMAX':     {'attribute': 'temp_C',       'calibrate_b': True},
    'MFMIN':     {'attribute': 'snow_frac',    'calibrate_b': True},
    'PXTEMP':    {'attribute': 'elev_m',       'calibrate_b': True},
    'LAPSE':     {'attribute': 'elev_m',       'calibrate_b': True},
}
