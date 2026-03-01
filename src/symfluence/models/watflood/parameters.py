# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Parameter Definitions.

WATFLOOD uses Grouped Response Units (GRUs) on a regular grid. Parameters
control surface and subsurface runoff generation, channel routing, and
snow processes. Parameters are stored in `.par` files with per-land-class blocks.

The 16-parameter set below was validated via DDS calibration on the Bow at
Banff basin (KGE=0.84 cal / 0.90 eval).

References:
    Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
    & Flood Forecasting System. University of Waterloo.
"""

from typing import Dict, Union

PARAM_BOUNDS: Dict[str, Dict[str, Union[float, str]]] = {
    'FLZCOEF': {'min': 1e-6, 'max': 0.01, 'transform': 'log'},  # lower zone function coefficient
    'PWR': {'min': 0.5, 'max': 4.0},                              # power on lower zone function
    'R2N': {'min': 0.01, 'max': 0.30},                            # channel Manning roughness multiplier
    'AK': {'min': 1.0, 'max': 100.0},                             # upper zone interflow coefficient (mm/h)
    'AKF': {'min': 1.0, 'max': 100.0},                            # interflow recession coefficient (mm/h)
    'REESSION': {'min': 0.01, 'max': 1.0},                        # baseflow recession coefficient
    'RETN': {'min': 10.0, 'max': 500.0},                          # retention constant (h)
    'AK2': {'min': 0.001, 'max': 1.0, 'transform': 'log'},        # lower zone depletion coefficient
    'AK2FS': {'min': 0.001, 'max': 1.0, 'transform': 'log'},      # lower zone depletion (snow-covered)
    'R3': {'min': 1.0, 'max': 100.0},                              # overbank roughness multiplier
    'DS': {'min': 0.0, 'max': 20.0},                               # surface depression storage (mm)
    'FPET': {'min': 0.5, 'max': 5.0},                              # PET adjustment factor
    'FTALL': {'min': 0.01, 'max': 1.0},                            # forest canopy adjustment
    'FM': {'min': 0.01, 'max': 0.50},                              # melt factor (mm/°C/h)
    'BASE': {'min': -3.0, 'max': 2.0},                             # base temperature for melt (°C)
    'SUBLIM_FACTOR': {'min': 0.0, 'max': 0.5},                     # sublimation fraction
}

DEFAULT_PARAMS: Dict[str, float] = {
    'FLZCOEF': 1e-4,
    'PWR': 2.0,
    'R2N': 0.03,
    'AK': 20.0,
    'AKF': 5.0,
    'REESSION': 0.08,
    'RETN': 40.0,
    'AK2': 0.1,
    'AK2FS': 0.04,
    'R3': 20.0,
    'DS': 1.0,
    'FPET': 2.0,
    'FTALL': 0.2,
    'FM': 0.25,
    'BASE': -0.8,
    'SUBLIM_FACTOR': 0.25,
}

# Parameter file specification for .par format
# Maps parameter names to their section identifiers in WATFLOOD .par files
WATFLOOD_PARAM_SPEC: Dict[str, Dict] = {
    'FLZCOEF': {'section': 'lower_zone', 'per_class': True},
    'PWR': {'section': 'lower_zone_power', 'per_class': True},
    'R2N': {'section': 'river_roughness', 'per_class': False},
    'AK': {'section': 'interflow', 'per_class': True},
    'AKF': {'section': 'interflow_recession', 'per_class': True},
    'REESSION': {'section': 'baseflow', 'per_class': True},
    'RETN': {'section': 'retention', 'per_class': True},
    'AK2': {'section': 'lower_zone_depletion', 'per_class': True},
    'AK2FS': {'section': 'lower_zone_depletion_snow', 'per_class': True},
    'R3': {'section': 'overbank_roughness', 'per_class': False},
    'DS': {'section': 'depression_storage', 'per_class': True},
    'FPET': {'section': 'pet_factor', 'per_class': True},
    'FTALL': {'section': 'forest_canopy', 'per_class': True},
    'FM': {'section': 'melt_factor', 'per_class': True},
    'BASE': {'section': 'base_temperature', 'per_class': True},
    'SUBLIM_FACTOR': {'section': 'sublimation', 'per_class': True},
}

# Maps calibration parameter names to par file keywords used by CHARM's
# read_par_parser.f — keywords appear as `:keyword,` in the .par file.
# The regex for substitution uses these with a colon prefix pattern.
PAR_KEYWORD_MAP: Dict[str, str] = {
    'FLZCOEF': 'flz',          # lower zone function coeff (:RoutingParameters)
    'PWR': 'pwr',              # power on lower zone function (:RoutingParameters)
    'R2N': 'r2n',              # channel Manning's n (:RoutingParameters)
    'AK': 'ak',                # infiltration coeff bare ground (:HydrologicalParameters)
    'AKF': 'akfs',             # infiltration coeff snow-covered (:HydrologicalParameters)
    'REESSION': 'rec',         # interflow coefficient (:HydrologicalParameters)
    'RETN': 'retn',            # retention constant (:HydrologicalParameters)
    'AK2': 'ak2',              # lower zone depletion (:HydrologicalParameters)
    'AK2FS': 'ak2fs',          # lower zone depletion snow-covered (:HydrologicalParameters)
    'R3': 'r3',                # overbank roughness multiplier (:RoutingParameters)
    'DS': 'ds',                # depression storage mm (:HydrologicalParameters)
    'FPET': 'fpet',            # PET adjustment factor (:HydrologicalParameters)
    'FTALL': 'ftall',          # forest canopy adjustment (:HydrologicalParameters)
    'FM': 'fm',                # melt factor (:SnowParameters)
    'BASE': 'base',            # base temperature for melt (:SnowParameters)
    'SUBLIM_FACTOR': 'sublim_factor',  # sublimation fraction (:SnowParameters)
}
