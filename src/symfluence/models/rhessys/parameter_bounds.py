# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RHESSys calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing RHESSYS's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters RHESSYS calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  RHESSYS resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for RHESSYS: ``m``, ``rhessys_soil_depth``.

:func:`register_bounds` is called from this package's ``register()``, i.e. at
plugin-discovery time, which runs on ``import symfluence`` -- before any
calibration code can read bounds.
"""
from __future__ import annotations

from typing import Dict, List

from symfluence.core.calibration.parameters.parameter_bounds_registry import (
    ParameterInfo,
    register_model_bounds,
)

#: Tier B -- definitions only RHESSYS resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'sat_to_gw_coeff': ParameterInfo(0.0001, 0.1, '1/day', 'Saturation to groundwater coefficient', 'baseflow', 'log'),
    'gw_loss_coeff': ParameterInfo(0.001, 0.5, '-', 'Groundwater loss coefficient (controls slow baseflow)', 'baseflow', 'log'),
    'gw_loss_fast_coeff': ParameterInfo(0.01, 1.0, '-', 'Fast groundwater loss coefficient', 'baseflow', 'log'),
    'gw_loss_fast_threshold': ParameterInfo(0.05, 0.5, 'm', 'GW storage threshold for fast flow activation', 'baseflow'),
    'psi_air_entry': ParameterInfo(-10.0, -1.0, 'kPa', 'Air entry pressure (negative)', 'soil'),
    'pore_size_index': ParameterInfo(0.05, 0.4, '-', 'Pore size distribution index', 'soil'),
    'porosity_0': ParameterInfo(0.3, 0.6, 'm³/m³', 'Surface porosity', 'soil'),
    'porosity_decay': ParameterInfo(0.1, 0.8, 'm³/m³', 'Porosity decay with depth', 'soil'),
    'Ksat_0': ParameterInfo(0.0001, 0.1, 'm/day', 'Surface saturated conductivity (lateral)', 'soil', 'log'),
    'Ksat_0_v': ParameterInfo(0.0001, 0.5, 'm/day', 'Vertical saturated conductivity', 'soil', 'log'),
    'm_z': ParameterInfo(0.2, 3.0, '-', 'Vertical decay of Ksat with depth', 'soil'),
    'active_zone_z': ParameterInfo(0.5, 3.0, 'm', 'Active zone depth', 'soil'),
    'theta_mean_std_p1': ParameterInfo(0.01, 0.5, '-', 'Std dev of saturation deficit (controls partial saturation area)', 'soil'),
    'theta_mean_std_p2': ParameterInfo(0.0, 0.3, '-', 'Second parameter for saturation deficit variance', 'soil'),
    'max_snow_temp': ParameterInfo(-2.0, 2.0, '°C', 'Max temp for snow (rain/snow threshold)', 'snow'),
    'min_rain_temp': ParameterInfo(-6.0, 0.0, '°C', 'Min temp for rain (all snow below this)', 'snow'),
    'snow_melt_Tcoef': ParameterInfo(0.5, 8.0, 'mm/°C/day', 'Snow melt temperature coefficient', 'snow'),
    'snow_water_capacity': ParameterInfo(0.1, 1.5, '-', 'Snow water holding capacity coefficient', 'snow'),
    'maximum_snow_energy_deficit': ParameterInfo(-1500.0, -100.0, 'kJ/m²', 'Maximum snow energy deficit (must be negative)', 'snow'),
    'epc.max_lai': ParameterInfo(0.5, 8.0, 'm²/m²', 'Maximum LAI', 'et'),
    'epc.gl_smax': ParameterInfo(0.001, 0.02, 'm/s', 'Maximum stomatal conductance', 'et', 'log'),
    'epc.gl_c': ParameterInfo(1e-05, 0.001, 'm/s', 'Cuticular conductance', 'et', 'log'),
    'epc.vpd_open': ParameterInfo(0.1, 2.0, 'kPa', 'VPD at stomatal opening', 'et'),
    'epc.vpd_close': ParameterInfo(2.0, 6.0, 'kPa', 'VPD at stomatal closure', 'et'),
    'n_routing_power': ParameterInfo(0.1, 1.0, '-', 'Routing power exponent', 'routing'),
    'precip_lapse_rate': ParameterInfo(0.5, 1.5, '-', 'Precipitation multiplier (corrects forcing bias)', 'forcing'),
}

#: Tier A -- the catalogue names composing RHESSYS's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'sat_to_gw_coeff',
    'gw_loss_coeff',
    'gw_loss_fast_coeff',
    'gw_loss_fast_threshold',
    'psi_air_entry',
    'pore_size_index',
    'porosity_0',
    'porosity_decay',
    'Ksat_0',
    'Ksat_0_v',
    'm',
    'm_z',
    'rhessys_soil_depth',
    'active_zone_z',
    'theta_mean_std_p1',
    'theta_mean_std_p2',
    'max_snow_temp',
    'min_rain_temp',
    'snow_melt_Tcoef',
    'snow_water_capacity',
    'maximum_snow_energy_deficit',
    'epc.max_lai',
    'epc.gl_smax',
    'epc.gl_c',
    'epc.vpd_open',
    'epc.vpd_close',
    'n_routing_power',
    'precip_lapse_rate',
]

#: Catalogue keys are namespaced; parameter managers use unprefixed names.
STRIP_PREFIX = 'rhessys_'


def register_bounds() -> None:
    """Contribute RHESSYS's bounds to the central catalogue."""
    register_model_bounds(
        'RHESSYS',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
