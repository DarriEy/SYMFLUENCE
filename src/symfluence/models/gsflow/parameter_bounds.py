# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GSFLOW (PRMS + MODFLOW-NWT) calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing GSFLOW's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters GSFLOW calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  GSFLOW resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for GSFLOW: ``gsflow_K``.

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


#: Tier B -- definitions only GSFLOW resolves.
#: The last 5 are contributed to the catalogue but are NOT part of
#: :data:`BOUND_SET` (see the note there).
PARAMS: Dict[str, ParameterInfo] = {
    'gsflow_soil_moist_max': ParameterInfo(1.0, 15.0, 'inches', 'Max soil moisture storage', 'soil'),
    'gsflow_soil_rechr_max': ParameterInfo(0.5, 5.0, 'inches', 'Max recharge zone storage', 'soil'),
    'gsflow_ssr2gw_rate': ParameterInfo(0.001, 0.5, '1/day', 'Gravity reservoir to GW rate', 'baseflow'),
    'gsflow_gwflow_coef': ParameterInfo(0.001, 0.5, '1/day', 'GW outflow coefficient', 'baseflow'),
    'gsflow_gw_seep_coef': ParameterInfo(0.001, 0.2, '1/day', 'GW seepage coefficient', 'baseflow'),
    'gsflow_SY': ParameterInfo(0.01, 0.4, '-', 'Specific yield', 'soil'),
    'gsflow_slowcoef_lin': ParameterInfo(0.001, 0.5, '1/day', 'Linear gravity drainage coeff', 'baseflow'),
    'gsflow_carea_max': ParameterInfo(0.1, 1.0, '-', 'Max contributing area fraction', 'soil'),
    'gsflow_smidx_coef': ParameterInfo(0.001, 0.1, '-', 'Surface runoff equation coeff', 'soil'),
    'gsflow_jh_coef': ParameterInfo(0.005, 0.03, '-', 'Jensen-Haise PET coefficient', 'et'),
    'gsflow_tmax_allrain': ParameterInfo(1.0, 7.0, 'degC', 'All-rain temperature threshold', 'snow'),
    'gsflow_tmax_allsnow': ParameterInfo(-3.0, 2.0, 'degC', 'All-snow temperature threshold', 'snow'),
    'gsflow_rain_adj': ParameterInfo(0.5, 2.0, '-', 'Rainfall adjustment multiplier', 'snow'),
    'gsflow_snow_adj': ParameterInfo(0.5, 2.0, '-', 'Snowfall adjustment multiplier', 'snow'),
}

#: Tier A -- the catalogue names composing GSFLOW's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
#: NOTE: 5 further GSFLOW entries are registered above but deliberately
#: absent from this list. That mismatch is pre-existing behaviour of
#: get_gsflow_bounds() and is preserved verbatim here; it is tracked
#: separately, not fixed as a side effect of the extraction.
BOUND_SET: List[str] = [
    'gsflow_soil_moist_max',
    'gsflow_soil_rechr_max',
    'gsflow_ssr2gw_rate',
    'gsflow_gwflow_coef',
    'gsflow_gw_seep_coef',
    'gsflow_K',
    'gsflow_SY',
    'gsflow_slowcoef_lin',
    'gsflow_carea_max',
    'gsflow_smidx_coef',
]

#: Catalogue keys are namespaced; parameter managers use unprefixed names.
STRIP_PREFIX = 'gsflow_'


def register_bounds() -> None:
    """Contribute GSFLOW's bounds to the central catalogue."""
    register_model_bounds(
        'GSFLOW',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
