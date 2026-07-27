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

This module is also the ONE place GSFLOW's bound numbers are written.
``symfluence.models.gsflow.parameters.PARAM_BOUNDS`` -- what
``GSFLOWParameterManager._load_parameter_bounds()`` actually reads at
calibration time -- used to be an independent literal dict here-and-there
duplicate; it is now :data:`CALIBRATION_BOUNDS`, derived below. See
:data:`LOCAL_ONLY` for the single number that still cannot be sourced from the
central catalogue.
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
#: NOTE: this list does NOT match what GSFLOW actually calibrates, and the
#: mismatch is pre-existing behaviour of ``get_gsflow_bounds()`` preserved
#: verbatim (the model-bounds parity snapshot is pinned to it). Measured
#: against ``GSFLOWParameterManager``'s default ``GSFLOW_PARAMS_TO_CALIBRATE``:
#:
#: * absent here but calibrated by default -- ``jh_coef``, ``tmax_allsnow``,
#:   ``rain_adj``, ``snow_adj`` (``tmax_allrain`` is registered but correctly
#:   not calibrated: PRMS6 ignores it in COUPLED mode);
#: * present here but NOT calibrated -- ``soil_rechr_max``, ``gwflow_coef``,
#:   ``gw_seep_coef``, for the same "inert in COUPLED mode" reason.
#:
#: Fixing it changes ``get_gsflow_bounds()`` output and so requires
#: regenerating ``tests/unit/core/data/model_bounds_snapshot.json``; it is
#: reported, not changed as a side effect. What the manager reads is
#: :data:`CALIBRATION_BOUNDS`, which covers every name in either set.
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

#: Definitions GSFLOW calibrates against that are NOT sourced from the central
#: catalogue, keyed by SERVED (unprefixed) name.
#:
#: ``K`` is the only entry, and it is a known, deliberate divergence rather
#: than a second source of truth by accident:
#:
#: * Central ``gsflow_K`` is ``(0.001, 100.0, 'm/d', log)``. It has to stay
#:   central because its served name ``K`` collides with Xinanjiang's.
#: * What every GSFLOW calibration has actually used is ``(0.1, 5000.0,
#:   linear)`` -- justified in-place as "Iceland basalt: 1e2-1e4 m/d", i.e.
#:   the domain this model is run on. The central range CAPS at 100 m/d and
#:   therefore excludes most of that interval.
#:
#: The in-use value is preserved here verbatim; converging the two requires
#: editing the central definition (owned by
#: ``symfluence.core.calibration.parameters``) AND regenerating
#: ``tests/unit/core/data/model_bounds_snapshot.json``, so it is reported
#: rather than changed as a side effect. Do not add entries here without the
#: same justification -- a package-local definition is exactly the ``fuse_MBASE``
#: failure mode when it is not deliberate.
LOCAL_ONLY: Dict[str, ParameterInfo] = {
    'K': ParameterInfo(
        0.1, 5000.0, 'm/d',
        'Hydraulic conductivity (MODFLOW-NWT UPW); Iceland basalt 1e2-1e4 m/d',
        'soil',
    ),
}


def _served(params: Dict[str, ParameterInfo]) -> Dict[str, Dict[str, float]]:
    """Strip the catalogue namespace and flatten to the bounds-dict form."""
    return {
        (name[len(STRIP_PREFIX):] if name.startswith(STRIP_PREFIX) else name): {
            'min': info.min, 'max': info.max, 'transform': info.transform,
        }
        for name, info in params.items()
    }


#: The bounds ``GSFLOWParameterManager`` resolves, keyed by served name.
#:
#: Every number comes from :data:`PARAMS` (which is also what this package
#: contributes to the central catalogue) plus :data:`LOCAL_ONLY`, so a GSFLOW
#: bound change is a one-line edit in one file. Re-exported as
#: ``symfluence.models.gsflow.parameters.PARAM_BOUNDS`` for back-compat.
CALIBRATION_BOUNDS: Dict[str, Dict[str, float]] = {
    **_served(PARAMS),
    **_served(LOCAL_ONLY),
}


def register_bounds() -> None:
    """Contribute GSFLOW's bounds to the central catalogue."""
    register_model_bounds(
        'GSFLOW',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
