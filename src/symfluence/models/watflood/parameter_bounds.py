# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""WATFLOOD calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing WATFLOOD's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters WATFLOOD calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  WATFLOOD resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for WATFLOOD: ``watflood_PWR``, ``watflood_R2N``.

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


#: Tier B -- definitions only WATFLOOD resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'watflood_FLZCOEF': ParameterInfo(1e-06, 0.01, '-', 'Lower zone function coefficient', 'baseflow', 'log'),
    'watflood_AK': ParameterInfo(1.0, 100.0, 'mm/h', 'Upper zone interflow coefficient', 'baseflow'),
    'watflood_AKF': ParameterInfo(1.0, 100.0, 'mm/h', 'Interflow recession coefficient', 'baseflow'),
    'watflood_REESSION': ParameterInfo(0.01, 1.0, '-', 'Baseflow recession coefficient', 'baseflow'),
    'watflood_RETN': ParameterInfo(10.0, 500.0, 'h', 'Retention constant', 'routing'),
    'watflood_AK2': ParameterInfo(0.001, 1.0, '-', 'Lower zone depletion coefficient', 'baseflow', 'log'),
    'watflood_AK2FS': ParameterInfo(0.001, 1.0, '-', 'Lower zone depletion (snow-covered)', 'baseflow', 'log'),
    'watflood_R3': ParameterInfo(1.0, 100.0, '-', 'Overbank roughness multiplier', 'routing'),
    'watflood_DS': ParameterInfo(0.0, 20.0, 'mm', 'Surface depression storage', 'soil'),
    'watflood_FPET': ParameterInfo(0.5, 5.0, '-', 'PET adjustment factor', 'et'),
    'watflood_FTALL': ParameterInfo(0.01, 1.0, '-', 'Forest canopy adjustment', 'et'),
    'watflood_FM': ParameterInfo(0.01, 0.5, 'mm/degC/h', 'Melt factor', 'snow'),
    'watflood_BASE': ParameterInfo(-3.0, 2.0, 'degC', 'Base temperature for melt', 'snow'),
    'watflood_SUBLIM_FACTOR': ParameterInfo(0.0, 0.5, '-', 'Sublimation fraction', 'snow'),
}

#: Tier A -- the catalogue names composing WATFLOOD's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'watflood_FLZCOEF',
    'watflood_PWR',
    'watflood_R2N',
    'watflood_AK',
    'watflood_AKF',
    'watflood_REESSION',
    'watflood_RETN',
    'watflood_AK2',
    'watflood_AK2FS',
    'watflood_R3',
    'watflood_DS',
    'watflood_FPET',
    'watflood_FTALL',
    'watflood_FM',
    'watflood_BASE',
    'watflood_SUBLIM_FACTOR',
]

#: Catalogue keys are namespaced; parameter managers use unprefixed names.
STRIP_PREFIX = 'watflood_'


def register_bounds() -> None:
    """Contribute WATFLOOD's bounds to the central catalogue."""
    register_model_bounds(
        'WATFLOOD',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
