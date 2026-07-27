# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NOAH-MP calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing NOAHMP's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters NOAHMP calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  NOAHMP resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for NOAHMP: ``ZREF``, ``bexp``, ``dksat``, ``noah_czil``, ``psisat``, ``rain_snow_thresh``, ``refkdt``, ``slope``, ``smcmax``, ``smcref``, ``smcwlt``.

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


#: Tier B -- definitions only NOAHMP resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'route_k': ParameterInfo(1.0, 40.0, 'days', 'Nash-cascade routing time constant for post-model runoff routing (column LSMs emit unrouted runoff)', 'routing'),
}

#: Tier A -- the catalogue names composing NOAHMP's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'slope',
    'dksat',
    'psisat',
    'bexp',
    'smcmax',
    'smcwlt',
    'smcref',
    'refkdt',
    'noah_czil',
    'rain_snow_thresh',
    'ZREF',
    'route_k',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute NOAHMP's bounds to the central catalogue."""
    register_model_bounds(
        'NOAHMP',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
