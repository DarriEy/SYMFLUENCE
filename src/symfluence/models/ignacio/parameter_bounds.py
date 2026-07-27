# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""IGNACIO (Canadian FBP fire spread) calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing IGNACIO's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters IGNACIO calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  IGNACIO resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Every IGNACIO parameter is solo, so the whole set lives here.

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


#: Tier B -- definitions only IGNACIO resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'ffmc': ParameterInfo(0.0, 101.0, '-', 'Fine Fuel Moisture Code', 'fire'),
    'dmc': ParameterInfo(0.0, 200.0, '-', 'Duff Moisture Code', 'fire'),
    'dc': ParameterInfo(0.0, 800.0, '-', 'Drought Code', 'fire'),
    'fmc': ParameterInfo(50.0, 150.0, '%', 'Foliar Moisture Content', 'fire'),
    'curing': ParameterInfo(0.0, 100.0, '%', 'Grass curing percentage', 'fire'),
    'initial_radius': ParameterInfo(1.0, 100.0, 'm', 'Initial fire radius', 'fire'),
}

#: Tier A -- the catalogue names composing IGNACIO's bound set, in served order.
BOUND_SET: List[str] = [
    'ffmc',
    'dmc',
    'dc',
    'fmc',
    'curing',
    'initial_radius',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute IGNACIO's bounds to the central catalogue."""
    register_model_bounds(
        'IGNACIO',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
