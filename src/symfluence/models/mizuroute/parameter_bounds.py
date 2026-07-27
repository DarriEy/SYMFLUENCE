# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mizuRoute calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing MIZUROUTE's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters MIZUROUTE calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  MIZUROUTE resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Every MIZUROUTE parameter is solo, so the whole set lives here.

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


#: Tier B -- definitions only MIZUROUTE resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'velo': ParameterInfo(0.1, 5.0, 'm/s', 'Flow velocity', 'routing'),
    'diff': ParameterInfo(100.0, 5000.0, 'm²/s', 'Diffusion coefficient', 'routing'),
    'mann_n': ParameterInfo(0.01, 0.1, '-', 'Manning roughness coefficient', 'routing'),
    'wscale': ParameterInfo(0.0001, 0.01, '-', 'Width scale parameter', 'routing'),
    'fshape': ParameterInfo(1.0, 5.0, '-', 'Shape parameter', 'routing'),
    'tscale': ParameterInfo(3600, 172800, 's', 'Time scale parameter', 'routing'),
}

#: Tier A -- the catalogue names composing MIZUROUTE's bound set, in served order.
BOUND_SET: List[str] = [
    'velo',
    'diff',
    'mann_n',
    'wscale',
    'fshape',
    'tscale',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute MIZUROUTE's bounds to the central catalogue."""
    register_model_bounds(
        'MIZUROUTE',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
