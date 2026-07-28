# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE (Framework for Understanding Structural Errors) calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing FUSE's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters FUSE calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  FUSE resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Stays central for FUSE: ``PXTEMP``, ``fuse_MBASE``, ``fuse_MFMAX``, ``fuse_MFMIN``.

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

#: Tier B -- definitions only FUSE resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'LAPSE': ParameterInfo(3.0, 10.0, '°C/km', 'Temperature lapse rate', 'snow'),
    'MAXWATR_1': ParameterInfo(50.0, 1000.0, 'mm', 'Maximum storage upper layer', 'soil'),
    'MAXWATR_2': ParameterInfo(100.0, 2000.0, 'mm', 'Maximum storage lower layer', 'soil'),
    'FRACTEN': ParameterInfo(0.1, 0.9, '-', 'Fraction tension storage', 'soil'),
    'PERCRTE': ParameterInfo(0.01, 100.0, 'mm/day', 'Percolation rate', 'soil'),
    'PERCEXP': ParameterInfo(1.0, 20.0, '-', 'Percolation exponent', 'soil'),
    'BASERTE': ParameterInfo(0.001, 1.0, 'mm/day', 'Baseflow rate', 'baseflow'),
    'QB_POWR': ParameterInfo(1.0, 10.0, '-', 'Baseflow exponent', 'baseflow'),
    'QBRATE_2A': ParameterInfo(0.001, 0.1, '1/day', 'Primary baseflow depletion', 'baseflow'),
    'QBRATE_2B': ParameterInfo(0.0001, 0.01, '1/day', 'Secondary baseflow depletion', 'baseflow'),
    'TIMEDELAY': ParameterInfo(0.0, 10.0, 'days', 'Time delay in routing', 'routing'),
    'RTFRAC1': ParameterInfo(0.1, 0.9, '-', 'Fraction roots upper layer', 'et'),
    'RTFRAC2': ParameterInfo(0.1, 0.9, '-', 'Fraction roots lower layer', 'et'),
}

#: Tier A -- the catalogue names composing FUSE's bound set, in served order.
#: Names absent from :data:`PARAMS` are shared and defined centrally.
BOUND_SET: List[str] = [
    'fuse_MBASE',
    'fuse_MFMAX',
    'fuse_MFMIN',
    'PXTEMP',
    'LAPSE',
    'MAXWATR_1',
    'MAXWATR_2',
    'FRACTEN',
    'PERCRTE',
    'PERCEXP',
    'BASERTE',
    'QB_POWR',
    'QBRATE_2A',
    'QBRATE_2B',
    'TIMEDELAY',
    'RTFRAC1',
    'RTFRAC2',
]

#: Catalogue keys are namespaced; parameter managers use unprefixed names.
STRIP_PREFIX = 'fuse_'


def register_bounds() -> None:
    """Contribute FUSE's bounds to the central catalogue."""
    register_model_bounds(
        'FUSE',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
