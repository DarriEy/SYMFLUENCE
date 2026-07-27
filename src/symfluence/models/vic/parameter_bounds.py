# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""VIC (Variable Infiltration Capacity) calibration parameter bounds -- owned by this package.

Service decomposition, item 2: a model must be able to change its own
calibration bounds without a ``core`` release, so this package owns

* **tier A** -- :data:`BOUND_SET`, the catalogue names composing VIC's
  bound set (plus :data:`STRIP_PREFIX`). Which parameters VIC calibrates is
  model identity, not shared physics.
* **tier B** -- :data:`PARAMS`, the ``ParameterInfo`` definitions that only
  VIC resolves.

Parameters shared with another model stay in
``symfluence.core.calibration.parameters.parameter_bounds_registry`` and are
composed here **by name only** -- never redefined. Duplicating one locally is
the ``fuse_MBASE`` / Snow-17 ``MBASE`` failure mode fixed in #368;
``register_model_bounds()`` keeps the central definition and records the
disagreement in ``bounds_registration_conflicts()``.

Every VIC parameter is solo, so the whole set lives here.

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


#: Tier B -- definitions only VIC resolves.
PARAMS: Dict[str, ParameterInfo] = {
    'infilt': ParameterInfo(0.001, 0.9, '-', 'Variable infiltration curve parameter', 'soil'),
    'Ds': ParameterInfo(0.0, 1.0, '-', 'Fraction of Dsmax where nonlinear baseflow begins', 'baseflow'),
    'Dsmax': ParameterInfo(0.1, 30.0, 'mm/day', 'Maximum baseflow velocity', 'baseflow'),
    'Ws': ParameterInfo(0.1, 1.0, '-', 'Fraction of max soil moisture for nonlinear baseflow', 'baseflow'),
    'c': ParameterInfo(1.0, 4.0, '-', 'Exponent in baseflow curve', 'baseflow'),
    'depth1': ParameterInfo(0.05, 0.5, 'm', 'Soil layer 1 depth', 'soil'),
    'depth2': ParameterInfo(0.1, 1.5, 'm', 'Soil layer 2 depth', 'soil'),
    'depth3': ParameterInfo(0.1, 2.0, 'm', 'Soil layer 3 depth', 'soil'),
    'Ksat_vic': ParameterInfo(1.0, 5000.0, 'mm/day', 'VIC saturated hydraulic conductivity', 'soil'),
    'expt_vic': ParameterInfo(4.0, 30.0, '-', 'VIC soil layer exponent', 'soil'),
    'bulk_density': ParameterInfo(1200.0, 1800.0, 'kg/m³', 'Soil bulk density', 'soil'),
    'snow_rough': ParameterInfo(0.0001, 0.01, 'm', 'Snow surface roughness', 'snow'),
}

#: Tier A -- the catalogue names composing VIC's bound set, in served order.
BOUND_SET: List[str] = [
    'infilt',
    'Ds',
    'Dsmax',
    'Ws',
    'c',
    'depth1',
    'depth2',
    'depth3',
    'Ksat_vic',
    'expt_vic',
    'bulk_density',
    'snow_rough',
]

#: Catalogue keys are already the served names.
STRIP_PREFIX = ''


def register_bounds() -> None:
    """Contribute VIC's bounds to the central catalogue."""
    register_model_bounds(
        'VIC',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
