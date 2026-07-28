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
``GSFLOWParameterManager._load_parameter_bounds()`` reads at calibration time --
used to be an independent literal dict duplicating these values; it is now
:data:`CALIBRATION_BOUNDS`, derived below. Every number is either owned here
(:data:`PARAMS`) or read back from the central catalogue, never copied.

GSFLOW needs :data:`CALIBRATION_BOUNDS` on top of the common
PARAMS/BOUND_SET/``register_bounds`` shape because its manager calibrates names
outside :data:`BOUND_SET`; ``get_model_bounds('GSFLOW')`` serves the bound set
alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterator, List

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
#: catalogue, keyed by SERVED (unprefixed) name. Empty, and it must stay that
#: way: a package-local definition shadowing a central one is the ``fuse_MBASE``
#: failure mode #368 had to fix, and is how local ``K`` (0.1..5000 linear) came
#: to disagree with central ``gsflow_K`` (0.001..100 log) unnoticed.
LOCAL_ONLY: Dict[str, ParameterInfo] = {}


def _served(params: Dict[str, ParameterInfo]) -> Dict[str, Dict[str, Any]]:
    """Strip the catalogue namespace and flatten to the bounds-dict form.

    Values are mixed by design: ``min``/``max`` are floats and ``transform`` is
    a string, matching what ``get_model_bounds`` serves.
    """
    return {
        (name[len(STRIP_PREFIX):] if name.startswith(STRIP_PREFIX) else name): {
            'min': info.min, 'max': info.max, 'transform': info.transform,
        }
        for name, info in params.items()
    }


def _shared_from_catalogue() -> Dict[str, Dict[str, float]]:
    """Served bounds for the BOUND_SET names this package does not own.

    ``gsflow_K`` is the only one today: its served name ``K`` collides with
    Xinanjiang's, so it has to stay namespaced in the central catalogue. It is
    read back through the public accessor rather than copied, because a local
    copy of a central definition is exactly how the two drifted apart in the
    first place -- the catalogue said 0.001-100 log while calibration actually
    searched 0.1-5000 linear.
    """
    from symfluence.core.calibration.parameters.parameter_bounds_registry import (
        get_registry,
    )

    shared = [name for name in BOUND_SET if name not in PARAMS]
    if not shared:
        return {}
    resolved = get_registry().get_bounds_for_params(shared)
    return {
        (name[len(STRIP_PREFIX):] if name.startswith(STRIP_PREFIX) else name): bounds
        for name, bounds in resolved.items()
    }


class _CalibrationBounds(Mapping):
    """The bounds ``GSFLOWParameterManager`` resolves, keyed by served name.

    Numbers come from :data:`PARAMS` (what this package contributes to the
    catalogue) plus, for the BOUND_SET names owned centrally because they are
    shared, the catalogue definition itself. Nothing is copied, so a GSFLOW
    bound change is a one-line edit in exactly one place.

    A live view rather than a dict built at import time. The shared half is read
    back from the registry, and at *this module's* import neither
    :func:`register_bounds` nor any other package's has necessarily run. It
    happens to work today only because ``gsflow_K`` is a central (tier C) name,
    present before any registration; a shared name contributed by another
    package's tier B would be silently missing from an import-time snapshot.
    :func:`get_model_bounds` resolves at call time for the same reason.
    """

    @staticmethod
    def _table() -> Dict[str, Dict[str, float]]:
        return {
            **_served(PARAMS),
            **_shared_from_catalogue(),
            **_served(LOCAL_ONLY),
        }

    def __getitem__(self, key: str) -> Dict[str, float]:
        return self._table()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())

    def copy(self) -> Dict[str, Dict[str, float]]:
        """A plain-dict snapshot. ``Mapping`` has no ``copy``, and this name is
        re-exported as ``parameters.PARAM_BOUNDS``, which used to be a dict."""
        return self._table()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{type(self).__name__}({self._table()!r})"


#: Re-exported as ``symfluence.models.gsflow.parameters.PARAM_BOUNDS``.
CALIBRATION_BOUNDS: Mapping = _CalibrationBounds()


def register_bounds() -> None:
    """Contribute GSFLOW's bounds to the central catalogue."""
    register_model_bounds(
        'GSFLOW',
        params=PARAMS,
        names=BOUND_SET,
        strip_prefix=STRIP_PREFIX,
    )
