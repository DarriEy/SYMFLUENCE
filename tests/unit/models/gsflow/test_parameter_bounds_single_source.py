# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GSFLOW must resolve its calibration bounds from one place.

GSFLOW used to carry two divergent bound sources: the literal ``PARAM_BOUNDS``
dict in ``models/gsflow/parameters.py`` (what ``GSFLOWParameterManager``
actually reads at calibration time) and the central catalogue reached through
``get_gsflow_bounds()``. ``K`` disagreed between them (0.1..5000 linear vs
0.001..100 log) and four names the manager calibrates by default were absent
from the catalogue's bound set entirely.

These tests pin the convergence: every number now comes from
``models/gsflow/parameter_bounds.py``, and the one value that still cannot
(``K``) is an explicit, enumerated exception rather than an accident.
"""
from __future__ import annotations

import pytest

from symfluence.core.calibration.parameters.parameter_bounds_registry import (
    bounds_registration_conflicts,
    get_gsflow_bounds,
)
from symfluence.models.gsflow import parameter_bounds as pb
from symfluence.models.gsflow.parameters import DEFAULT_PARAMS, PARAM_BOUNDS

pytestmark = [pytest.mark.unit]


# The GSFLOW_PARAMS_TO_CALIBRATE default in GSFLOWParameterManager.
_DEFAULT_CALIBRATED = [
    'soil_moist_max', 'ssr2gw_rate', 'K', 'SY', 'slowcoef_lin', 'carea_max',
    'smidx_coef', 'jh_coef', 'tmax_allsnow', 'rain_adj', 'snow_adj',
]

# The bounds every GSFLOW calibration has used to date. Preserved verbatim by
# the convergence: the structure changed, the numbers did not.
_IN_USE = {
    'soil_moist_max': (1.0, 15.0),
    'ssr2gw_rate': (0.001, 0.5),
    'K': (0.1, 5000.0),
    'SY': (0.01, 0.4),
    'slowcoef_lin': (0.001, 0.5),
    'carea_max': (0.1, 1.0),
    'smidx_coef': (0.001, 0.10),
    'jh_coef': (0.005, 0.030),
    'tmax_allsnow': (-3.0, 2.0),
    'rain_adj': (0.5, 2.0),
    'snow_adj': (0.5, 2.0),
}


def test_param_bounds_is_the_package_definition_not_a_copy():
    """``parameters.PARAM_BOUNDS`` must be the derived view, not a second dict."""
    assert PARAM_BOUNDS is pb.CALIBRATION_BOUNDS


@pytest.mark.parametrize("name", _DEFAULT_CALIBRATED)
def test_default_calibrated_bounds_are_unchanged(name):
    """Converging the sources must not move any bound a GSFLOW run uses."""
    got = PARAM_BOUNDS[name]
    assert (got['min'], got['max']) == _IN_USE[name]
    assert got.get('transform', 'linear') == 'linear'


def test_every_manager_parameter_has_a_real_bound():
    """No default-calibrated name may fall through to the 0.001..10 fallback.

    This is the half of the divergence that is fixed outright: ``jh_coef``,
    ``tmax_allsnow``, ``rain_adj`` and ``snow_adj`` are calibrated by default,
    and opting into ``soil_rechr_max`` / ``gwflow_coef`` / ``gw_seep_coef`` /
    ``tmax_allrain`` through ``GSFLOW_PARAMS_TO_CALIBRATE`` no longer silently
    yields a placeholder range.
    """
    optional = ['soil_rechr_max', 'gwflow_coef', 'gw_seep_coef', 'tmax_allrain']
    for name in _DEFAULT_CALIBRATED + optional:
        assert name in PARAM_BOUNDS, f"{name} would fall back to a placeholder bound"
        assert PARAM_BOUNDS[name]['min'] < PARAM_BOUNDS[name]['max']


def test_defaults_lie_inside_their_bounds():
    for name, value in DEFAULT_PARAMS.items():
        bounds = PARAM_BOUNDS[name]
        assert bounds['min'] <= value <= bounds['max'], name


def test_nothing_diverges_from_the_central_catalogue():
    """Both GSFLOW bound paths must resolve to the in-use definitions.

    ``K`` used to be the exception: the central ``gsflow_K`` said 0.001..100
    (log) while GSFLOW was actually calibrated over 0.1..5000 (linear; Iceland
    basalt is 1e2-1e4 m/d), and the catalogue entry was inert because nothing
    calls ``get_gsflow_bounds()`` — which is how the two were free to drift.
    The central definition now carries the in-use range, so there is one source
    again.

    The two paths are compared against ``_IN_USE`` rather than only against
    each other. Comparing them to each other alone cannot detect the very
    drift this test is named for: ``CALIBRATION_BOUNDS`` is *derived* from the
    catalogue for exactly the shared names ``get_gsflow_bounds()`` resolves, so
    both sides move together — changing central ``gsflow_K`` left this green
    while the pinned ``_IN_USE`` table went red. ``_IN_USE`` is the independent
    third party: the numbers every GSFLOW calibration to date has searched.
    """
    assert set(pb.LOCAL_ONLY) == set()

    catalogue = get_gsflow_bounds()

    # The catalogue path must serve the in-use numbers. This is what has teeth
    # against a central-definition change; ``K`` is the name it was written for.
    for name in sorted(set(catalogue) & set(_IN_USE)):
        assert (catalogue[name]['min'], catalogue[name]['max']) == _IN_USE[name], (
            f"get_gsflow_bounds() serves {name}="
            f"({catalogue[name]['min']}, {catalogue[name]['max']}), but GSFLOW "
            f"calibrates over {_IN_USE[name]}. The central catalogue and the "
            "in-use range have drifted apart again."
        )
    assert 'K' in catalogue, (
        "K left GSFLOW's catalogue bound set — it is the name the two sources "
        "drifted on, so it must stay covered here"
    )

    # And the manager path must agree with the catalogue path name for name, so
    # a package-local definition cannot shadow a central one (the ``fuse_MBASE``
    # failure mode #368 had to fix).
    shared = set(catalogue) & set(PARAM_BOUNDS)
    diverged = {n for n in shared if catalogue[n] != PARAM_BOUNDS[n]}
    assert diverged == set(), (
        f"GSFLOW bound sources diverged on {sorted(diverged)} — every shared "
        "name must resolve to the same definition through both paths"
    )


def test_package_contribution_is_accepted_by_the_catalogue():
    """No tier-B definition may be silently overridden by a central one."""
    conflicts = [c for c in bounds_registration_conflicts() if 'GSFLOW' in c]
    assert conflicts == []
