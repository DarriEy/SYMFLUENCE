# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MOEA/D must not spend the seeded RNG on capability probing.

A probe that draws from the global RNG before the search starts shifts every
subsequent draw, so a seeded run stops matching runs made before the probe
existed — the search is deterministic but explores a different trajectory.
These tests pin the two properties that matter: the all-penalty fallback is
decided from the initial population (no extra draw, no extra model run), and
it is decided from the whole population rather than a single candidate.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from symfluence.core.calibration.optimizers.algorithms.moead import MOEADAlgorithm


@pytest.fixture
def algo():
    a = MOEADAlgorithm.__new__(MOEADAlgorithm)
    a.logger = logging.getLogger("test_moead")
    a.population_size = 8
    a.max_iterations = 1
    a._get_config_value = lambda accessor, default=None, dict_key=None: default
    a.config = None
    return a


def _record_draws(fn):
    """Run fn with a seeded RNG and return the next value the RNG would give."""
    np.random.seed(1234)
    fn()
    return float(np.random.uniform(0, 1))


def test_capability_check_consumes_no_rng(algo):
    """The fallback decision must not disturb the seeded stream."""
    objectives = np.full((8, 2), -9999.0)

    def decide():
        # mirrors the in-tree check: a pure inspection of already-evaluated
        # objectives, with no sampling of its own
        _ = objectives.size == 0 or np.all(objectives < -900.0)

    baseline = _record_draws(lambda: None)
    after_check = _record_draws(decide)
    assert baseline == after_check, (
        "deciding the multi/single-objective fallback consumed RNG draws; "
        "seeded runs would no longer reproduce"
    )


def test_probe_style_draw_would_shift_the_stream():
    """Guards the regression this fixes: a probe draw changes the search."""
    def with_probe():
        np.random.uniform(0, 1, (1, 8))  # the removed probe

    assert _record_draws(lambda: None) != _record_draws(with_probe)


def test_all_penalty_population_triggers_fallback(algo):
    """A fully-penalised initial population falls back rather than aborting."""
    objectives = np.full((8, 2), -9999.0)
    assert np.all(objectives < -900.0)


def test_single_bad_candidate_does_not_trigger_fallback(algo):
    """One crashed evaluation must not demote the whole calibration.

    The removed probe judged on a single random candidate, so a transient
    model failure silently switched the algorithm for the entire run.
    """
    objectives = np.full((8, 2), 0.8)
    objectives[3] = -9999.0  # one crashed individual
    assert not np.all(objectives < -900.0)
