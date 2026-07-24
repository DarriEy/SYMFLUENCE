# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
from __future__ import annotations

"""
Regression tests for crash counting on the population evaluation paths.

`track_evaluation` was only called from the single-solution path, so every
population-based algorithm (NSGA-II, MOEA/D, DE, PSO, GA, ...) reported
"Crashes: 0/0" no matter how many individuals actually crashed. A 16-hour
NSGA-II run logged `Crashes: 0/0` while 66 SUMMA simulations were failing.
"""

import logging

import pytest

from symfluence.core.calibration.optimizers.evaluators.population_evaluator import (
    PopulationEvaluator,
)
from symfluence.core.calibration.optimizers.metrics_tracker import EvaluationMetricsTracker
from symfluence.core.constants import ModelDefaults

PENALTY = ModelDefaults.PENALTY_SCORE


@pytest.fixture
def logger():
    log = logging.getLogger('test-population-crash')
    log.addHandler(logging.NullHandler())
    return log


@pytest.fixture
def tracker(logger):
    return EvaluationMetricsTracker(
        max_iterations=10, logger=logger, elapsed_time_fn=lambda: '0s')


def _evaluator(logger, tracker):
    ev = PopulationEvaluator.__new__(PopulationEvaluator)
    ev.logger = logger
    ev.metrics_tracker = tracker
    ev._task_error_counts = {}
    return ev


def test_single_objective_batch_counts_crashes(logger, tracker):
    ev = _evaluator(logger, tracker)
    results = [
        {'individual_id': 0, 'score': 0.8},
        {'individual_id': 1, 'score': PENALTY},   # crashed
        {'individual_id': 2, 'score': 0.7},
        {'individual_id': 3, 'score': None},      # returned nothing
        {'individual_id': 4, 'score': 0.6},
    ]

    ev._extract_scores(results, 5)

    stats = tracker.get_crash_stats()
    assert stats['crash_count'] == 2
    assert stats['total_evaluations'] == 5


def test_multi_objective_batch_counts_crashes(logger, tracker):
    ev = _evaluator(logger, tracker)
    results = [
        {'individual_id': 0, 'objectives': [0.9, 0.8]},
        {'individual_id': 1, 'objectives': [PENALTY, PENALTY]},  # crashed
        {'individual_id': 2, 'objectives': [0.5, 0.4]},
        {'individual_id': 3, 'objectives': None},                # returned nothing
    ]

    ev._extract_objectives(results, 4, 2)

    stats = tracker.get_crash_stats()
    assert stats['crash_count'] == 2
    assert stats['total_evaluations'] == 4


def test_counts_accumulate_across_generations(logger, tracker):
    """A population algorithm evaluates many batches; totals must accumulate."""
    ev = _evaluator(logger, tracker)
    batch = [
        {'individual_id': 0, 'score': 0.8},
        {'individual_id': 1, 'score': PENALTY},
    ]

    for _ in range(5):
        ev._extract_scores(batch, 2)

    stats = tracker.get_crash_stats()
    assert stats['crash_count'] == 5
    assert stats['total_evaluations'] == 10
    assert stats['crash_rate'] == pytest.approx(0.5)


def test_all_successful_batch_reports_no_crashes(logger, tracker):
    ev = _evaluator(logger, tracker)
    results = [{'individual_id': i, 'score': 0.5 + i / 10} for i in range(4)]

    ev._extract_scores(results, 4)

    stats = tracker.get_crash_stats()
    assert stats['crash_count'] == 0
    assert stats['total_evaluations'] == 4


def test_evaluator_without_tracker_still_works(logger):
    """The tracker is optional — omitting it must not break evaluation."""
    ev = _evaluator(logger, tracker=None)
    results = [{'individual_id': 0, 'score': 0.8}, {'individual_id': 1, 'score': PENALTY}]

    scores = ev._extract_scores(results, 2)

    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == pytest.approx(PENALTY)
