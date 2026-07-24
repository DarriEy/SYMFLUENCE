# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests for PopulationEvaluator log de-duplication.

A broken model/forcing setup makes every evaluation fail with the same
exception. The evaluator must log the first occurrence at ERROR (with the
worker traceback when available) and demote identical repeats to DEBUG,
plus emit one actionable ERROR when an entire batch is penalty-only.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from symfluence.core.calibration.optimizers.evaluators.population_evaluator import (
    PopulationEvaluator,
)


@pytest.fixture
def evaluator():
    """PopulationEvaluator with mocked collaborators (extraction-only tests)."""
    return PopulationEvaluator(
        task_builder=MagicMock(),
        worker=MagicMock(),
        execute_batch=MagicMock(),
        use_parallel=True,
        num_processes=2,
        model_name='SUMMA',
        logger=logging.getLogger('test_population_evaluator'),
    )


PENALTY = PopulationEvaluator.DEFAULT_PENALTY_SCORE


def _error_result(idx, message, score=None, traceback=None):
    result = {'individual_id': idx, 'score': score, 'error': message}
    if traceback is not None:
        result['traceback'] = traceback
    return result


class TestTaskErrorDedup:
    """Identical worker exceptions must not repeat at ERROR/WARNING."""

    def test_first_occurrence_error_then_debug(self, evaluator, caplog):
        results = [
            _error_result(i, "AttributeError: 'NoneType' has no attribute 'sim'")
            for i in range(4)
        ]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores(results, 4)

        error_records = [
            r for r in caplog.records
            if 'worker error' in r.message and r.levelno == logging.ERROR
        ]
        debug_repeats = [
            r for r in caplog.records
            if 'worker error' in r.message and r.levelno == logging.DEBUG
        ]
        assert len(error_records) == 1
        assert len(debug_repeats) == 3
        # Repeats carry an occurrence counter
        assert any('repeat #4' in r.message for r in debug_repeats)

    def test_distinct_errors_each_logged_at_error(self, evaluator, caplog):
        results = [
            _error_result(0, 'ValueError: bad forcing'),
            _error_result(1, 'OSError: disk full'),
        ]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores(results, 2)

        error_records = [
            r for r in caplog.records
            if 'worker error' in r.message and r.levelno == logging.ERROR
        ]
        assert len(error_records) == 2

    def test_first_occurrence_includes_traceback_when_available(
        self, evaluator, caplog
    ):
        results = [
            _error_result(0, 'ValueError: boom', traceback='Traceback (most recent call last):\n  ...'),
        ]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores(results, 1)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any('Traceback (most recent call last)' in r.message for r in error_records)

    def test_dedup_shared_across_extract_methods(self, evaluator, caplog):
        """The same exception seen via objectives extraction is also a repeat."""
        message = 'RuntimeError: model exploded'
        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores([_error_result(0, message)], 1)
            evaluator._extract_objectives(
                [_error_result(0, message, score=None)], 1, 2,
                objective_names=['KGE', 'NSE'],
            )

        error_records = [
            r for r in caplog.records
            if 'worker error' in r.message and r.levelno == logging.ERROR
        ]
        assert len(error_records) == 1


class TestAllPenaltyBatchSummary:
    """A penalty-only batch emits one actionable ERROR, then goes quiet."""

    def test_all_penalty_scores_emits_single_actionable_error(
        self, evaluator, caplog
    ):
        results = [
            {'individual_id': i, 'score': PENALTY, 'error': None} for i in range(3)
        ]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores(results, 3)

        summary = [
            r for r in caplog.records
            if 'penalty' in r.message and r.levelno == logging.ERROR
        ]
        assert len(summary) == 1
        # Actionable wording: what happened, likely cause, what to do
        assert 'broken model or forcing setup' in summary[0].message
        assert 'verify the model runs outside calibration' in summary[0].message

    def test_all_penalty_summary_demoted_on_repeat_batches(self, evaluator, caplog):
        results = [{'individual_id': 0, 'score': PENALTY, 'error': None}]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_scores(results, 1)
            evaluator._extract_scores(results, 1)

        summary_error = [
            r for r in caplog.records
            if 'broken model or forcing setup' in r.message and r.levelno == logging.ERROR
        ]
        summary_debug = [
            r for r in caplog.records
            if 'broken model or forcing setup' in r.message and r.levelno == logging.DEBUG
        ]
        assert len(summary_error) == 1
        assert len(summary_debug) == 1

    def test_all_penalty_objectives_emits_summary(self, evaluator, caplog):
        results = [
            {'individual_id': i, 'objectives': [PENALTY, PENALTY], 'error': None}
            for i in range(2)
        ]

        with caplog.at_level(logging.DEBUG):
            evaluator._extract_objectives(results, 2, 2, objective_names=['KGE', 'NSE'])

        summary = [
            r for r in caplog.records
            if 'penalty objectives' in r.message and r.levelno == logging.ERROR
        ]
        assert len(summary) == 1

    def test_valid_scores_do_not_trigger_summary(self, evaluator, caplog):
        results = [
            {'individual_id': 0, 'score': 0.7, 'error': None},
            {'individual_id': 1, 'score': PENALTY, 'error': None},
        ]

        with caplog.at_level(logging.DEBUG):
            fitness = evaluator._extract_scores(results, 2)

        assert fitness[0] == 0.7
        assert not any(
            'broken model or forcing setup' in r.message for r in caplog.records
        )
