# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Unit tests for EvaluationMetricsTracker's standardized progress schema.

Schema under test (fixed — every field always present)::

    [P##] {ALG} {i}/{max} {unit} ({pct}%) | Best: {score} | \
Improved: {a}/{b} | Crashes: {c}/{d} | Elapsed: {t}
"""
from __future__ import annotations

import logging
import re

import pytest

from symfluence.core.calibration.optimizers.metrics_tracker import EvaluationMetricsTracker

LOGGER_NAME = 'symfluence.test.metrics_tracker'

#: Full-schema regex: optional [P##] tag, then the five fixed fields.
SCHEMA_RE = re.compile(
    r'^(?:\[[^\]]+\] )?'
    r'(?P<alg>\S+) (?P<i>\d+)/(?P<max>\d+) (?P<unit>evals|gens|epochs|loops) '
    r'\((?P<pct>\d+)%\) \| '
    r'Best: (?P<best>[^|]+) \| '
    r'(?:[^|]+ \| )?'  # optional secondary objective
    r'Improved: (?P<a>[\d-]+)/(?P<b>[\d-]+) \| '
    r'Crashes: (?P<c>\d+)/(?P<d>\d+) \| '
    r'Elapsed: (?P<t>.+)$'
)


@pytest.fixture
def tracker_logger():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    return logger


def make_tracker(tracker_logger, max_iterations=100, **kwargs):
    return EvaluationMetricsTracker(
        max_iterations, tracker_logger, lambda: '00:01:23', **kwargs
    )


def progress_records(caplog):
    return [r.message for r in caplog.records if 'Elapsed:' in r.message]


class TestFixedSchema:
    """All fields are always present, in fixed order."""

    def test_full_schema_with_all_fields(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=1000)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress(
                'DDS', 20, 0.7231, n_improved=3, population_size=20,
                crash_stats={'crash_count': 0, 'total_evaluations': 20, 'crash_rate': 0.0},
                unit='evals'
            )
        lines = progress_records(caplog)
        assert len(lines) == 1
        assert lines[0] == (
            'DDS 20/1000 evals (2%) | Best: 0.7231 | Improved: 3/20 | '
            'Crashes: 0/20 | Elapsed: 00:01:23'
        )
        assert SCHEMA_RE.match(lines[0])

    def test_fields_present_even_when_counts_missing(self, caplog, tracker_logger):
        """Improved and Crashes render placeholders/zeroes, never disappear."""
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('SCE-UA', 10, -0.5)
        line = progress_records(caplog)[0]
        assert 'Improved: -/- |' in line
        assert 'Crashes: 0/0 |' in line
        assert SCHEMA_RE.match(line)

    def test_crash_field_defaults_to_internal_counters(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100)
        tracker.track_evaluation(0.5)
        tracker.track_evaluation(EvaluationMetricsTracker.PENALTY_SCORE)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('DDS', 10, 0.5, unit='evals')
        assert 'Crashes: 1/2 |' in progress_records(caplog)[0]

    def test_secondary_objective_is_inserted_after_best(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress(
                'NSGA-II', 10, 0.8, secondary_score=0.6, secondary_label='NSE',
                n_improved=4, population_size=30, unit='gens'
            )
        line = progress_records(caplog)[0]
        assert '| Best: 0.8000 | NSE: 0.6000 | Improved: 4/30 |' in line
        assert SCHEMA_RE.match(line)


class TestUnitRendering:
    def test_each_valid_unit_renders(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            for unit in EvaluationMetricsTracker.VALID_UNITS:
                tracker.log_iteration_progress('ALG', 10, 1.0, unit=unit)
        lines = progress_records(caplog)
        assert [line.split()[2] for line in lines] == list(EvaluationMetricsTracker.VALID_UNITS)

    def test_invalid_unit_falls_back_to_evals(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('ALG', 10, 1.0, unit='bananas')
        assert 'ALG 10/50 evals' in progress_records(caplog)[0]

    def test_total_override_changes_denominator_and_pct(self, caplog, tracker_logger):
        """ADAM-style: epochs come from ADAM_STEPS, not NUMBER_OF_ITERATIONS."""
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress(
                'ADAM', 50, 0.9, unit='epochs', total=500,
                n_improved=12, population_size=50
            )
        assert 'ADAM 50/500 epochs (10%)' in progress_records(caplog)[0]


class TestWorkerTag:
    def test_constructor_tag_prefixes_line(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50, worker_tag='P02')
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('CMA-ES', 10, 0.68, n_improved=12,
                                           population_size=20, unit='gens')
        line = progress_records(caplog)[0]
        assert line.startswith('[P02] CMA-ES 10/50 gens (20%)')
        assert SCHEMA_RE.match(line)

    def test_tag_applies_to_every_line(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50, worker_tag='P02')
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('DDS', 10, 0.5)
            tracker.log_iteration_progress('DDS', 20, 0.6)
        assert all(line.startswith('[P02] ') for line in progress_records(caplog))

    def test_no_tag_no_prefix(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('DDS', 10, 0.5)
        assert not progress_records(caplog)[0].startswith('[')

    def test_initial_population_carries_tag(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=50, worker_tag='P03')
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_initial_population('PSO', 30, 0.42)
        assert caplog.records[0].message == (
            '[P03] PSO initial population (30 individuals) complete | Best score: 0.4200'
        )


class TestThrottle:
    def test_default_interval_suppresses_off_multiples(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            for i in range(1, 26):
                tracker.log_iteration_progress('DDS', i, 0.5, unit='evals')
        lines = progress_records(caplog)
        assert [int(line.split()[1].split('/')[0]) for line in lines] == [10, 20]

    def test_final_iteration_always_emitted(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=25)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('DDS', 25, 0.5, unit='evals')
        assert len(progress_records(caplog)) == 1

    def test_custom_interval(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100, log_interval=5)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            for i in range(1, 11):
                tracker.log_iteration_progress('PSO', i, 0.5, unit='evals')
        lines = progress_records(caplog)
        assert [int(line.split()[1].split('/')[0]) for line in lines] == [5, 10]

    def test_force_bypasses_throttle(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('DDS', 3, 0.5, force=True)
        assert len(progress_records(caplog)) == 1

    def test_total_override_is_used_for_final_iteration(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            tracker.log_iteration_progress('ADAM', 37, 0.5, unit='epochs', total=37)
        assert len(progress_records(caplog)) == 1


class TestCrashTracking:
    def test_high_crash_rate_warning(self, caplog, tracker_logger):
        tracker = make_tracker(tracker_logger, max_iterations=100)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            for i in range(50):
                score = EvaluationMetricsTracker.PENALTY_SCORE if i < 10 else 0.5
                tracker.track_evaluation(score)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'High crash rate: 10/50' in warnings[0].message

    def test_get_crash_stats(self, tracker_logger):
        tracker = make_tracker(tracker_logger)
        tracker.track_evaluation(0.9)
        tracker.track_evaluation(EvaluationMetricsTracker.PENALTY_SCORE)
        stats = tracker.get_crash_stats()
        assert stats == {
            'crash_count': 1, 'total_evaluations': 2, 'crash_rate': 0.5
        }
