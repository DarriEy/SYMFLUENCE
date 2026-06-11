# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Catchment-area resolution contract for the worker metrics path.

Trial metrics (StreamflowMetrics) and the final evaluation
(StreamflowEvaluator) resolve catchment area independently, and a
disagreement makes the calibration score and the final-evaluation score
incomparable. The contract: FIXED_CATCHMENT_AREA always wins, and the
last-resort default (1 km²) is identical in both paths.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.evaluation.utilities.streamflow_metrics import StreamflowMetrics


@pytest.fixture
def metrics():
    return StreamflowMetrics()


class TestFixedAreaOverride:
    def test_fixed_catchment_area_wins(self, metrics, tmp_path):
        """FIXED_CATCHMENT_AREA (m²) must drive trial metrics too, not just
        the final-evaluation chain."""
        config = {'FIXED_CATCHMENT_AREA': 2.207e9}  # 2207 km² in m²
        area = metrics.get_catchment_area(config, tmp_path, 'nodomain')
        assert area == pytest.approx(2207.0)

    def test_fixed_area_skips_shapefile_lookup(self, metrics, tmp_path):
        """No shapefile exists under tmp_path; the override must not care."""
        config = {'FIXED_CATCHMENT_AREA': 5.0e8}
        area = metrics.get_catchment_area(
            config, tmp_path, 'nodomain', source='shapefile'
        )
        assert area == pytest.approx(500.0)


class TestDefaultConsistency:
    def test_last_resort_default_matches_evaluator(self, metrics, tmp_path, caplog):
        """With no area source at all, the fallback is 1 km² — the same
        last resort the StreamflowEvaluator chain uses — and it warns."""
        with caplog.at_level(logging.WARNING):
            area = metrics.get_catchment_area({}, tmp_path, 'nodomain')
        assert area == 1.0
        assert any('FIXED_CATCHMENT_AREA' in r.message for r in caplog.records)
