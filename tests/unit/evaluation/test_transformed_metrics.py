# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for config-selectable transformed calibration metrics.

Covers the Box-Cox (lambda=0.2) KGE variant exposed as ``KGE_BOX_COX``:
the TRANSFORMED_METRICS mapping in StreamflowMetrics, optimization-direction
inference in MetricTransformer, and acceptance by the OptimizationConfig
metric validator.
"""
from __future__ import annotations

import numpy as np
import pytest

from symfluence.core.config.models.optimization import OptimizationConfig
from symfluence.evaluation.metric_transformer import MetricTransformer
from symfluence.evaluation.utilities.streamflow_metrics import StreamflowMetrics


@pytest.fixture
def flows():
    rng = np.random.default_rng(42)
    obs = np.abs(rng.gamma(2.0, 5.0, size=365)) + 0.1
    sim = obs * rng.uniform(0.8, 1.2, size=365)
    return obs, sim


class TestKgeBoxCox:
    def test_kge_box_cox_is_computed(self, flows):
        obs, sim = flows
        result = StreamflowMetrics().calculate_metrics(obs, sim, metrics=['kge_box_cox'])
        assert 'kge_box_cox' in result
        assert np.isfinite(result['kge_box_cox'])

    def test_uppercase_alias_matches_lowercase(self, flows):
        obs, sim = flows
        sm = StreamflowMetrics()
        lower = sm.calculate_metrics(obs, sim, metrics=['kge_box_cox'])['kge_box_cox']
        upper = sm.calculate_metrics(obs, sim, metrics=['KGE_BOX_COX'])['KGE_BOX_COX']
        assert lower == pytest.approx(upper)

    def test_perfect_simulation_scores_one(self, flows):
        obs, _ = flows
        result = StreamflowMetrics().calculate_metrics(obs, obs.copy(), metrics=['kge_box_cox'])
        assert result['kge_box_cox'] == pytest.approx(1.0)

    def test_differs_from_untransformed_kge(self, flows):
        obs, sim = flows
        result = StreamflowMetrics().calculate_metrics(obs, sim, metrics=['kge', 'kge_box_cox'])
        assert result['kge'] != pytest.approx(result['kge_box_cox'])

    def test_matches_manual_box_cox_transform(self, flows):
        obs, sim = flows
        result = StreamflowMetrics().calculate_metrics(obs, sim, metrics=['kge_box_cox'])

        from symfluence.evaluation.metrics_core import kge
        epsilon = max(np.mean(obs) * 0.01, 1e-6)
        lam = 0.2
        obs_t = ((obs + epsilon) ** lam - 1) / lam
        sim_t = ((sim + epsilon) ** lam - 1) / lam
        assert result['kge_box_cox'] == pytest.approx(float(kge(obs_t, sim_t, transfo=1)))


class TestBoxCoxDirection:
    def test_kge_box_cox_maximized(self):
        assert MetricTransformer.get_direction('KGE_BOX_COX') == 'maximize'
        assert MetricTransformer.get_direction('kge_box_cox') == 'maximize'

    def test_box_cox_suffix_preserves_minimize_direction(self):
        # Suffix stripping must find the base metric; without it RMSE_BOX_COX
        # would fall through to the maximize default.
        assert MetricTransformer.get_direction('RMSE_BOX_COX') == 'minimize'


class TestConfigAcceptsTransformedMetrics:
    @pytest.mark.parametrize('name', [
        'KGE_BOX_COX', 'kge_box_cox',
        'KGE_LOG', 'KGE_INV', 'KGE_SQRT',
        'NSE_LOG', 'NSE_SQRT', 'RMSE_LOG',
    ])
    def test_metric_accepted(self, name):
        cfg = OptimizationConfig(OPTIMIZATION_METRIC=name)
        assert cfg.metric == name.upper()
