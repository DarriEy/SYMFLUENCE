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
from symfluence.core.metrics.metric_transformer import MetricTransformer
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics


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

        from symfluence.core.metrics.metrics_core import kge
        epsilon = max(np.mean(obs) * 0.01, 1e-6)
        lam = 0.2
        obs_t = ((obs + epsilon) ** lam - 1) / lam
        sim_t = ((sim + epsilon) ** lam - 1) / lam
        assert result['kge_box_cox'] == pytest.approx(float(kge(obs_t, sim_t, transfo=1)))


class TestConfigurableBoxCoxLambda:
    def _manual_box_cox_kge(self, obs, sim, lam):
        from symfluence.core.metrics.metrics_core import kge
        epsilon = max(np.mean(obs) * 0.01, 1e-6)
        obs_t = ((obs + epsilon) ** lam - 1) / lam
        sim_t = ((sim + epsilon) ** lam - 1) / lam
        return float(kge(obs_t, sim_t, transfo=1))

    def test_custom_lambda_honored(self, flows):
        obs, sim = flows
        sm = StreamflowMetrics(box_cox_lambda=0.5)
        result = sm.calculate_metrics(obs, sim, metrics=['kge_box_cox'])
        assert result['kge_box_cox'] == pytest.approx(self._manual_box_cox_kge(obs, sim, 0.5))

    def test_default_lambda_is_02(self, flows):
        obs, sim = flows
        result = StreamflowMetrics().calculate_metrics(obs, sim, metrics=['kge_box_cox'])
        assert result['kge_box_cox'] == pytest.approx(self._manual_box_cox_kge(obs, sim, 0.2))

    def test_lambda_zero_equals_log_transform(self, flows):
        obs, sim = flows
        sm = StreamflowMetrics(box_cox_lambda=0.0)
        result = sm.calculate_metrics(obs, sim, metrics=['kge_box_cox', 'kge_log'])
        assert result['kge_box_cox'] == pytest.approx(result['kge_log'])

    def test_base_worker_picks_up_configured_lambda(self):
        from symfluence.core.calibration.workers.base_worker import BaseWorker

        class DummyWorker(BaseWorker):
            _streamflow_metrics = StreamflowMetrics()

            def apply_parameters(self, *a, **k):
                return True

            def run_model(self, *a, **k):
                return True

            def calculate_metrics(self, *a, **k):
                return {}

        configured = DummyWorker(config={'BOX_COX_LAMBDA': 0.5})
        assert configured._streamflow_metrics.box_cox_lambda == 0.5

        default = DummyWorker(config={})
        assert default._streamflow_metrics.box_cox_lambda == 0.2

    def test_config_field_validation(self):
        assert OptimizationConfig(BOX_COX_LAMBDA=0.35).box_cox_lambda == 0.35
        assert OptimizationConfig().box_cox_lambda == 0.2
        with pytest.raises(ValueError):
            OptimizationConfig(BOX_COX_LAMBDA=1.5)
        with pytest.raises(ValueError):
            OptimizationConfig(BOX_COX_LAMBDA=-0.1)


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
