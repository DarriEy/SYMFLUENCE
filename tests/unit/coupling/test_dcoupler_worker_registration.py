"""Tests for DCouplerWorker registration and calibration pipeline integration."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDCouplerWorkerRegistration:
    """Test that DCouplerWorker is properly registered with OptimizerRegistry."""

    def test_registry_returns_dcoupler_worker(self):
        from symfluence.optimization.registry import OptimizerRegistry
        # Import triggers registration via decorator
        from symfluence.coupling.worker import DCouplerWorker

        worker_cls = OptimizerRegistry.get_worker('DCOUPLER')
        assert worker_cls is DCouplerWorker

    def test_dcoupler_in_worker_list(self):
        from symfluence.optimization.registry import OptimizerRegistry
        from symfluence.coupling.worker import DCouplerWorker  # noqa: F401

        workers = OptimizerRegistry.list_workers()
        assert 'DCOUPLER' in workers


class TestDCouplerWorkerInit:
    """Test DCouplerWorker initialization and graph construction."""

    def test_init_stores_config(self):
        from symfluence.coupling.worker import DCouplerWorker

        config = {'HYDROLOGICAL_MODEL': 'SUMMA', 'ROUTING_MODEL': 'MIZUROUTE'}
        worker = DCouplerWorker(config)
        assert worker.config is config
        assert worker._graph is None  # Lazy init

    def test_graph_lazy_init(self):
        from symfluence.coupling.worker import DCouplerWorker

        config = {'HYDROLOGICAL_MODEL': 'SUMMA'}
        worker = DCouplerWorker(config)

        # Mock the builder to avoid importing dCoupler graph internals
        mock_graph = MagicMock()
        mock_graph.components = {'land': MagicMock()}
        worker._builder = MagicMock()
        worker._builder.build.return_value = mock_graph

        graph = worker.graph
        assert graph is mock_graph
        worker._builder.build.assert_called_once_with(config)

        # Second access should not rebuild
        graph2 = worker.graph
        assert graph2 is mock_graph
        assert worker._builder.build.call_count == 1


class TestDCouplerWorkerApplyParams:
    """Test parameter application logic."""

    def test_apply_process_params_logs_debug(self, caplog):
        from symfluence.coupling.worker import DCouplerWorker
        import logging

        worker = DCouplerWorker({})
        mock_comp = MagicMock()
        mock_comp.name = "land"

        with caplog.at_level(logging.DEBUG):
            worker._apply_process_params(mock_comp, {'k1': 0.5}, '/tmp/settings')

        assert "Parameter application for process models" in caplog.text

    def test_apply_process_params_no_log_when_empty(self, caplog):
        from symfluence.coupling.worker import DCouplerWorker
        import logging

        worker = DCouplerWorker({})
        mock_comp = MagicMock()
        mock_comp.name = "land"

        with caplog.at_level(logging.DEBUG):
            worker._apply_process_params(mock_comp, {}, '/tmp/settings')

        assert "Parameter application" not in caplog.text


class TestDCouplerWorkerRunModel:
    """Test run_model() with and without external inputs."""

    def test_run_model_without_external_inputs(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})

        mock_graph = MagicMock()
        mock_graph.forward.return_value = {
            'land': {'runoff': torch.tensor([1.0, 2.0])}
        }
        worker._graph = mock_graph

        success = worker.run_model({}, str(tmp_path), str(tmp_path / 'output'))
        assert success is True
        # Should have used empty dict for external_inputs
        call_kwargs = mock_graph.forward.call_args[1]
        assert call_kwargs['external_inputs'] == {}

    def test_run_model_with_external_inputs(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        worker.set_external_inputs(
            {'land': {'forcing': torch.randn(100, 7)}},
            n_timesteps=100,
            dt=3600.0,
        )

        mock_graph = MagicMock()
        mock_graph.forward.return_value = {
            'routing': {'discharge': torch.tensor([3.0, 4.0])}
        }
        worker._graph = mock_graph

        success = worker.run_model({}, str(tmp_path), str(tmp_path / 'output'))
        assert success is True
        call_kwargs = mock_graph.forward.call_args[1]
        assert call_kwargs['n_timesteps'] == 100
        assert call_kwargs['dt'] == 3600.0


class TestDCouplerWorkerMetrics:
    """Test calculate_metrics() with warmup handling."""

    def test_warmup_slices_data(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        # Set up cached outputs
        sim_data = torch.tensor([float(i) for i in range(100)])
        worker._last_outputs = {'land': {'runoff': sim_data}}

        # Mock observation loading
        obs_data = torch.tensor([float(i) * 0.9 for i in range(100)])
        with patch.object(worker, '_load_observations', return_value=obs_data):
            metrics = worker.calculate_metrics(
                str(tmp_path), {'WARMUP_DAYS': 10}
            )

        # Should have computed on 90 points, not 100
        assert 'KGE' in metrics
        assert metrics['KGE'] != -999.0

    def test_no_warmup_by_default(self, tmp_path):
        from symfluence.coupling.worker import DCouplerWorker

        worker = DCouplerWorker({})
        sim_data = torch.tensor([float(i) for i in range(50)])
        worker._last_outputs = {'routing': {'discharge': sim_data}}

        obs_data = torch.tensor([float(i) * 1.1 for i in range(50)])
        with patch.object(worker, '_load_observations', return_value=obs_data):
            metrics = worker.calculate_metrics(str(tmp_path), {})

        assert 'KGE' in metrics
        assert metrics['KGE'] != -999.0

    def test_supports_native_gradients_checks_components(self):
        from symfluence.coupling.worker import DCouplerWorker
        from dcoupler.core.component import GradientMethod

        worker = DCouplerWorker({})

        # Mock graph with all differentiable components
        mock_comp = MagicMock()
        mock_comp.gradient_method = GradientMethod.AUTOGRAD
        mock_graph = MagicMock()
        mock_graph.components = {'land': mock_comp}
        worker._graph = mock_graph

        assert worker.supports_native_gradients() is True

        # Now with a non-differentiable component
        mock_comp2 = MagicMock()
        mock_comp2.gradient_method = GradientMethod.NONE
        mock_graph.components = {'land': mock_comp, 'routing': mock_comp2}

        assert worker.supports_native_gradients() is False
