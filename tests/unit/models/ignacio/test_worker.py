"""Tests for IGNACIO calibration worker."""

import pytest
import tempfile
from pathlib import Path


class TestIGNACIOWorkerRegistration:
    """Tests for IGNACIO worker registration."""

    def test_worker_can_be_imported(self):
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker
        assert IGNACIOWorker is not None

    def test_worker_registered(self):
        from symfluence.optimization.registry import OptimizerRegistry
        assert 'IGNACIO' in OptimizerRegistry._workers

    def test_worker_is_correct_class(self):
        from symfluence.optimization.registry import OptimizerRegistry
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker
        assert OptimizerRegistry._workers.get('IGNACIO') == IGNACIOWorker


class TestIGNACIOWorkerProperties:
    """Tests for IGNACIO worker properties."""

    def test_no_gradient_support(self):
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker
        worker = IGNACIOWorker()
        assert worker.supports_native_gradients() is False


class TestIGNACIOParameterApplication:
    """Tests for IGNACIO parameter application to YAML."""

    def test_apply_params_in_memory(self):
        """Without a config file, params should be stored in-memory."""
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker

        worker = IGNACIOWorker()
        params = {'ffmc': 85.0, 'dmc': 40.0, 'dc': 200.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = worker.apply_parameters(params, Path(tmpdir))
            assert result is True
            assert worker._current_params == params

    def test_apply_params_to_yaml(self):
        """With a config file, params should be written to YAML."""
        import yaml
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker

        worker = IGNACIOWorker()
        params = {'ffmc': 90.0, 'dmc': 50.0, 'curing': 80.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_path = tmpdir / 'ignacio_config.yaml'
            config_path.write_text(yaml.dump({'fbp': {'ffmc_default': 88.0}}))

            result = worker.apply_parameters(params, tmpdir)
            assert result is True

            with open(config_path) as f:
                updated = yaml.safe_load(f)
            assert updated['fbp']['ffmc_default'] == 90.0
            assert updated['fbp']['dmc_default'] == 50.0
            assert updated['fbp']['curing'] == 80.0

    def test_apply_initial_radius(self):
        """initial_radius should be written to simulation section."""
        import yaml
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker

        worker = IGNACIOWorker()
        params = {'initial_radius': 25.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config_path = tmpdir / 'ignacio_config.yaml'
            config_path.write_text(yaml.dump({'simulation': {}}))

            worker.apply_parameters(params, tmpdir)

            with open(config_path) as f:
                updated = yaml.safe_load(f)
            assert updated['simulation']['initial_radius'] == 25.0


class TestIGNACIOMetricCalculation:
    """Tests for IGNACIO spatial metric calculation."""

    def test_metric_penalty_when_no_perimeters(self):
        """Should return penalty score when no simulated perimeters found."""
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker

        worker = IGNACIOWorker()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = worker.calculate_metrics(Path(tmpdir), {})
            assert 'error' in metrics

    def test_metric_penalty_when_no_observations(self):
        """Should return penalty when no observed perimeters found."""
        from symfluence.models.ignacio.calibration.worker import IGNACIOWorker

        worker = IGNACIOWorker()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = worker.calculate_metrics(Path(tmpdir), {'IGNACIO_OBSERVED_PERIMETER': '/nonexistent'})
            assert 'error' in metrics
