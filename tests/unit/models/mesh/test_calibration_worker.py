"""
Tests for MESH calibration worker.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd


class TestMESHWorkerInitialization:
    """Tests for MESH worker initialization."""

    def test_worker_can_be_imported(self):
        """Test that MESHWorker can be imported."""
        from symfluence.models.mesh.calibration.worker import MESHWorker
        assert MESHWorker is not None

    def test_worker_initialization(self, mock_logger):
        """Test worker initializes without config."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)
        assert worker is not None

    def test_worker_initialization_with_config(self, mesh_config, mock_logger):
        """Test worker initializes with config."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        config_dict = mesh_config.model_dump()
        worker = MESHWorker(config=config_dict, logger=mock_logger)
        assert worker is not None


class TestMESHWorkerRegistry:
    """Tests for MESH worker registry integration."""

    def test_worker_registered_with_registry(self):
        """Test MESH worker is registered with optimizer registry."""
        from symfluence.optimization.registry import OptimizerRegistry

        workers = OptimizerRegistry._workers
        assert 'MESH' in workers

    def test_registered_worker_is_correct_class(self):
        """Test registered worker is MESHWorker."""
        from symfluence.optimization.registry import OptimizerRegistry
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker_class = OptimizerRegistry._workers.get('MESH')
        assert worker_class == MESHWorker


class TestMESHWorkerParameterApplication:
    """Tests for MESH worker parameter application."""

    def test_apply_parameters_calls_parameter_manager(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test apply_parameters uses MESHParameterManager."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        params = {'ZSNL': 0.05, 'MANN': 0.15, 'RCHARG': 0.5}
        settings_dir = setup_mesh_directories['forcing_dir']

        with patch('symfluence.optimization.registry.OptimizerRegistry.get_parameter_manager') as mock_get_pm:
            mock_pm_class = MagicMock()
            mock_pm_instance = MagicMock()
            mock_pm_instance.update_model_files.return_value = True
            mock_pm_class.return_value = mock_pm_instance
            mock_get_pm.return_value = mock_pm_class

            config_dict = mesh_config.model_dump()
            result = worker.apply_parameters(params, settings_dir, config=config_dict)

            mock_get_pm.assert_called_once_with('MESH')
            mock_pm_instance.update_model_files.assert_called_once_with(params)
            assert result is True

    def test_apply_parameters_returns_false_on_error(self, mock_logger, setup_mesh_directories):
        """Test apply_parameters returns False on error."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        params = {'ZSNL': 0.05}
        settings_dir = setup_mesh_directories['forcing_dir']

        with patch('symfluence.optimization.registry.OptimizerRegistry.get_parameter_manager') as mock_get_pm:
            mock_get_pm.return_value = None

            result = worker.apply_parameters(params, settings_dir)

            assert result is False


class TestMESHWorkerModelExecution:
    """Tests for MESH worker model execution."""

    def test_run_model_initializes_runner(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test run_model initializes MESHRunner."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        config_dict = mesh_config.model_dump()
        settings_dir = setup_mesh_directories['forcing_dir']
        output_dir = setup_mesh_directories['simulations_dir']

        with patch('symfluence.models.mesh.calibration.worker.MESHRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run_mesh.return_value = output_dir
            mock_runner_class.return_value = mock_runner

            result = worker.run_model(
                config_dict, settings_dir, output_dir,
                proc_forcing_dir=str(settings_dir)
            )

            mock_runner_class.assert_called_once()
            assert result is True

    def test_run_model_returns_false_on_failure(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test run_model returns False when runner fails."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        config_dict = mesh_config.model_dump()
        settings_dir = setup_mesh_directories['forcing_dir']
        output_dir = setup_mesh_directories['simulations_dir']

        with patch('symfluence.models.mesh.calibration.worker.MESHRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run_mesh.return_value = None  # Failure
            mock_runner_class.return_value = mock_runner

            result = worker.run_model(config_dict, settings_dir, output_dir)

            assert result is False


class TestMESHWorkerMetricCalculation:
    """Tests for MESH worker metric calculation."""

    def test_calculate_metrics_returns_penalty_when_missing(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test metric calculation returns penalty when output missing."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        config_dict = mesh_config.model_dump()
        output_dir = setup_mesh_directories['simulations_dir']

        result = worker.calculate_metrics(output_dir, config_dict)

        assert 'kge' in result
        assert result['kge'] == worker.penalty_score

    def test_calculate_metrics_handles_exception(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test metric calculation handles exceptions gracefully."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(logger=mock_logger)

        config_dict = mesh_config.model_dump()
        output_dir = setup_mesh_directories['simulations_dir']

        # Create a file that will cause parsing issues
        bad_file = output_dir / 'MESH_output_streamflow.csv'
        bad_file.write_text("bad,data\n{not,csv}\n")

        result = worker.calculate_metrics(output_dir, config_dict)

        assert 'kge' in result
        assert result['kge'] == worker.penalty_score


class TestMESHWorkerFunction:
    """Tests for MESH module-level worker function."""

    def test_evaluate_worker_function_exists(self):
        """Test evaluate_worker_function static method exists."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        assert hasattr(MESHWorker, 'evaluate_worker_function')

    def test_module_level_worker_function_importable(self):
        """Test module-level worker function can be imported."""
        from symfluence.models.mesh.calibration.worker import _evaluate_mesh_parameters_worker

        assert _evaluate_mesh_parameters_worker is not None
        assert callable(_evaluate_mesh_parameters_worker)
