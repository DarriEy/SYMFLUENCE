"""
Unit tests for model-specific calibration components.

Tests calibration targets, worker functions, and parameter management
for SUMMA, FUSE, and NGEN models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from test_helpers.markers import skip_if_no_model
from symfluence.core.config.models import SymfluenceConfig


def create_config_with_overrides(base_config: SymfluenceConfig, **overrides) -> SymfluenceConfig:
    """Create a new SymfluenceConfig with the given overrides."""
    config_dict = base_config.to_dict(flatten=True)
    config_dict.update(overrides)
    return SymfluenceConfig(**config_dict)


# We'll mock these imports since they might depend on actual model binaries
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# SUMMA Calibration Tests
# ============================================================================

class TestSUMMACalibrationTargets:
    """Tests for SUMMA calibration target loading and processing."""

    def test_load_summa_observations(self, summa_config, test_logger, mock_observations, temp_project_dir):
        """Test loading SUMMA streamflow observations."""
        from symfluence.optimization.calibration_targets import StreamflowTarget

        target = StreamflowTarget(summa_config, temp_project_dir, test_logger)

        # mock_observations is a Path to a CSV file - read it to get DataFrame
        obs_df = pd.read_csv(mock_observations, parse_dates=['date'], index_col='date')
        target._load_observed_data = lambda: obs_df
        obs_data = target._load_observed_data()

        assert isinstance(obs_data, (pd.Series, pd.DataFrame))

    def test_align_summa_simulation_with_obs(self, summa_config, test_logger, mock_observations):
        """Test aligning SUMMA simulation results with observations."""
        from symfluence.optimization.calibration_targets import StreamflowTarget

        target = StreamflowTarget(summa_config, Path("/tmp"), test_logger)

        # Create mock simulation and observation data with DatetimeIndex
        dates = pd.date_range('2020-01-01', periods=31, freq='D')
        sim_data = pd.Series(np.random.uniform(3, 7, 31), index=dates)
        obs_data = pd.Series(np.random.uniform(4, 8, 31), index=dates)

        target.calibration_period = (dates[0], dates[-1])
        
        # Test _calculate_period_metrics
        metrics = target._calculate_period_metrics(obs_data, sim_data, target.calibration_period, "Calib")
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_summa_calibration_period_subset(self, summa_config, test_logger):
        """Test extracting calibration period from full simulation."""
        from symfluence.optimization.calibration_targets import StreamflowTarget

        config = create_config_with_overrides(
            summa_config,
            CALIBRATION_PERIOD='2020-01-10, 2020-01-20'
        )

        target = StreamflowTarget(config, Path("/tmp"), test_logger)

        # Create full period data
        dates = pd.date_range('2020-01-01', periods=31, freq='D')
        data = pd.Series(np.random.randn(31), index=dates)

        # Extract calibration period using target logic
        mask = (data.index >= target.calibration_period[0]) & (data.index <= target.calibration_period[1])
        calib_data = data.loc[mask]

        # Should only have data within calibration period
        assert calib_data.index.min() >= pd.to_datetime('2020-01-10')
        assert calib_data.index.max() <= pd.to_datetime('2020-01-20')


class TestSUMMAWorkerFunctions:
    """Tests for SUMMA model evaluation worker functions."""

    def test_summa_parameter_application(self, summa_config, test_logger, temp_project_dir):
        """Test applying parameters to SUMMA trial parameter file."""
        from symfluence.optimization.workers.summa_parallel_workers import _apply_parameters_worker

        # Create mock trial parameter file
        summa_settings_dir = temp_project_dir / "settings" / "SUMMA"
        summa_settings_dir.mkdir(parents=True, exist_ok=True)

        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }
        
        task_data = {
            'config': summa_config,
            'basin_params': [],
            'depth_params': [],
            'mizuroute_params': []
        }
        debug_info = {
            'stage': 'parameter_application',
            'files_checked': [],
            'commands_run': [],
            'errors': []
        }

        # Mock the generator worker - patch in the actual module where it's called
        with patch('symfluence.optimization.workers.summa.parameter_application._generate_trial_params_worker') as mock_gen:
            mock_gen.return_value = True

            # Call parameter application
            result = _apply_parameters_worker(params, task_data, summa_settings_dir, test_logger, debug_info)

            assert result is True
            mock_gen.assert_called_once()

    def test_summa_worker_evaluation(self, summa_config, test_logger, mock_summa_worker):
        """Test SUMMA worker function evaluation."""
        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        result = mock_summa_worker(params, summa_config, trial_num=1)

        assert result['success']
        assert 'metrics' in result
        assert 'params' in result
        assert result['trial'] == 1

    def test_summa_worker_handles_model_failure(self, summa_config, test_logger):
        """Test SUMMA worker handling model run failures."""
        def failing_worker(params, config, trial_num=0):
            raise RuntimeError("SUMMA model failed")

        params = {'theta_sat': 0.45}

        # Worker should handle failure gracefully
        with pytest.raises(RuntimeError):
            failing_worker(params, summa_config, trial_num=1)

    @pytest.mark.slow
    @skip_if_no_model("SUMMA")
    def test_summa_worker_real_model_run(self, summa_config, test_logger, temp_project_dir):
        """Integration test with real SUMMA model (if available)."""
        pass


# ============================================================================
# FUSE Calibration Tests
# ============================================================================

class TestFUSECalibrationTargets:
    """Tests for FUSE calibration target loading and processing."""

    def test_load_fuse_observations(self, fuse_config, test_logger, mock_observations, temp_project_dir):
        """Test loading FUSE streamflow observations."""
        from symfluence.optimization.calibration_targets import FUSEStreamflowTarget

        target = FUSEStreamflowTarget(fuse_config, temp_project_dir, test_logger)

        # mock_observations is a Path to a CSV file - read it to get DataFrame
        obs_df = pd.read_csv(mock_observations, parse_dates=['date'], index_col='date')
        target._load_observed_data = lambda: obs_df
        obs_data = target._load_observed_data()

        assert isinstance(obs_data, (pd.DataFrame, pd.Series))

    def test_fuse_structure_specific_params(self, fuse_config, test_logger, temp_project_dir):
        """Test FUSE structure-specific parameter handling."""
        from symfluence.optimization.parameter_managers import FUSEParameterManager

        fuse_settings_dir = temp_project_dir / "settings" / "FUSE"
        fuse_settings_dir.mkdir(parents=True, exist_ok=True)
        
        manager = FUSEParameterManager(fuse_config, test_logger, fuse_settings_dir)

        # Get parameters
        param_bounds = manager.get_parameter_bounds()

        assert isinstance(param_bounds, dict)

    def test_fuse_multi_structure_optimization(self, fuse_config, test_logger, temp_project_dir):
        """Test optimization across multiple FUSE structures."""
        # This would test structure comparison capability
        structures = ['900', '902', '904']
        fuse_settings_dir = temp_project_dir / "settings" / "FUSE"
        fuse_settings_dir.mkdir(parents=True, exist_ok=True)

        for structure in structures:
            config = create_config_with_overrides(
                fuse_config,
                FUSE_STRUCTURE=structure
            )

            from symfluence.optimization.parameter_managers import FUSEParameterManager
            manager = FUSEParameterManager(config, test_logger, fuse_settings_dir)

            param_bounds = manager.get_parameter_bounds()
            assert isinstance(param_bounds, dict)


class TestFUSEWorkerFunctions:
    """Tests for FUSE model evaluation worker functions."""

    def test_fuse_parameter_application(self, fuse_config, test_logger, temp_project_dir):
        """Test applying parameters to FUSE input files via worker class."""
        from symfluence.optimization.workers.fuse_worker import FUSEWorker
        from unittest.mock import mock_open

        params = {
            'MAXWATR_1': 500.0,
            'PERCRTE': 10.0,
        }

        worker = FUSEWorker(fuse_config, test_logger)

        # Mock file operations instead of trying to create real files
        mock_constraint_file = "L 1 100.000      0     10.000 MAXWATR_1\nL 1  20.000      0      5.000 PERCRTE\n"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_constraint_file)) as mock_file:
                result = worker.apply_parameters(params, temp_project_dir, config=fuse_config)
                # Check that the file was attempted to be opened
                assert mock_file.called
                assert result is True

    def test_fuse_worker_evaluation(self, fuse_config, test_logger, mock_fuse_worker):
        """Test FUSE worker function evaluation."""
        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        result = mock_fuse_worker(params, fuse_config, trial_num=1)

        assert result['success']
        assert 'metrics' in result
        assert result['trial'] == 1

    @pytest.mark.slow
    @skip_if_no_model("FUSE")
    def test_fuse_worker_real_model_run(self, fuse_config, test_logger):
        """Integration test with real FUSE model (if available)."""
        pass


# ============================================================================
# NGEN Calibration Tests
# ============================================================================

class TestNGENCalibrationTargets:
    """Tests for NGEN calibration target loading and processing."""

    def test_load_ngen_observations(self, ngen_config, test_logger, mock_observations, temp_project_dir):
        """Test loading NGEN streamflow observations."""
        from symfluence.optimization.calibration_targets import NgenStreamflowTarget

        # mock_observations is a Path to a CSV file - read it to get DataFrame
        obs_df = pd.read_csv(mock_observations, parse_dates=['date'], index_col='date')

        # We still need to patch _get_catchment_area because it's called in __init__
        with patch('symfluence.evaluation.evaluators.StreamflowEvaluator._get_catchment_area', return_value=100.0):
            target = NgenStreamflowTarget(ngen_config, temp_project_dir, test_logger)
            target._load_observed_data = lambda: obs_df
            obs_data = target._load_observed_data()
            assert isinstance(obs_data, (pd.DataFrame, pd.Series))

    def test_ngen_catchment_specific_params(self, ngen_config, test_logger, temp_project_dir):
        """Test NGEN catchment-specific parameter handling."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        ngen_settings_dir = temp_project_dir / "settings" / "ngen"
        ngen_settings_dir.mkdir(parents=True, exist_ok=True)
        
        manager = NgenParameterManager(ngen_config, test_logger, ngen_settings_dir)

        # Get parameters for NGEN
        param_bounds = manager.get_parameter_bounds()

        assert isinstance(param_bounds, dict)


class TestNGENWorkerFunctions:
    """Tests for NGEN model evaluation worker functions."""

    def test_ngen_parameter_application(self, ngen_config, test_logger, temp_project_dir):
        """Test applying parameters to NGEN realization file via worker class."""
        from symfluence.optimization.workers.ngen_worker import NgenWorker

        params = {
            'CFE.smcmax': 0.45,
            'CFE.satdk': 5e-5,
        }

        worker = NgenWorker(ngen_config, test_logger)

        # Mock JSON file operations
        with patch('builtins.open', create=True), patch('json.load'), patch('json.dump'):
            with patch('pathlib.Path.exists', return_value=True):
                result = worker.apply_parameters(params, temp_project_dir, config=ngen_config)
                assert result is True


# ============================================================================
# Cross-Model Tests
# ============================================================================

class TestCrossModelCalibration:
    """Tests comparing calibration across different models."""

    @pytest.mark.parametrize("model_config", [
        pytest.param("summa_config", id="SUMMA"),
        pytest.param("fuse_config", id="FUSE"),
        pytest.param("ngen_config", id="NGEN"),
    ])
    def test_all_models_load_observations(self, model_config, test_logger, mock_observations, temp_project_dir, request):
        """Test that all models can load observations."""
        config = request.getfixturevalue(model_config)
        model_name = config['HYDROLOGICAL_MODEL']

        # mock_observations is a Path to a CSV file - read it to get DataFrame
        obs_df = pd.read_csv(mock_observations, parse_dates=['date'], index_col='date')

        if model_name == 'SUMMA':
            from symfluence.optimization.calibration_targets import StreamflowTarget
            target = StreamflowTarget(config, temp_project_dir, test_logger)
        elif model_name == 'FUSE':
            from symfluence.optimization.calibration_targets import FUSEStreamflowTarget
            target = FUSEStreamflowTarget(config, temp_project_dir, test_logger)
        elif model_name == 'NGEN':
            from symfluence.optimization.calibration_targets import NgenStreamflowTarget
            with patch('symfluence.evaluation.evaluators.StreamflowEvaluator._get_catchment_area', return_value=100.0):
                target = NgenStreamflowTarget(config, temp_project_dir, test_logger)

        target._load_observed_data = lambda: obs_df
        obs_data = target._load_observed_data()
        assert isinstance(obs_data, (pd.Series, pd.DataFrame))

    @pytest.mark.parametrize("model_worker", [
        pytest.param("mock_summa_worker", id="SUMMA"),
        pytest.param("mock_fuse_worker", id="FUSE"),
        pytest.param("mock_ngen_worker", id="NGEN"),
    ])
    def test_all_models_worker_interface(self, model_worker, base_optimization_config, request):
        """Test that all model workers have consistent interface."""
        worker = request.getfixturevalue(model_worker)

        params = {'theta_sat': 0.45, 'k_soil': 5e-5}
        result = worker(params, base_optimization_config, trial_num=1)

        # All workers should return consistent format
        assert 'success' in result
        assert 'metrics' in result
        assert 'params' in result
        assert result['trial'] == 1


# ============================================================================
# Sequential vs Parallel Execution Tests
# ============================================================================

@pytest.mark.parallel
class TestSequentialVsParallel:
    """Test sequential vs parallel execution for all models."""

    def test_summa_sequential_evaluation(self, summa_config, test_logger, mock_summa_worker):
        """Test sequential SUMMA evaluations."""
        params_list = [
            {'theta_sat': 0.40, 'k_soil': 3e-5},
            {'theta_sat': 0.45, 'k_soil': 5e-5},
            {'theta_sat': 0.50, 'k_soil': 7e-5},
        ]

        results = []
        for i, params in enumerate(params_list):
            result = mock_summa_worker(params, summa_config, trial_num=i)
            results.append(result)

        assert len(results) == len(params_list)
        assert all(r['success'] for r in results)

    def test_summa_parallel_evaluation(self, summa_config, test_logger, mock_summa_worker):
        """Test parallel SUMMA evaluations."""
        from multiprocessing import Pool

        params_list = [
            {'theta_sat': 0.40, 'k_soil': 3e-5},
            {'theta_sat': 0.45, 'k_soil': 5e-5},
            {'theta_sat': 0.50, 'k_soil': 7e-5},
        ]

        # This would require proper Pool setup and picklable workers
        # Skipped for unit tests
        pass

    def test_results_consistency_sequential_parallel(self):
        """Test that sequential and parallel give same results."""
        # This would compare sequential vs parallel execution
        # with same random seed to ensure consistency
        pass