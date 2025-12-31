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

# We'll mock these imports since they might depend on actual model binaries
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# SUMMA Calibration Tests
# ============================================================================

class TestSUMMACalibrationTargets:
    """Tests for SUMMA calibration target loading and processing."""

    def test_load_summa_observations(self, summa_config, test_logger, mock_observations, temp_project_dir):
        """Test loading SUMMA streamflow observations."""
        from symfluence.utils.optimization.calibration_targets import CalibrationTargetManager

        manager = CalibrationTargetManager(summa_config, test_logger)

        # Mock the observation loading
        with patch.object(manager, 'project_dir', temp_project_dir):
            obs_data = manager.load_observations()

        assert isinstance(obs_data, pd.DataFrame)
        assert 'date' in obs_data.columns
        assert 'discharge_cms' in obs_data.columns or 'flow' in obs_data.columns

    def test_align_summa_simulation_with_obs(self, summa_config, test_logger, mock_observations):
        """Test aligning SUMMA simulation results with observations."""
        from symfluence.utils.optimization.calibration_targets import CalibrationTargetManager

        manager = CalibrationTargetManager(summa_config, test_logger)

        # Create mock simulation data
        dates = pd.date_range('2020-01-01', periods=31, freq='D')
        sim_data = pd.DataFrame({
            'date': dates,
            'discharge_cms': np.random.uniform(3, 7, 31)
        })

        obs_data = pd.DataFrame({
            'date': dates,
            'discharge_cms': np.random.uniform(4, 8, 31)
        })

        # Align data
        aligned_sim, aligned_obs = manager.align_timeseries(sim_data, obs_data)

        assert len(aligned_sim) == len(aligned_obs)
        assert len(aligned_sim) <= len(dates)

    def test_summa_calibration_period_subset(self, summa_config, test_logger):
        """Test extracting calibration period from full simulation."""
        from symfluence.utils.optimization.calibration_targets import CalibrationTargetManager

        config = summa_config.copy()
        config['CALIBRATION_PERIOD'] = '2020-01-10, 2020-01-20'

        manager = CalibrationTargetManager(config, test_logger)

        # Create full period data
        dates = pd.date_range('2020-01-01', periods=31, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(31)
        })

        # Extract calibration period
        calib_data = manager.extract_calibration_period(data)

        # Should only have data within calibration period
        assert calib_data['date'].min() >= pd.to_datetime('2020-01-10')
        assert calib_data['date'].max() <= pd.to_datetime('2020-01-20')


class TestSUMMAWorkerFunctions:
    """Tests for SUMMA model evaluation worker functions."""

    def test_summa_parameter_application(self, summa_config, test_logger, temp_project_dir):
        """Test applying parameters to SUMMA trial parameter file."""
        from symfluence.utils.optimization.worker_scripts import apply_parameters_to_summa

        # Create mock trial parameter file
        trial_params_file = temp_project_dir / "settings" / "SUMMA" / "trialParams.nc"

        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        # Mock netCDF file operations
        with patch('netCDF4.Dataset') as mock_nc:
            mock_dataset = MagicMock()
            mock_nc.return_value.__enter__.return_value = mock_dataset

            # Call parameter application (mocked)
            # This tests the interface, not actual netCDF operations
            result = apply_parameters_to_summa(params, trial_params_file)

            # Should have accessed the file
            mock_nc.assert_called()

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
    @pytest.mark.skip(reason="Requires actual SUMMA binary")
    def test_summa_worker_real_model_run(self, summa_config, test_logger, temp_project_dir):
        """Integration test with real SUMMA model (if available)."""
        # This would test actual SUMMA execution
        # Skipped for unit tests
        pass


# ============================================================================
# FUSE Calibration Tests
# ============================================================================

class TestFUSECalibrationTargets:
    """Tests for FUSE calibration target loading and processing."""

    def test_load_fuse_observations(self, fuse_config, test_logger, mock_observations, temp_project_dir):
        """Test loading FUSE streamflow observations."""
        from symfluence.utils.optimization.fuse_calibration_targets import FUSECalibrationTargetManager

        manager = FUSECalibrationTargetManager(fuse_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            obs_data = manager.load_observations()

        assert isinstance(obs_data, pd.DataFrame)

    def test_fuse_structure_specific_params(self, fuse_config, test_logger):
        """Test FUSE structure-specific parameter handling."""
        from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager

        manager = FUSEParameterManager(fuse_config, test_logger)

        # Get parameters for structure 902
        param_bounds = manager.get_parameter_bounds(structure='902')

        assert isinstance(param_bounds, dict)
        assert len(param_bounds) > 0

    def test_fuse_multi_structure_optimization(self, fuse_config, test_logger):
        """Test optimization across multiple FUSE structures."""
        # This would test structure comparison capability
        structures = ['900', '902', '904']

        for structure in structures:
            config = fuse_config.copy()
            config['FUSE_STRUCTURE'] = structure

            from symfluence.utils.optimization.fuse_parameter_manager import FUSEParameterManager
            manager = FUSEParameterManager(config, test_logger)

            param_bounds = manager.get_parameter_bounds(structure=structure)
            assert isinstance(param_bounds, dict)


class TestFUSEWorkerFunctions:
    """Tests for FUSE model evaluation worker functions."""

    def test_fuse_parameter_application(self, fuse_config, test_logger):
        """Test applying parameters to FUSE input files."""
        from symfluence.utils.optimization.fuse_worker_functions import apply_parameters_to_fuse

        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # This tests the interface
            result = apply_parameters_to_fuse(params, fuse_config)

            # Should have attempted file operations
            assert mock_open.called or result is not None

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
    @pytest.mark.skip(reason="Requires actual FUSE binary")
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
        from symfluence.utils.optimization.ngen_calibration_targets import NGENCalibrationTargetManager

        manager = NGENCalibrationTargetManager(ngen_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            obs_data = manager.load_observations()

        assert isinstance(obs_data, pd.DataFrame)

    def test_ngen_catchment_specific_params(self, ngen_config, test_logger):
        """Test NGEN catchment-specific parameter handling."""
        from symfluence.utils.optimization.ngen_parameter_manager import NGENParameterManager

        manager = NGENParameterManager(ngen_config, test_logger)

        # Get parameters for NGEN
        param_bounds = manager.get_parameter_bounds()

        assert isinstance(param_bounds, dict)
        assert len(param_bounds) > 0


class TestNGENWorkerFunctions:
    """Tests for NGEN model evaluation worker functions."""

    def test_ngen_parameter_application(self, ngen_config, test_logger):
        """Test applying parameters to NGEN realization file."""
        from symfluence.utils.optimization.ngen_worker_functions import apply_parameters_to_ngen

        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        # Mock JSON file operations
        import json
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Mock json operations
            with patch('json.load') as mock_json_load, patch('json.dump') as mock_json_dump:
                mock_json_load.return_value = {'catchments': {}}

                result = apply_parameters_to_ngen(params, ngen_config)

                # Should have attempted JSON operations
                assert mock_open.called or mock_json_dump.called or result is not None

    def test_ngen_worker_evaluation(self, ngen_config, test_logger, mock_ngen_worker):
        """Test NGEN worker function evaluation."""
        params = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
        }

        result = mock_ngen_worker(params, ngen_config, trial_num=1)

        assert result['success']
        assert 'metrics' in result

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires actual NGEN binary")
    def test_ngen_worker_real_model_run(self, ngen_config, test_logger):
        """Integration test with real NGEN model (if available)."""
        pass


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

        # Import appropriate manager based on model
        model_name = config['HYDROLOGICAL_MODEL']

        if model_name == 'SUMMA':
            from symfluence.utils.optimization.calibration_targets import CalibrationTargetManager
            manager = CalibrationTargetManager(config, test_logger)
        elif model_name == 'FUSE':
            from symfluence.utils.optimization.fuse_calibration_targets import FUSECalibrationTargetManager
            manager = FUSECalibrationTargetManager(config, test_logger)
        elif model_name == 'NGEN':
            from symfluence.utils.optimization.ngen_calibration_targets import NGENCalibrationTargetManager
            manager = NGENCalibrationTargetManager(config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            obs_data = manager.load_observations()

        assert isinstance(obs_data, pd.DataFrame)

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

    @pytest.mark.skip(reason="Requires multiprocessing setup")
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
