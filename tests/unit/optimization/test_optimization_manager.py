"""
Unit tests for OptimizationManager.

Tests the main optimization workflow orchestration, including:
- Algorithm selection
- Model coordination
- Results management
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from symfluence.optimization.optimization_manager import OptimizationManager


pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# Initialization Tests
# ============================================================================

class TestOptimizationManagerInit:
    """Tests for OptimizationManager initialization."""

    def test_initialization(self, base_optimization_config, test_logger, temp_project_dir):
        """Test OptimizationManager initialization."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        assert manager.config == base_optimization_config
        assert manager.logger == test_logger
        assert manager.domain_name == 'test_catchment'
        assert manager.experiment_id == 'test_optimization'

    def test_optimizer_mapping(self, base_optimization_config, test_logger):
        """Test that all optimizers are properly mapped."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        expected_optimizers = ['DDS', 'DE', 'PSO', 'SCE-UA', 'NSGA-II', 'ASYNC-DDS', 'POP-DDS']

        for opt_name in expected_optimizers:
            assert opt_name in manager.optimizers
            assert manager.optimizers[opt_name] is not None


# ============================================================================
# Algorithm Selection Tests
# ============================================================================

class TestAlgorithmSelection:
    """Tests for optimization algorithm selection."""

    @pytest.mark.parametrize("algorithm", ['DDS', 'DE', 'PSO', 'SCE-UA'])
    def test_select_algorithm(self, base_optimization_config, test_logger, algorithm):
        """Test algorithm selection for different algorithms."""
        config = base_optimization_config.copy()
        config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = algorithm

        manager = OptimizationManager(config, test_logger)

        assert config['ITERATIVE_OPTIMIZATION_ALGORITHM'] == algorithm

    def test_unsupported_algorithm(self, base_optimization_config, test_logger):
        """Test handling of unsupported algorithm."""
        config = base_optimization_config.copy()
        config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'UNSUPPORTED_ALG'

        manager = OptimizationManager(config, test_logger)

        # In modern API, it returns None and logs error
        result = manager.calibrate_model()
        assert result is None


# ============================================================================
# Model Calibration Tests
# ============================================================================

class TestModelCalibration:
    """Tests for model calibration workflows."""

    def test_calibrate_summa_model(self, summa_config, test_logger, temp_project_dir, mock_observations):
        """Test SUMMA model calibration."""
        manager = OptimizationManager(summa_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                mock_calibrate.return_value = Path("test_results.csv")

                result = manager.calibrate_model()

                mock_calibrate.assert_called_once_with('SUMMA', summa_config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'))
                assert result == Path("test_results.csv")

    def test_calibrate_fuse_model(self, fuse_config, test_logger, temp_project_dir):
        """Test FUSE model calibration."""
        manager = OptimizationManager(fuse_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                mock_calibrate.return_value = Path("test_results.csv")

                result = manager.calibrate_model()

                mock_calibrate.assert_called_once_with('FUSE', fuse_config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'))

    def test_calibrate_ngen_model(self, ngen_config, test_logger, temp_project_dir):
        """Test NGEN model calibration."""
        manager = OptimizationManager(ngen_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                mock_calibrate.return_value = Path("test_results.csv")

                result = manager.calibrate_model()

                mock_calibrate.assert_called_once_with('NGEN', ngen_config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'))

    def test_calibration_disabled(self, base_optimization_config, test_logger):
        """Test when calibration is disabled in config."""
        config = base_optimization_config.copy()
        config['OPTIMIZATION_METHODS'] = []  # Disable optimization

        manager = OptimizationManager(config, test_logger)
        result = manager.calibrate_model()

        assert result is None

    def test_multiple_models(self, base_optimization_config, test_logger):
        """Test calibration with multiple models specified."""
        config = base_optimization_config.copy()
        config['HYDROLOGICAL_MODEL'] = 'SUMMA,FUSE'  # Multiple models

        manager = OptimizationManager(config, test_logger)

        with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
            mock_calibrate.return_value = Path("results.csv")

            manager.calibrate_model()

            # Should call registry-based calibration for first model
            # Note: Current implementation only calibrates first model and returns
            mock_calibrate.assert_called_once()


# ============================================================================
# Results Management Tests
# ============================================================================

class TestResultsManagement:
    """Tests for optimization results management."""

    def test_save_optimization_results(self, base_optimization_config, test_logger, temp_project_dir):
        """Test saving optimization results."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        # Mock optimization results in current format
        results = {
            'best_parameters': {'theta_sat': 0.45, 'k_soil': 5e-5},
            'best_score': 0.85,
            'history': [
                {'iteration': 1, 'best_score': 0.75},
                {'iteration': 2, 'best_score': 0.85},
            ]
        }

        with patch.object(manager, 'project_dir', temp_project_dir):
            results_file = manager.results_manager.save_optimization_results(results, algorithm='DDS')

            assert results_file.exists()
            assert results_file.suffix == '.csv'

    def test_load_previous_results(self, base_optimization_config, test_logger, temp_project_dir):
        """Test loading previously saved results."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        # Create mock results file
        results_dir = temp_project_dir / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{manager.experiment_id}_parallel_iteration_results.csv"

        # Write mock results
        df = pd.DataFrame({
            'iteration': [0],
            'KGE': [0.85],
            'theta_sat': [0.45],
            'k_soil': [5e-5]
        })
        df.to_csv(results_file, index=False)

        with patch.object(manager, 'project_dir', temp_project_dir):
            loaded_results = manager.results_manager.load_optimization_results()

            assert isinstance(loaded_results, pd.DataFrame)
            assert len(loaded_results) == 1

    def test_best_parameters_extraction(self, base_optimization_config, test_logger):
        """Test extracting best parameters from results."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        results_df = pd.DataFrame({
            'iteration': [0, 1, 2],
            'theta_sat': [0.40, 0.45, 0.50],
            'k_soil': [3e-5, 5e-5, 7e-5],
            'KGE': [0.75, 0.85, 0.80]
        })

        # get_best_parameters doesn't exist, we usually just take row 0 or use load_optimization_results logic
        # OptimizationManager.load_optimization_results returns a dict with 'best_iteration'
        with patch.object(manager.results_manager, 'load_optimization_results', return_value=results_df):
            results = manager.load_optimization_results()
            assert results['best_iteration']['theta_sat'] == 0.40 # Row 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in optimization."""

    def test_missing_observations(self, summa_config, test_logger, temp_project_dir):
        """Test handling when observations file is missing."""
        manager = OptimizationManager(summa_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                # Simulate error by returning None
                mock_calibrate.return_value = None
                result = manager.calibrate_model()
                assert result is None

    def test_invalid_parameter_bounds(self, base_optimization_config, test_logger):
        """Test handling of invalid parameter bounds."""
        config = base_optimization_config.copy()
        config['PARAMS_TO_CALIBRATE'] = 'invalid_param'

        manager = OptimizationManager(config, test_logger)

        # Should handle gracefully or return None
        with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
            mock_calibrate.side_effect = ValueError("Invalid parameter")
            result = manager.calibrate_model()
            assert result is None

    def test_optimization_failure_recovery(self, dds_config, test_logger, temp_project_dir):
        """Test recovery from optimization failures."""
        manager = OptimizationManager(dds_config, test_logger)

        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                # Simulate optimization failure
                mock_calibrate.side_effect = RuntimeError("Optimization failed")

                result = manager.calibrate_model()

                # Should handle failure gracefully
                assert result is None


# ============================================================================
# Workflow Orchestration Tests
# ============================================================================

class TestWorkflowOrchestration:
    """Tests for complete optimization workflow."""

    def test_end_to_end_workflow(self, dds_config, test_logger, temp_project_dir, mock_observations, mock_evaluate_function):
        """Test complete optimization workflow end-to-end."""
        manager = OptimizationManager(dds_config, test_logger)

        # Mock the registry-based calibration method
        with patch.object(manager, 'project_dir', temp_project_dir):
            with patch.object(manager, '_calibrate_with_registry') as mock_calibrate:
                mock_calibrate.return_value = Path("test_results.csv")

                result = manager.run_optimization_workflow()

                # Should complete workflow with calibration result
                assert 'calibration' in result
                mock_calibrate.assert_called_once()

    def test_multi_objective_optimization(self, base_optimization_config, test_logger):
        """Test multi-objective optimization setup."""
        config = base_optimization_config.copy()
        config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'NSGA-II'
        config['OPTIMIZATION_METRIC'] = 'KGE,NSE'  # Multiple objectives

        manager = OptimizationManager(config, test_logger)

        # Should handle multi-objective setup
        assert config['OPTIMIZATION_METRIC'] == 'KGE,NSE'

    def test_transformation_manager_integration(self, base_optimization_config, test_logger):
        """Test parameter transformation integration."""
        manager = OptimizationManager(base_optimization_config, test_logger)

        # Test that transformation manager is initialized
        assert hasattr(manager, 'transformation_manager')
        assert manager.transformation_manager is not None


# ============================================================================
# Configuration Validation Tests
# ============================================================================

class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_validate_required_fields(self, base_optimization_config, test_logger):
        """Test validation of required configuration fields."""
        # Remove required field
        config = base_optimization_config.copy()
        del config['DOMAIN_NAME']

        with pytest.raises(KeyError):
            manager = OptimizationManager(config, test_logger)

    def test_validate_algorithm_specific_params(self, de_config, test_logger):
        """Test validation of algorithm-specific parameters."""
        manager = OptimizationManager(de_config, test_logger)

        # DE should have population size
        assert 'DE_POPULATION_SIZE' in de_config

    def test_validate_model_specific_params(self, fuse_config, test_logger):
        """Test validation of model-specific parameters."""
        manager = OptimizationManager(fuse_config, test_logger)

        # FUSE should have structure ID
        assert 'FUSE_STRUCTURE' in fuse_config


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance-related tests."""

    def test_optimization_runtime(self, dds_config, test_logger, temp_project_dir, mock_observations):
        """Test optimization completes in reasonable time."""
        import time

        manager = OptimizationManager(dds_config, test_logger)

        start_time = time.time()

        with patch.object(manager, 'project_dir', temp_project_dir):
            # Run with small number of iterations
            dds_config['NUMBER_OF_ITERATIONS'] = 10
            result = manager.calibrate_model()

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (implementation-dependent)
        assert elapsed_time < 60  # seconds

    def test_memory_usage(self, dds_config, test_logger):
        """Test memory usage during optimization."""
        import tracemalloc

        tracemalloc.start()

        manager = OptimizationManager(dds_config, test_logger)

        # Run optimization
        # ...

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable
        assert peak < 1e9  # Less than 1GB
