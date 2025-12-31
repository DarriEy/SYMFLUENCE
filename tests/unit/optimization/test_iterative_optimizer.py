"""
Unit tests for iterative optimization algorithms.

Tests DDS, DE, PSO, SCE-UA algorithms in both sequential and parallel modes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from symfluence.utils.optimization.iterative_optimizer import (
    DDSOptimizer,
    DEOptimizer,
    PSOOptimizer,
    SCEUAOptimizer,
    AsyncDDSOptimizer,
    PopulationDDSOptimizer,
)


pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# Helper Functions
# ============================================================================

def simple_objective(x):
    """Simple sphere function for testing (minimum at origin)."""
    return np.sum(x**2)


def rosenbrock(x):
    """Rosenbrock function (minimum at x=[1,1,...])."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# ============================================================================
# DDS Optimizer Tests
# ============================================================================

class TestDDSOptimizer:
    """Tests for Dynamically Dimensioned Search algorithm."""

    def test_dds_initialization(self, dds_config, test_logger, mock_evaluate_function):
        """Test DDS optimizer initialization."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        assert optimizer.config == dds_config
        assert optimizer.logger == test_logger
        assert len(optimizer.param_bounds) == 2
        assert optimizer.max_iterations == dds_config['NUMBER_OF_ITERATIONS']

    def test_dds_single_iteration(self, dds_config, test_logger, mock_evaluate_function):
        """Test a single DDS iteration."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        # Set seed for reproducibility
        np.random.seed(42)

        # Run 1 iteration
        dds_config_single = dds_config.copy()
        dds_config_single['NUMBER_OF_ITERATIONS'] = 1

        optimizer_single = DDSOptimizer(
            config=dds_config_single,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        # Mock the run method to just return initial results
        with patch.object(optimizer_single, '_evaluate_parameters', wraps=optimizer_single._evaluate_parameters):
            result = optimizer_single.run()

        assert 'best_params' in result
        assert 'best_metric' in result
        assert 'history' in result
        assert isinstance(result['best_params'], dict)

    def test_dds_convergence(self, dds_config, test_logger):
        """Test that DDS converges on a simple problem."""
        # Use sphere function (minimum at origin)
        def sphere_eval(params):
            x = np.array([params['x1'], params['x2']])
            obj_value = simple_objective(x)
            return {'KGE': -obj_value}  # Negative because we maximize KGE

        param_bounds = {'x1': (-5.0, 5.0), 'x2': (-5.0, 5.0)}

        # Run with more iterations for convergence
        config = dds_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 20

        optimizer = DDSOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=sphere_eval,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # Check convergence towards origin
        best_x1 = result['best_params']['x1']
        best_x2 = result['best_params']['x2']

        # Should be closer to 0 than the bounds
        assert abs(best_x1) < 2.0
        assert abs(best_x2) < 2.0

    def test_dds_respects_bounds(self, dds_config, test_logger, mock_evaluate_function):
        """Test that DDS respects parameter bounds."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # Check all parameters in history respect bounds
        for record in result['history']:
            params = record['params']
            assert 0.3 <= params['theta_sat'] <= 0.6
            assert 1e-6 <= params['k_soil'] <= 1e-4

    def test_dds_handles_failed_evaluation(self, dds_config, test_logger):
        """Test DDS handling of failed model evaluations."""
        param_bounds = {'theta_sat': (0.3, 0.6)}

        call_count = [0]

        def failing_eval(params):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                # Every other evaluation fails
                raise RuntimeError("Model failed")
            return {'KGE': 0.5}

        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger,
            evaluate_func=failing_eval,
            param_bounds=param_bounds
        )

        # Should handle failures gracefully
        result = optimizer.run()

        # Should still have some successful evaluations
        assert len(result['history']) > 0


# ============================================================================
# DE Optimizer Tests
# ============================================================================

class TestDEOptimizer:
    """Tests for Differential Evolution algorithm."""

    def test_de_initialization(self, de_config, test_logger, mock_evaluate_function):
        """Test DE optimizer initialization."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        assert optimizer.config == de_config
        assert optimizer.population_size == de_config.get('DE_POPULATION_SIZE', 10)
        assert optimizer.max_generations == de_config['NUMBER_OF_ITERATIONS']

    def test_de_population_initialization(self, de_config, test_logger, mock_evaluate_function):
        """Test DE creates initial population correctly."""
        param_bounds = {'x1': (0.0, 1.0), 'x2': (0.0, 1.0)}

        de_config['DE_POPULATION_SIZE'] = 5
        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        # Check population is created with correct size
        # This would require accessing internal state or mocking
        result = optimizer.run()

        # Population should be evaluated
        assert len(result['history']) >= de_config['DE_POPULATION_SIZE']

    def test_de_convergence(self, de_config, test_logger):
        """Test that DE converges on Rosenbrock function."""
        def rosenbrock_eval(params):
            x = np.array([params['x1'], params['x2']])
            obj_value = rosenbrock(x)
            return {'KGE': -obj_value}

        param_bounds = {'x1': (-2.0, 2.0), 'x2': (-2.0, 2.0)}

        config = de_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 10
        config['DE_POPULATION_SIZE'] = 10

        optimizer = DEOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=rosenbrock_eval,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # Should converge towards (1, 1)
        best_x1 = result['best_params']['x1']
        best_x2 = result['best_params']['x2']

        # Loose convergence check (Rosenbrock is hard)
        assert abs(best_x1 - 1.0) < 1.0
        assert abs(best_x2 - 1.0) < 1.0

    def test_de_respects_bounds(self, de_config, test_logger, mock_evaluate_function):
        """Test that DE respects parameter bounds."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # All parameters should respect bounds
        for record in result['history']:
            params = record['params']
            assert 0.3 <= params['theta_sat'] <= 0.6
            assert 1e-6 <= params['k_soil'] <= 1e-4


# ============================================================================
# PSO Optimizer Tests
# ============================================================================

class TestPSOOptimizer:
    """Tests for Particle Swarm Optimization algorithm."""

    def test_pso_initialization(self, pso_config, test_logger, mock_evaluate_function):
        """Test PSO optimizer initialization."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = PSOOptimizer(
            config=pso_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        assert optimizer.config == pso_config
        assert optimizer.swarm_size == pso_config.get('PSO_SWARM_SIZE', 10)

    def test_pso_swarm_initialization(self, pso_config, test_logger, mock_evaluate_function):
        """Test PSO creates swarm correctly."""
        param_bounds = {'x1': (0.0, 1.0), 'x2': (0.0, 1.0)}

        pso_config['PSO_SWARM_SIZE'] = 5
        optimizer = PSOOptimizer(
            config=pso_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # Swarm should be evaluated
        assert len(result['history']) >= pso_config['PSO_SWARM_SIZE']

    def test_pso_convergence(self, pso_config, test_logger):
        """Test PSO convergence on sphere function."""
        def sphere_eval(params):
            x = np.array([params['x1'], params['x2']])
            obj_value = simple_objective(x)
            return {'KGE': -obj_value}

        param_bounds = {'x1': (-5.0, 5.0), 'x2': (-5.0, 5.0)}

        config = pso_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 15
        config['PSO_SWARM_SIZE'] = 10

        optimizer = PSOOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=sphere_eval,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # Should converge towards origin
        best_x1 = result['best_params']['x1']
        best_x2 = result['best_params']['x2']

        assert abs(best_x1) < 2.0
        assert abs(best_x2) < 2.0


# ============================================================================
# SCE-UA Optimizer Tests
# ============================================================================

class TestSCEUAOptimizer:
    """Tests for Shuffled Complex Evolution algorithm."""

    def test_sceua_initialization(self, sceua_config, test_logger, mock_evaluate_function):
        """Test SCE-UA optimizer initialization."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = SCEUAOptimizer(
            config=sceua_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        assert optimizer.config == sceua_config
        assert optimizer.num_complexes == sceua_config.get('SCEUA_COMPLEXES', 2)

    def test_sceua_respects_bounds(self, sceua_config, test_logger, mock_evaluate_function):
        """Test SCE-UA respects parameter bounds."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        optimizer = SCEUAOptimizer(
            config=sceua_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        # All parameters should respect bounds
        for record in result['history']:
            params = record['params']
            assert 0.3 <= params['theta_sat'] <= 0.6
            assert 1e-6 <= params['k_soil'] <= 1e-4


# ============================================================================
# Parallel Execution Tests
# ============================================================================

@pytest.mark.parallel
class TestParallelOptimization:
    """Tests for parallel optimization execution."""

    def test_async_dds(self, dds_config, test_logger, mock_evaluate_function):
        """Test AsyncDDS optimizer."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        config = dds_config.copy()
        config['PARALLEL_WORKERS'] = 2

        optimizer = AsyncDDSOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        assert 'best_params' in result
        assert 'best_metric' in result
        assert 'history' in result

    def test_population_dds(self, dds_config, test_logger, mock_evaluate_function):
        """Test PopulationDDS optimizer."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        config = dds_config.copy()
        config['DDS_POPULATION_SIZE'] = 5

        optimizer = PopulationDDSOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        assert 'best_params' in result
        assert isinstance(result['history'], list)

    @pytest.mark.skip(reason="Requires multiprocessing setup")
    def test_de_parallel_evaluation(self, de_config, test_logger, mock_evaluate_function):
        """Test DE with parallel evaluation of population."""
        param_bounds = {'theta_sat': (0.3, 0.6), 'k_soil': (1e-6, 1e-4)}

        config = de_config.copy()
        config['PARALLEL_WORKERS'] = 2
        config['DE_POPULATION_SIZE'] = 6

        # This would require proper multiprocessing setup
        # Skipped for now as it needs careful mocking
        pass


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameter_space(self, dds_config, test_logger, mock_evaluate_function):
        """Test with no parameters to calibrate."""
        param_bounds = {}

        with pytest.raises((ValueError, KeyError, Exception)):
            optimizer = DDSOptimizer(
                config=dds_config,
                logger=test_logger,
                evaluate_func=mock_evaluate_function,
                param_bounds=param_bounds
            )
            optimizer.run()

    def test_single_parameter(self, dds_config, test_logger, mock_evaluate_function):
        """Test optimization with single parameter."""
        param_bounds = {'theta_sat': (0.3, 0.6)}

        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        result = optimizer.run()

        assert 'theta_sat' in result['best_params']
        assert 0.3 <= result['best_params']['theta_sat'] <= 0.6

    def test_zero_iterations(self, dds_config, test_logger, mock_evaluate_function):
        """Test with zero iterations (should handle gracefully)."""
        param_bounds = {'theta_sat': (0.3, 0.6)}

        config = dds_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 0

        optimizer = DDSOptimizer(
            config=config,
            logger=test_logger,
            evaluate_func=mock_evaluate_function,
            param_bounds=param_bounds
        )

        # Should either return initial evaluation or handle gracefully
        result = optimizer.run()
        assert isinstance(result, dict)
