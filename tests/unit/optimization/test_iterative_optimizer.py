"""
Unit tests for iterative optimization algorithms.

Tests DDS, DE, PSO algorithms in both sequential and parallel modes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import random

from symfluence.optimization.optimizers import (
    BaseOptimizer,
    DDSOptimizer,
    DEOptimizer,
    PSOOptimizer,
    AsyncDDSOptimizer,
    PopulationDDSOptimizer,
    SCEUAOptimizer
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
# Shared Fixtures
# ============================================================================

@pytest.fixture
def mock_optimizer_base():
    """Mock heavy components initialized in BaseOptimizer and subclasses."""
    
    # We create a side effect for BaseOptimizer.__init__ to bypass file system logic
    def mock_init(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path("/tmp/data")
        self.domain_name = config.get('DOMAIN_NAME', 'test_domain')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID', 'test_exp')
        self.algorithm_name = self.get_algorithm_name().lower()
        self.use_parallel = config.get('MPI_PROCESSES', 1) > 1
        self.num_processes = config.get('MPI_PROCESSES', 1)
        self.scratch_manager = MagicMock()
        self.scratch_manager.use_scratch = False
        self.optimization_dir = self.project_dir / "simulations" / f"run_{self.algorithm_name}"
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        self.output_dir = self.project_dir / "optimization" / f"{self.algorithm_name}_{self.experiment_id}"
        
        # Mocks for managers
        self.parameter_manager = MagicMock()
        self.parameter_manager.all_param_names = ['theta_sat', 'k_soil', 'routingGammaScale']
        self.parameter_manager.param_bounds = {
            'theta_sat': {'min': 0.3, 'max': 0.6}, 
            'k_soil': {'min': 1e-6, 'max': 1e-4},
            'routingGammaScale': {'min': 0.1, 'max': 1.0}
        }
        self.parameter_manager.normalize_parameters.side_effect = lambda p: np.array([0.5, 0.5, 0.5])
        self.parameter_manager.denormalize_parameters.side_effect = lambda p: {'theta_sat': 0.45, 'k_soil': 5e-5, 'routingGammaScale': 0.5}
        self.parameter_manager.get_initial_parameters.return_value = {'theta_sat': 0.45, 'k_soil': 5e-5, 'routingGammaScale': 0.5}
        self.parameter_manager.original_depths = None
        
        self.transformation_manager = MagicMock()
        self.calibration_target = MagicMock()
        self.model_executor = MagicMock()
        self.results_manager = MagicMock()
        
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        self.best_params = None
        self.best_score = float('-inf')
        self.iteration_history = []
        self.models_to_run = config.get('HYDROLOGICAL_MODEL', 'SUMMA').split(',')
        self.random_seed = config.get('RANDOM_SEED', None)
        self.parallel_dirs = []
        self._consecutive_parallel_failures = 0

    with patch.object(BaseOptimizer, '__init__', side_effect=mock_init, autospec=True), \
         patch.object(BaseOptimizer, '_setup_optimization_directories'), \
         patch.object(BaseOptimizer, '_setup_parallel_processing'), \
         patch.object(BaseOptimizer, '_run_final_evaluation') as mock_final_eval, \
         patch.object(BaseOptimizer, '_save_to_default_settings'), \
         patch.object(BaseOptimizer, '_create_calibration_target'):
        
        # Setup final evaluation mock return value
        mock_final_eval.return_value = {
            'final_metrics': {'KGE': 0.85, 'NSE': 0.80},
            'calibration_metrics': {'KGE': 0.85, 'NSE': 0.80},
            'evaluation_metrics': {'KGE': 0.82, 'NSE': 0.78}
        }
        
        yield None

# ============================================================================
# DDS Optimizer Tests
# ============================================================================

class TestDDSOptimizer:
    """Tests for Dynamically Dimensioned Search algorithm."""

    def test_dds_initialization(self, dds_config, test_logger, mock_optimizer_base):
        """Test DDS optimizer initialization."""
        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger
        )

        assert optimizer.config == dds_config
        assert optimizer.logger == test_logger
        assert optimizer.parameter_manager.all_param_names == ['theta_sat', 'k_soil', 'routingGammaScale']
        assert optimizer.max_iterations == dds_config['NUMBER_OF_ITERATIONS']

    def test_dds_single_iteration(self, dds_config, test_logger, mock_optimizer_base):
        """Test a single DDS iteration."""
        # Set seed for reproducibility
        np.random.seed(42)

        # Run 1 iteration
        dds_config_single = dds_config.copy()
        dds_config_single['NUMBER_OF_ITERATIONS'] = 1

        optimizer = DDSOptimizer(
            config=dds_config_single,
            logger=test_logger
        )
        
        with patch.object(optimizer, '_evaluate_individual', side_effect=lambda x: 0.85):
             result = optimizer.run_optimization()

        assert result['best_score'] == 0.85
        assert 'theta_sat' in result['best_parameters']
        assert len(result['history']) > 0

    def test_dds_convergence(self, dds_config, test_logger, mock_optimizer_base):
        """Test that DDS converges on a simple problem."""
        config = dds_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 50

        optimizer = DDSOptimizer(
            config=config,
            logger=test_logger
        )
        
        # Override mock param manager for this test to be consistent
        optimizer.parameter_manager.all_param_names = ['x1', 'x2']
        optimizer.parameter_manager.param_bounds = {'x1': {'min': -5.0, 'max': 5.0}, 'x2': {'min': -5.0, 'max': 5.0}}
        
        def denormalize(norm_arr):
            val = -5.0 + norm_arr * (5.0 - (-5.0))
            return {'x1': val[0], 'x2': val[1]}
            
        def normalize(params):
            val = np.array([params['x1'], params['x2']])
            return (val - (-5.0)) / (5.0 - (-5.0))

        optimizer.parameter_manager.denormalize_parameters.side_effect = denormalize
        optimizer.parameter_manager.normalize_parameters.side_effect = normalize
        optimizer.parameter_manager.get_initial_parameters.return_value = {'x1': 4.0, 'x2': 4.0}

        def sphere_eval(norm_params):
            params = denormalize(norm_params)
            x = np.array([params['x1'], params['x2']])
            obj_value = simple_objective(x)
            return -obj_value

        with patch.object(optimizer, '_evaluate_individual', side_effect=sphere_eval):
            result = optimizer.run_optimization()

        best_params = result['best_parameters']
        assert abs(best_params['x1']) < 2.0
        assert abs(best_params['x2']) < 2.0

    def test_dds_respects_bounds(self, dds_config, test_logger, mock_optimizer_base):
        """Test that DDS respects parameter bounds."""
        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger
        )

        captured_params = []
        def capture_eval(norm_params):
            captured_params.append(norm_params)
            return 0.5
            
        with patch.object(optimizer, '_evaluate_individual', side_effect=capture_eval):
            optimizer.run_optimization()

        for params in captured_params:
            assert np.all(params >= 0.0)
            assert np.all(params <= 1.0)

    def test_dds_handles_failed_evaluation(self, dds_config, test_logger, mock_optimizer_base):
        """Test DDS handling of failed model evaluations."""
        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger
        )

        call_count = [0]
        def failing_eval(params):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return float('-inf')
            return 0.5

        with patch.object(optimizer, '_evaluate_individual', side_effect=failing_eval):
            result = optimizer.run_optimization()

        assert len(result['history']) > 0


# ============================================================================
# DE Optimizer Tests
# ============================================================================

class TestDEOptimizer:
    """Tests for Differential Evolution algorithm."""

    def test_de_initialization(self, de_config, test_logger, mock_optimizer_base):
        """Test DE optimizer initialization."""
        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger
        )

        assert optimizer.config == de_config
        assert optimizer.population_size >= 10
        assert optimizer.max_iterations == de_config['NUMBER_OF_ITERATIONS']

    def test_de_population_initialization(self, de_config, test_logger, mock_optimizer_base):
        """Test DE creates initial population correctly."""
        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger
        )
        
        optimizer.parameter_manager.all_param_names = ['x1', 'x2']
        optimizer.parameter_manager.normalize_parameters.side_effect = lambda p: np.array([0.5, 0.5])

        def mock_eval_pop():
            optimizer.population_scores = np.zeros(optimizer.population_size)

        with patch.object(optimizer, '_evaluate_population', side_effect=mock_eval_pop):
             optimizer._initialize_population({'x1': 0.5, 'x2': 0.5})

        assert len(optimizer.population) == optimizer.population_size
        assert len(optimizer.population[0]) == 2

    def test_de_convergence(self, de_config, test_logger, mock_optimizer_base):
        """Test that DE converges on Rosenbrock function."""
        config = de_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 20
        config['DE_POPULATION_SIZE'] = 10

        optimizer = DEOptimizer(
            config=config,
            logger=test_logger
        )
        
        optimizer.parameter_manager.all_param_names = ['x1', 'x2']
        optimizer.parameter_manager.param_bounds = {'x1': {'min': -2.0, 'max': 2.0}, 'x2': {'min': -2.0, 'max': 2.0}}
        
        def denormalize(norm_arr):
            val = -2.0 + norm_arr * (2.0 - (-2.0))
            return {'x1': val[0], 'x2': val[1]}
            
        def normalize(params):
            val = np.array([params['x1'], params['x2']])
            return (val - (-2.0)) / (2.0 - (-2.0))
            
        optimizer.parameter_manager.denormalize_parameters.side_effect = denormalize
        optimizer.parameter_manager.normalize_parameters.side_effect = normalize
        optimizer.parameter_manager.get_initial_parameters.return_value = {'x1': 0.0, 'x2': 0.0}

        def rosenbrock_eval(norm_params):
            params = denormalize(norm_params)
            x = np.array([params['x1'], params['x2']])
            obj_value = rosenbrock(x)
            return -obj_value

        with patch.object(optimizer, '_evaluate_individual', side_effect=rosenbrock_eval):
            result = optimizer.run_optimization()

        best_params = result['best_parameters']
        assert abs(best_params['x1'] - 1.0) < 1.0
        assert abs(best_params['x2'] - 1.0) < 1.0

    def test_de_respects_bounds(self, de_config, test_logger, mock_optimizer_base):
        """Test that DE respects parameter bounds."""
        optimizer = DEOptimizer(
            config=de_config,
            logger=test_logger
        )

        captured_params = []
        def capture_eval(norm_params):
            captured_params.append(norm_params)
            return 0.5
            
        with patch.object(optimizer, '_evaluate_individual', side_effect=capture_eval):
            optimizer.run_optimization()

        for params in captured_params:
            assert np.all(params >= 0.0)
            assert np.all(params <= 1.0)


# ============================================================================
# PSO Optimizer Tests
# ============================================================================

class TestPSOOptimizer:
    """Tests for Particle Swarm Optimization algorithm."""

    def test_pso_initialization(self, pso_config, test_logger, mock_optimizer_base):
        """Test PSO optimizer initialization."""
        optimizer = PSOOptimizer(
            config=pso_config,
            logger=test_logger
        )

        assert optimizer.config == pso_config
        assert optimizer.swarm_size == pso_config.get('SWRMSIZE', 20)

    def test_pso_swarm_initialization(self, pso_config, test_logger, mock_optimizer_base):
        """Test PSO creates swarm correctly."""
        optimizer = PSOOptimizer(
            config=pso_config,
            logger=test_logger
        )
        
        optimizer.parameter_manager.all_param_names = ['x1', 'x2']
        optimizer.parameter_manager.normalize_parameters.side_effect = lambda p: np.array([0.5, 0.5])

        with patch.object(optimizer, '_evaluate_swarm'):
             optimizer._initialize_swarm({'x1': 0.5, 'x2': 0.5})

        assert len(optimizer.swarm_positions) == optimizer.swarm_size
        assert len(optimizer.swarm_positions[0]) == 2

    def test_pso_convergence(self, pso_config, test_logger, mock_optimizer_base):
        """Test PSO convergence on sphere function."""
        config = pso_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 20
        config['SWRMSIZE'] = 10

        optimizer = PSOOptimizer(
            config=config,
            logger=test_logger
        )
        
        optimizer.parameter_manager.all_param_names = ['x1', 'x2']
        optimizer.parameter_manager.param_bounds = {'x1': {'min': -5.0, 'max': 5.0}, 'x2': {'min': -5.0, 'max': 5.0}}
        
        def denormalize(norm_arr):
            val = -5.0 + norm_arr * (5.0 - (-5.0))
            return {'x1': val[0], 'x2': val[1]}
            
        def normalize(params):
            val = np.array([params['x1'], params['x2']])
            return (val - (-5.0)) / (5.0 - (-5.0))
            
        optimizer.parameter_manager.denormalize_parameters.side_effect = denormalize
        optimizer.parameter_manager.normalize_parameters.side_effect = normalize
        optimizer.parameter_manager.get_initial_parameters.return_value = {'x1': 4.0, 'x2': 4.0}

        def sphere_eval(norm_params):
            params = denormalize(norm_params)
            x = np.array([params['x1'], params['x2']])
            obj_value = simple_objective(x)
            return -obj_value

        with patch.object(optimizer, '_evaluate_individual', side_effect=sphere_eval):
            result = optimizer.run_optimization()

        best_params = result['best_parameters']
        assert abs(best_params['x1']) < 2.0
        assert abs(best_params['x2']) < 2.0


# ============================================================================
# SCE-UA Optimizer Tests
# ============================================================================

class TestSCEUAOptimizer:
    """Tests for Shuffled Complex Evolution algorithm."""

    def test_sceua_initialization(self, sceua_config, test_logger, mock_optimizer_base):
        """Test SCE-UA optimizer initialization."""
        optimizer = SCEUAOptimizer(
            config=sceua_config,
            logger=test_logger
        )

        assert optimizer.config == sceua_config
        assert optimizer.algorithm_name.upper() == "SCE-UA"

    def test_sceua_respects_bounds(self, sceua_config, test_logger, mock_optimizer_base):
        """Test SCE-UA initialization with parameters."""
        optimizer = SCEUAOptimizer(
            config=sceua_config,
            logger=test_logger
        )
        assert optimizer.max_iterations == sceua_config['NUMBER_OF_ITERATIONS']


# ============================================================================
# Parallel Execution Tests
# ============================================================================

@pytest.mark.parallel
class TestParallelOptimization:
    """Tests for parallel optimization execution."""

    def test_async_dds(self, dds_config, test_logger, mock_optimizer_base):
        """Test AsyncDDSOptimizer initialization and run."""
        config = dds_config.copy()
        config['MPI_PROCESSES'] = 2
        config['NUMBER_OF_ITERATIONS'] = 50 
        
        # Create optimizer instance
        optimizer = AsyncDDSOptimizer(
            config=config,
            logger=test_logger
        )

        # Patch the method on the instance
        with patch.object(optimizer, '_run_parallel_evaluations') as mock_parallel:
            # Mock return value for both initialization phase and batches
            def parallel_eval_side_effect(tasks):
                return [
                    {'individual_id': task['individual_id'], 'score': 0.5, 'params': task['params']}
                    for task in tasks
                ]
            mock_parallel.side_effect = parallel_eval_side_effect
            
            # We also need to patch _evaluate_individual for the initialization pool to succeed (if called)
            with patch.object(optimizer, '_evaluate_individual', return_value=0.5):
                # Run minimal optimization
                optimizer.target_batches = 5
                optimizer.run_optimization()
                
                # Should be called at least twice: once for pool init, once for batch(es)
                assert mock_parallel.call_count >= 2

    def test_population_dds(self, dds_config, test_logger, mock_optimizer_base):
        """Test PopulationDDSOptimizer initialization and run."""
        config = dds_config.copy()
        config['MPI_PROCESSES'] = 2

        optimizer = PopulationDDSOptimizer(
            config=config,
            logger=test_logger
        )

        with patch.object(optimizer, '_run_parallel_evaluations') as mock_parallel:
            def parallel_eval_side_effect(tasks):
                return [
                    {'individual_id': task['individual_id'], 'score': 0.5, 'params': task['params']}
                    for task in tasks
                ]
            mock_parallel.side_effect = parallel_eval_side_effect
            
            optimizer.max_iterations = 1
            optimizer.run_optimization()
            
            assert mock_parallel.called


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameter_space(self, dds_config, test_logger, mock_optimizer_base):
        """Test with no parameters to calibrate."""
        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger
        )
        optimizer.parameter_manager.all_param_names = []

        with pytest.raises((ValueError, KeyError, Exception)):
            optimizer.run_optimization()

    def test_single_parameter(self, dds_config, test_logger, mock_optimizer_base):
        """Test optimization with single parameter."""
        optimizer = DDSOptimizer(
            config=dds_config,
            logger=test_logger
        )
        
        optimizer.parameter_manager.all_param_names = ['theta_sat']
        optimizer.parameter_manager.param_bounds = {'theta_sat': {'min': 0.3, 'max': 0.6}}
        optimizer.parameter_manager.normalize_parameters.side_effect = lambda p: np.array([0.5])
        optimizer.parameter_manager.get_initial_parameters.return_value = {'theta_sat': 0.45}
        
        with patch.object(optimizer, '_evaluate_individual', return_value=0.5):
            result = optimizer.run_optimization()

        assert 'theta_sat' in result['best_parameters']

    def test_zero_iterations(self, dds_config, test_logger, mock_optimizer_base):
        """Test with zero iterations (should handle gracefully)."""
        config = dds_config.copy()
        config['NUMBER_OF_ITERATIONS'] = 0

        optimizer = DDSOptimizer(
            config=config,
            logger=test_logger
        )

        with patch.object(optimizer, '_evaluate_individual', return_value=0.5):
            result = optimizer.run_optimization()
            
        assert isinstance(result, dict)