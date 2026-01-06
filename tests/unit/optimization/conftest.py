"""
Fixtures for optimization/calibration unit tests.

Provides mock configurations, data, and helpers for testing calibration
without actually running expensive model simulations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import logging


# ============================================================================
# Pytest markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "optimization: Optimization/calibration tests")
    config.addinivalue_line("markers", "parallel: Tests that use parallel execution")


# ============================================================================
# Logger fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_optimization')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def base_optimization_config(tmp_path):
    """Base configuration for optimization tests."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_catchment',
        'EXPERIMENT_ID': 'test_optimization',
        'HYDROLOGICAL_MODEL': 'SUMMA',

        # Optimization settings
        'OPTIMIZATION_METHODS': ['iteration'],
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'OPTIMIZATION_METRIC': 'KGE',
        'NUMBER_OF_ITERATIONS': 5,  # Small number for unit tests

        # Calibration period
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-31 23:00',
        'CALIBRATION_PERIOD': '2020-01-01, 2020-01-31',
        'CALIBRATION_TIMESTEP': 'daily',

        # Parameters to calibrate
        'PARAMS_TO_CALIBRATE': 'theta_sat,k_soil',
        'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaScale',

        # Paths
        'OBSERVATIONS_PATH': 'default',

        # Parallel settings
        'PARALLEL_WORKERS': 2,
        'USE_MPI': False,
    }


# ============================================================================
# Algorithm Configuration Fixtures (DDS, DE, PSO, SCE-UA)
# ============================================================================

# Algorithm-specific configuration overrides
_ALGORITHM_OVERRIDES = {
    'DDS': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'NUMBER_OF_ITERATIONS': 5,
    },
    'DE': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DE',
        'NUMBER_OF_ITERATIONS': 3,
        'DE_POPULATION_SIZE': 5,
    },
    'PSO': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'PSO',
        'NUMBER_OF_ITERATIONS': 3,
        'PSO_SWARM_SIZE': 5,
    },
    'SCE-UA': {
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'SCE-UA',
        'NUMBER_OF_ITERATIONS': 3,
        'SCEUA_COMPLEXES': 2,
    },
}


@pytest.fixture(params=['DDS', 'DE', 'PSO', 'SCE-UA'])
def algorithm_specific_config(request, base_optimization_config):
    """Parametrized fixture providing configurations for all optimization algorithms.
    
    Use this fixture to test all algorithm variants with a single test.
    Provides: dds_config, de_config, pso_config, sceua_config combinations.
    
    Example:
        def test_algorithm_initialization(algorithm_specific_config):
            # This test runs for DDS, DE, PSO, and SCE-UA
            config = algorithm_specific_config
            assert config['ITERATIVE_OPTIMIZATION_ALGORITHM'] in _ALGORITHM_OVERRIDES
    """
    algorithm = request.param
    config = base_optimization_config.copy()
    config.update(_ALGORITHM_OVERRIDES[algorithm])
    return config


# Convenience fixtures for individual algorithms (backward compatibility)
@pytest.fixture
def dds_config(base_optimization_config):
    """Configuration for DDS algorithm."""
    config = base_optimization_config.copy()
    config.update(_ALGORITHM_OVERRIDES['DDS'])
    return config


@pytest.fixture
def de_config(base_optimization_config):
    """Configuration for Differential Evolution algorithm."""
    config = base_optimization_config.copy()
    config.update(_ALGORITHM_OVERRIDES['DE'])
    return config


@pytest.fixture
def pso_config(base_optimization_config):
    """Configuration for PSO algorithm."""
    config = base_optimization_config.copy()
    config.update(_ALGORITHM_OVERRIDES['PSO'])
    return config


@pytest.fixture
def sceua_config(base_optimization_config):
    """Configuration for SCE-UA algorithm."""
    config = base_optimization_config.copy()
    config.update(_ALGORITHM_OVERRIDES['SCE-UA'])
    return config


# ============================================================================
# Model Configuration Fixtures (SUMMA, FUSE, NGEN)
# ============================================================================

# Model-specific configuration overrides
_MODEL_OVERRIDES = {
    'SUMMA': {
        'HYDROLOGICAL_MODEL': 'SUMMA',
    },
    'FUSE': {
        'HYDROLOGICAL_MODEL': 'FUSE',
        'FUSE_STRUCTURE': '902',
    },
    'NGEN': {
        'HYDROLOGICAL_MODEL': 'NGEN',
    },
}


@pytest.fixture(params=['SUMMA', 'FUSE', 'NGEN'])
def model_specific_config(request, base_optimization_config):
    """Parametrized fixture providing configurations for all hydrological models.
    
    Use this fixture to test all model variants with a single test.
    Provides: summa_config, fuse_config, ngen_config combinations.
    
    Example:
        def test_model_initialization(model_specific_config):
            # This test runs for SUMMA, FUSE, and NGEN
            config = model_specific_config
            assert config['HYDROLOGICAL_MODEL'] in _MODEL_OVERRIDES
    """
    model = request.param
    config = base_optimization_config.copy()
    config.update(_MODEL_OVERRIDES[model])
    return config


# Convenience fixtures for individual models (backward compatibility)
@pytest.fixture
def summa_config(base_optimization_config):
    """Configuration for SUMMA calibration."""
    config = base_optimization_config.copy()
    config.update(_MODEL_OVERRIDES['SUMMA'])
    return config


@pytest.fixture
def fuse_config(base_optimization_config):
    """Configuration for FUSE calibration."""
    config = base_optimization_config.copy()
    config.update(_MODEL_OVERRIDES['FUSE'])
    return config


@pytest.fixture
def ngen_config(base_optimization_config):
    """Configuration for NGEN calibration."""
    config = base_optimization_config.copy()
    config.update(_MODEL_OVERRIDES['NGEN'])
    return config


# ============================================================================
# Project directory fixtures
# ============================================================================

@pytest.fixture
def temp_project_dir(tmp_path, base_optimization_config):
    """Create a temporary project directory structure."""
    domain_name = base_optimization_config['DOMAIN_NAME']
    project_dir = tmp_path / f"domain_{domain_name}"

    # Create directory structure
    (project_dir / "optimization").mkdir(parents=True)
    (project_dir / "simulations").mkdir(parents=True)
    (project_dir / "settings" / "SUMMA").mkdir(parents=True)
    (project_dir / "observations" / "streamflow" / "preprocessed").mkdir(parents=True)
    (project_dir / "forcing" / "SUMMA_input").mkdir(parents=True)

    return project_dir


# ============================================================================
# Observation data fixtures
# ============================================================================

@pytest.fixture
def mock_observations(temp_project_dir, base_optimization_config):
    """Create mock streamflow observations."""
    domain_name = base_optimization_config['DOMAIN_NAME']
    obs_path = temp_project_dir / "observations" / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"

    # Create synthetic streamflow data
    start_date = pd.to_datetime('2020-01-01')
    dates = pd.date_range(start_date, periods=31, freq='D')

    # Synthetic flow with seasonal pattern
    baseflow = 5.0
    seasonal = 3.0 * np.sin(2 * np.pi * np.arange(31) / 31)
    noise = np.random.normal(0, 0.5, 31)
    flow = baseflow + seasonal + noise
    flow = np.maximum(flow, 0.1)  # Ensure positive

    obs_df = pd.DataFrame({
        'date': dates,
        'discharge_cms': flow
    })

    obs_df.to_csv(obs_path, index=False)
    return obs_path


@pytest.fixture
def mock_simulation_results():
    """Create mock simulation results (matching observations format)."""
    dates = pd.date_range('2020-01-01', periods=31, freq='D')

    # Simulation with some offset from "truth"
    baseflow = 4.5
    seasonal = 3.2 * np.sin(2 * np.pi * np.arange(31) / 31 + 0.1)
    noise = np.random.normal(0, 0.3, 31)
    flow = baseflow + seasonal + noise
    flow = np.maximum(flow, 0.1)

    return pd.DataFrame({
        'date': dates,
        'discharge_cms': flow
    })


# ============================================================================
# Parameter space fixtures
# ============================================================================

@pytest.fixture
def mock_parameter_bounds():
    """Mock parameter bounds for calibration."""
    return {
        'theta_sat': (0.3, 0.6),
        'k_soil': (1e-6, 1e-4),
        'routingGammaScale': (1.0, 10.0),
        'fieldCapacity': (0.1, 0.4),
    }


@pytest.fixture
def mock_initial_parameters():
    """Mock initial parameter values."""
    return {
        'theta_sat': 0.45,
        'k_soil': 5e-5,
        'routingGammaScale': 5.0,
        'fieldCapacity': 0.25,
    }


# ============================================================================
# Mock model evaluation functions
# ============================================================================

@pytest.fixture
def mock_evaluate_function():
    """Create a mock evaluation function that returns realistic metrics."""
    def evaluate(params):
        """
        Mock evaluation function.

        Returns metrics based on parameter values to simulate optimization.
        Better parameters (closer to optimal) give better KGE.
        """
        # Optimal parameters (ground truth for synthetic problem)
        optimal = {
            'theta_sat': 0.45,
            'k_soil': 5e-5,
            'routingGammaScale': 5.0,
        }

        # Calculate distance from optimal (normalized)
        distance = 0
        for key in optimal:
            if key in params:
                # Normalize based on typical ranges
                if key == 'k_soil':
                    norm_dist = abs(np.log10(params[key]) - np.log10(optimal[key])) / 2
                else:
                    norm_dist = abs(params[key] - optimal[key]) / optimal[key]
                distance += norm_dist ** 2

        distance = np.sqrt(distance)

        # Convert distance to KGE (better parameters = higher KGE)
        kge = 1.0 - distance

        # Add some noise to make it realistic
        kge += np.random.normal(0, 0.05)

        return {
            'KGE': kge,
            'NSE': kge - 0.1,  # NSE typically slightly lower
            'RMSE': distance * 10,
            'MAE': distance * 8,
        }

    return evaluate


def _create_mock_worker(model_name):
    """Factory function to create mock worker functions for any model.
    
    Args:
        model_name: Name of the model (SUMMA, FUSE, NGEN, etc.)
    
    Returns:
        A worker function that simulates model evaluation.
    """
    def worker_factory(mock_evaluate_function):
        def worker(params, config, trial_num=0):
            """Simulate model run."""
            metrics = mock_evaluate_function(params)
            return {
                'trial': trial_num,
                'params': params,
                'metrics': metrics,
                'success': True
            }
        return worker
    return worker_factory


@pytest.fixture
def mock_summa_worker(mock_evaluate_function):
    """Mock SUMMA worker function."""
    return _create_mock_worker('SUMMA')(mock_evaluate_function)


@pytest.fixture
def mock_fuse_worker(mock_evaluate_function):
    """Mock FUSE worker function."""
    return _create_mock_worker('FUSE')(mock_evaluate_function)


@pytest.fixture
def mock_ngen_worker(mock_evaluate_function):
    """Mock NGEN worker function."""
    return _create_mock_worker('NGEN')(mock_evaluate_function)


# ============================================================================
# Algorithm-specific fixtures
# ============================================================================

@pytest.fixture
def algorithm_configs():
    """Dictionary of all algorithm configurations for parametrized tests."""
    return {
        'DDS': {'name': 'DDS', 'iterations': 5},
        'DE': {'name': 'DE', 'iterations': 3, 'population': 5},
        'PSO': {'name': 'PSO', 'iterations': 3, 'swarm_size': 5},
        'SCE-UA': {'name': 'SCE-UA', 'iterations': 3, 'complexes': 2},
    }


@pytest.fixture
def model_configs():
    """Dictionary of all model configurations for parametrized tests."""
    return {
        'SUMMA': {'name': 'SUMMA', 'params': ['theta_sat', 'k_soil']},
        'FUSE': {'name': 'FUSE', 'params': ['theta_sat', 'k_soil'], 'structure': '902'},
        'NGEN': {'name': 'NGEN', 'params': ['theta_sat', 'k_soil']},
    }


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup any temporary files after each test."""
    yield
    # Cleanup happens here if needed
