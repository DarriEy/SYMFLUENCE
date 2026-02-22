"""
Shared fixtures for reporting unit tests.

Provides common test fixtures for plotters, processors, and the reporting manager.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime, timedelta


# ============================================================================
# Mock Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a basic mock configuration for unit tests."""
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/test',
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'RIVER_BASINS_NAME': 'default',
        'RIVER_NETWORK_SHP_NAME': 'default',
        'POUR_POINT_SHP_NAME': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'SIM_REACH_ID': 1,
        'OPTIMIZATION_METRIC': 'KGE',
        'OPTIMIZATION_TARGET': 'streamflow',
        'SPINUP_PERIOD': '1980-01-01,1981-01-01',
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for unit tests."""
    return MagicMock()


@pytest.fixture(scope="module")
def mock_plot_config():
    """Create a mock plot configuration."""
    from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG
    return DEFAULT_PLOT_CONFIG


# ============================================================================
# Time Series Data Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_dates():
    """Create sample date range for testing."""
    return pd.date_range('2020-01-01', periods=365, freq='D')


@pytest.fixture
def sample_obs_series(sample_dates):
    """Create sample observation time series."""
    np.random.seed(42)
    # Create realistic streamflow pattern with seasonal variation
    base = 10 + 5 * np.sin(np.linspace(0, 4 * np.pi, len(sample_dates)))
    noise = np.random.normal(0, 1, len(sample_dates))
    values = np.maximum(0.1, base + noise)
    return pd.Series(values, index=sample_dates, name='discharge_cms')


@pytest.fixture
def sample_sim_series(sample_obs_series):
    """Create sample simulated time series (correlated with observations)."""
    np.random.seed(43)
    # Simulated is observations with some bias and noise
    values = sample_obs_series.values * 0.95 + np.random.normal(0, 0.5, len(sample_obs_series))
    values = np.maximum(0.1, values)
    return pd.Series(values, index=sample_obs_series.index, name='SUMMA_discharge')


@pytest.fixture
def sample_results_df(sample_dates, sample_obs_series, sample_sim_series):
    """Create sample results DataFrame with multiple models."""
    np.random.seed(44)
    sim2 = sample_obs_series.values * 1.05 + np.random.normal(0, 0.8, len(sample_obs_series))

    df = pd.DataFrame({
        'obs_discharge': sample_obs_series.values,
        'SUMMA_discharge': sample_sim_series.values,
        'FUSE_discharge': np.maximum(0.1, sim2),
    }, index=sample_dates)
    return df


@pytest.fixture
def empty_results_df():
    """Create an empty results DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def results_df_no_discharge():
    """Create results DataFrame without discharge columns."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'temperature': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'precipitation': [0, 1, 0, 2, 0, 3, 0, 1, 0, 0],
    }, index=dates)


# ============================================================================
# Optimization History Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_optimization_history():
    """Create sample optimization history."""
    return [
        {'generation': 0, 'best_score': 0.5, 'best_params': {'k': 1.0, 'theta': 0.5}},
        {'generation': 1, 'best_score': 0.6, 'best_params': {'k': 1.1, 'theta': 0.6}},
        {'generation': 2, 'best_score': 0.7, 'best_params': {'k': 1.2, 'theta': 0.7}},
        {'generation': 3, 'best_score': 0.75, 'best_params': {'k': 1.3, 'theta': 0.75}},
        {'generation': 4, 'best_score': 0.8, 'best_params': {'k': 1.4, 'theta': 0.8}},
    ]


# ============================================================================
# Sensitivity Analysis Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_sensitivity_data():
    """Create sample sensitivity analysis data."""
    return pd.Series({
        'k': 0.45,
        'theta': 0.32,
        'n': 0.18,
        'alpha': 0.05,
    }, name='sensitivity')


@pytest.fixture(scope="module")
def sample_sensitivity_comparison():
    """Create sample sensitivity comparison data (multiple methods)."""
    return pd.DataFrame({
        'Sobol': [0.45, 0.32, 0.18, 0.05],
        'Morris': [0.42, 0.35, 0.15, 0.08],
        'FAST': [0.48, 0.30, 0.17, 0.05],
    }, index=['k', 'theta', 'n', 'alpha'])


# ============================================================================
# Model Output Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_model_outputs():
    """Create sample model outputs list."""
    return [
        ('SUMMA', '/path/to/summa_output.nc'),
        ('FUSE', '/path/to/fuse_output.nc'),
    ]


@pytest.fixture(scope="module")
def sample_obs_files():
    """Create sample observation files list."""
    return [
        ('USGS', '/path/to/usgs_obs.csv'),
    ]


# ============================================================================
# Plotter Mocking Utilities
# ============================================================================

@pytest.fixture
def mock_matplotlib_setup():
    """Fixture for mocking matplotlib setup in plotters."""
    def _create_mock():
        mock_plt = Mock()
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        return mock_plt, mock_fig, mock_ax
    return _create_mock


@pytest.fixture
def patch_plotter_methods():
    """Context manager for patching common plotter methods."""
    def _patch(plotter):
        mock_plt = Mock()
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        setup_patch = patch.object(plotter, '_setup_matplotlib', return_value=(mock_plt, None))
        save_patch = patch.object(plotter, '_save_and_close', return_value='/fake/path.png')

        return setup_patch, save_patch, mock_plt, mock_fig, mock_ax
    return _patch


# ============================================================================
# Benchmark Results Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_benchmark_results():
    """Create sample benchmark results."""
    return {
        'models': ['SUMMA', 'FUSE', 'HYPE'],
        'metrics': {
            'KGE': [0.75, 0.68, 0.72],
            'NSE': [0.70, 0.65, 0.68],
            'RMSE': [2.5, 3.1, 2.8],
        },
        'observations': 'USGS',
        'period': '2015-2020',
    }


# ============================================================================
# Decision Analysis Fixtures
# ============================================================================

@pytest.fixture
def sample_decision_results_df():
    """Create sample decision analysis results."""
    return pd.DataFrame({
        'Iteration': range(20),
        'soilDepth': ['shallow'] * 10 + ['deep'] * 10,
        'vegType': ['forest', 'grass'] * 10,
        'kge': np.random.uniform(0.5, 0.9, 20),
        'nse': np.random.uniform(0.4, 0.85, 20),
        'rmse': np.random.uniform(1, 5, 20),
    })


# ============================================================================
# Drop Analysis Fixtures
# ============================================================================

@pytest.fixture
def sample_drop_data():
    """Create sample drop analysis data."""
    return [
        {'threshold': 1000, 'total_drops': 150, 'max_drop': 50, 'mean_drop': 25},
        {'threshold': 2000, 'total_drops': 100, 'max_drop': 40, 'mean_drop': 20},
        {'threshold': 3000, 'total_drops': 75, 'max_drop': 35, 'mean_drop': 18},
        {'threshold': 4000, 'total_drops': 50, 'max_drop': 30, 'mean_drop': 15},
        {'threshold': 5000, 'total_drops': 30, 'max_drop': 25, 'mean_drop': 12},
    ]
