"""
Shared fixtures for evaluation suite tests.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return logging.getLogger('test_evaluation')


@pytest.fixture
def mock_config():
    """Create a mock SymfluenceConfig for testing evaluators."""
    config = MagicMock()

    # Domain configuration
    config.domain.name = 'test_domain'
    config.domain.calibration_period = '2010-01-01, 2015-12-31'
    config.domain.evaluation_period = '2016-01-01, 2018-12-31'
    config.domain.definition_method = 'lumped'

    # Optimization configuration
    config.optimization.target = 'streamflow'
    config.optimization.calibration_timestep = 'daily'

    # Evaluation configuration
    config.evaluation.fluxnet.station = 'US-Ton'
    config.evaluation.evaluation_variable = None
    config.evaluation.smap.surface_depth_m = 0.05
    config.evaluation.smap.rootzone_depth_m = 1.0
    config.evaluation.ismn.target_depth_m = 0.1
    config.evaluation.ismn.temporal_aggregation = 'daily_mean'

    # ET evaluator optional fields (must be None for dict fallback in tests)
    config.evaluation.et.obs_source = None
    config.evaluation.et.obs_path = None
    config.evaluation.et.temporal_aggregation = None
    config.evaluation.et.use_quality_control = None
    config.evaluation.et.max_quality_flag = None
    config.evaluation.et.modis_max_qc = None
    config.evaluation.et.gleam_max_relative_uncertainty = None

    # TWS evaluator optional fields
    config.evaluation.tws.grace_column = None
    config.evaluation.tws.anomaly_baseline = None
    config.evaluation.tws.unit_conversion = None
    config.evaluation.tws.detrend = None
    config.evaluation.tws.scale_to_obs = None
    config.evaluation.tws.storage_components = None
    config.evaluation.tws.obs_path = None

    # Make .get() return default for unknown keys (matches SymfluenceConfig behavior)
    config.get = MagicMock(side_effect=lambda key, default=None: default)

    # Mock to_dict() to return a proper dict (supports flatten parameter)
    config_dict_values = {
        'DOMAIN_NAME': 'test_domain',
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'OPTIMIZATION_TARGET': 'streamflow',
        'CALIBRATION_TIMESTEP': 'daily',
        'CALIBRATION_VARIABLE': 'streamflow',
        'ET_OBS_SOURCE': 'mod16',
        'SM_TARGET_DEPTH': 'auto',
        'SMAP_LAYER': 'surface_sm',
        'GW_BASE_DEPTH': 50.0,
        'GW_AUTO_ALIGN': True,
        'TWS_GRACE_COLUMN': 'grace_jpl_anomaly',
        'TWS_DETREND': False,
    }
    config.to_dict.side_effect = lambda flatten=True: config_dict_values

    return config


@pytest.fixture
def mock_config_dict():
    """Create a mock configuration dictionary for testing."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'OPTIMIZATION_TARGET': 'streamflow',
        'CALIBRATION_TIMESTEP': 'daily',
        'CALIBRATION_VARIABLE': 'streamflow',
        'ET_OBS_SOURCE': 'mod16',
        'SM_TARGET_DEPTH': 'auto',
        'SMAP_LAYER': 'surface_sm',
        'GW_BASE_DEPTH': 50.0,
        'GW_AUTO_ALIGN': True,
        'TWS_GRACE_COLUMN': 'grace_jpl_anomaly',
        'TWS_DETREND': False,
    }


@pytest.fixture
def sample_time_index():
    """Create a sample time index for testing."""
    return pd.date_range('2010-01-01', periods=365, freq='D')


@pytest.fixture
def sample_observed_series(sample_time_index):
    """Create sample observed data series for testing."""
    np.random.seed(42)
    values = np.random.uniform(10, 100, len(sample_time_index))
    return pd.Series(values, index=sample_time_index, name='observed')


@pytest.fixture
def sample_simulated_series(sample_time_index):
    """Create sample simulated data series for testing."""
    np.random.seed(43)
    values = np.random.uniform(10, 100, len(sample_time_index))
    return pd.Series(values, index=sample_time_index, name='simulated')


@pytest.fixture
def synthetic_netcdf_file(tmp_path):
    """Create a synthetic NetCDF file for testing evaluators."""
    time = pd.date_range('2010-01-01', periods=365, freq='D')
    hru = [0]

    np.random.seed(42)

    ds = xr.Dataset({
        'scalarSWE': (['time', 'hru'], np.random.uniform(0, 100, (365, 1))),
        'scalarTotalET': (['time', 'hru'], np.random.uniform(-1e-6, 0, (365, 1))),
        'scalarLatHeatTotal': (['time', 'hru'], np.random.uniform(-100, 0, (365, 1))),
        'mLayerVolFracLiq': (['time', 'hru', 'midToto'], np.random.uniform(0.1, 0.4, (365, 1, 5))),
        'mLayerDepth': (['hru', 'midToto'], [[0.1, 0.2, 0.3, 0.5, 1.0]]),
        'scalarTotalSoilWat': (['time', 'hru'], np.random.uniform(100, 500, (365, 1))),
        'scalarAquiferStorage': (['time', 'hru'], np.random.uniform(0, 10, (365, 1))),
        'scalarCanopyWat': (['time', 'hru'], np.random.uniform(0, 5, (365, 1))),
        'averageRoutedRunoff': (['time', 'hru'], np.random.uniform(1e-9, 1e-7, (365, 1))),
    },
    coords={
        'time': time,
        'hru': hru,
        'midToto': range(5),
    })

    # Add attributes
    ds['scalarSWE'].attrs['units'] = 'kg m-2'
    ds['scalarTotalET'].attrs['units'] = 'kg m-2 s-1'
    ds['averageRoutedRunoff'].attrs['units'] = 'm s-1'

    file_path = tmp_path / 'test_output_day.nc'
    ds.to_netcdf(file_path)

    return file_path


@pytest.fixture
def synthetic_multi_hru_netcdf(tmp_path):
    """Create a synthetic NetCDF file with multiple HRUs for testing spatial aggregation."""
    time = pd.date_range('2010-01-01', periods=100, freq='D')
    hru = [0, 1, 2]  # Multiple HRUs

    np.random.seed(44)

    ds = xr.Dataset({
        'scalarSWE': (['time', 'hru'], np.random.uniform(0, 100, (100, 3))),
        'scalarTotalET': (['time', 'hru'], np.random.uniform(-1e-6, 0, (100, 3))),
        'averageRoutedRunoff': (['time', 'hru'], np.random.uniform(1e-9, 1e-7, (100, 3))),
    },
    coords={
        'time': time,
        'hru': hru,
    })

    file_path = tmp_path / 'test_multi_hru_day.nc'
    ds.to_netcdf(file_path)

    return file_path


@pytest.fixture
def sample_obs_csv(tmp_path):
    """Create a sample observation CSV file for testing."""
    time = pd.date_range('2010-01-01', periods=365, freq='D')
    np.random.seed(45)

    df = pd.DataFrame({
        'DateTime': time,
        'streamflow': np.random.uniform(10, 100, 365),
        'swe': np.random.uniform(0, 50, 365),
        'et_mm_day': np.random.uniform(0, 5, 365),
        'sm_0.1': np.random.uniform(0.1, 0.4, 365),
        'Depth_m': np.random.uniform(5, 15, 365),
        'grace_jpl_anomaly': np.random.uniform(-50, 50, 365),
    })

    file_path = tmp_path / 'test_obs_processed.csv'
    df.to_csv(file_path, index=False)

    return file_path


@pytest.fixture
def project_dir_structure(tmp_path):
    """Create a realistic project directory structure for testing."""
    project_dir = tmp_path / 'test_project'

    # Create directory structure
    dirs_to_create = [
        'observations/streamflow/preprocessed',
        'observations/snow/swe/processed',
        'observations/et/preprocessed',
        'observations/soil_moisture/smap/processed',
        'observations/groundwater/depth/processed',
        'observations/grace/preprocessed',
        'simulations/summa',
        'settings/SUMMA',
        'shapefiles/river_basins',
    ]

    for dir_path in dirs_to_create:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)

    return project_dir
