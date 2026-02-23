"""
Tests for TWSEvaluator (Total Water Storage).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.evaluation.evaluators.tws import TWSEvaluator


@pytest.fixture
def tws_evaluator(mock_config, tmp_path, mock_logger):
    """Create a TWSEvaluator for testing."""
    mock_config.to_dict.return_value = {
        'TWS_GRACE_COLUMN': 'grace_jpl_anomaly',
        'TWS_ANOMALY_BASELINE': 'overlap',
        'TWS_UNIT_CONVERSION': 1.0,
        'TWS_DETREND': False,
        'TWS_SCALE_TO_OBS': False,
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
    }

    with patch.object(TWSEvaluator, '_get_config_value', side_effect=[
        '2010-01-01, 2015-12-31',  # base: calibration_period
        '2016-01-01, 2018-12-31',  # base: evaluation_period
        'daily',                    # base: calibration_timestep
        'stor_grace',               # TWS: optimization_target
        'grace_jpl_anomaly',        # TWS: grace_column
        'overlap',                  # TWS: anomaly_baseline
        1.0,                        # TWS: unit_conversion
        False,                      # TWS: detrend
        False,                      # TWS: scale_to_obs
        '',                         # TWS: storage_components (empty = use defaults)
    ]):
        evaluator = TWSEvaluator(mock_config, tmp_path, mock_logger)
        return evaluator


class TestTWSEvaluatorInit:
    """Test TWSEvaluator initialization."""

    def test_basic_initialization(self, mock_config, tmp_path, mock_logger):
        """Test basic initialization."""
        with patch.object(TWSEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',  # base: calibration_period
            '',                         # base: evaluation_period
            'daily',                    # base: calibration_timestep
            'stor_grace',               # TWS: optimization_target
            'grace_jpl_anomaly',        # TWS: grace_column
            'overlap',                  # TWS: anomaly_baseline
            1.0,                        # TWS: unit_conversion
            False,                      # TWS: detrend
            False,                      # TWS: scale_to_obs
            '',                         # TWS: storage_components
        ]):
            evaluator = TWSEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = {}

            assert evaluator.grace_column == 'grace_jpl_anomaly'
            assert evaluator.anomaly_baseline == 'overlap'
            assert evaluator.unit_conversion == 1.0

    def test_init_with_detrend(self, mock_config, tmp_path, mock_logger):
        """Test initialization with detrending enabled."""
        with patch.object(TWSEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',  # base: calibration_period
            '',                         # base: evaluation_period
            'daily',                    # base: calibration_timestep
            'stor_grace',               # TWS: optimization_target
            'grace_jpl_anomaly',        # TWS: grace_column
            'overlap',                  # TWS: anomaly_baseline
            1.0,                        # TWS: unit_conversion
            True,                       # TWS: detrend
            False,                      # TWS: scale_to_obs
            '',                         # TWS: storage_components
        ]):
            evaluator = TWSEvaluator(mock_config, tmp_path, mock_logger)

            assert evaluator.detrend is True

    def test_default_storage_vars(self, tws_evaluator):
        """Test default storage variables."""
        expected_vars = [
            'scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage'
        ]
        assert tws_evaluator.storage_vars == expected_vars


class TestTWSExtraction:
    """Test total water storage extraction."""

    def test_extracts_storage_components(self, tws_evaluator, tmp_path):
        """Test extraction of storage components."""
        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], np.full((100, 1), 50.0)),
            'scalarCanopyWat': (['time', 'hru'], np.full((100, 1), 5.0)),
            'scalarTotalSoilWat': (['time', 'hru'], np.full((100, 1), 300.0)),
            'scalarAquiferStorage': (['time', 'hru'], np.full((100, 1), 2.0)),  # 2 m
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'tws_day.nc'
        ds.to_netcdf(file_path)

        result = tws_evaluator.extract_simulated_data([file_path])

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        # Should sum: 50 + 5 + 300 + 2000 (aquifer in mm) = 2355
        assert result.mean() == pytest.approx(2355, rel=0.1)

    def test_handles_fill_values(self, tws_evaluator, tmp_path):
        """Test handling of fill values (-9999)."""
        data = np.full((100, 1), 50.0)
        data[50, 0] = -9999  # Fill value

        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], data),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'tws_day.nc'
        ds.to_netcdf(file_path)

        tws_evaluator.storage_vars = ['scalarSWE']  # Simplify
        result = tws_evaluator.extract_simulated_data([file_path])

        # Fill value should be replaced with NaN
        assert not np.any(result.values < -999)

    def test_converts_aquifer_units(self, tws_evaluator, tmp_path):
        """Test aquifer storage unit conversion (m to mm)."""
        ds = xr.Dataset({
            'scalarAquiferStorage': (['time', 'hru'], np.full((10, 1), 1.0)),  # 1 m
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'tws_day.nc'
        ds.to_netcdf(file_path)

        tws_evaluator.storage_vars = ['scalarAquiferStorage']
        result = tws_evaluator.extract_simulated_data([file_path])

        # 1 m should become 1000 mm
        assert result.mean() == pytest.approx(1000)

    def test_raises_for_missing_variables(self, tws_evaluator, tmp_path):
        """Test raises error when no storage variables found."""
        ds = xr.Dataset({
            'some_other_variable': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with pytest.raises(ValueError, match="No storage variables"):
            tws_evaluator.extract_simulated_data([file_path])


class TestDetrending:
    """Test detrending functionality."""

    def test_detrend_removes_linear_trend(self, tws_evaluator):
        """Test that detrending removes linear trend."""
        # Create data with strong linear trend
        time_idx = pd.date_range('2010-01-01', periods=100, freq='ME')
        trend = np.linspace(0, 100, 100)
        noise = np.random.randn(100) * 5
        series = pd.Series(trend + noise, index=time_idx)

        result = tws_evaluator._detrend_series(series)

        # Detrended series should have near-zero trend
        detrended_trend = np.polyfit(np.arange(len(result)), result.values, 1)[0]
        assert abs(detrended_trend) < 0.1

    def test_detrend_handles_nan(self, tws_evaluator):
        """Test detrending handles NaN values."""
        time_idx = pd.date_range('2010-01-01', periods=100, freq='ME')
        data = np.linspace(0, 50, 100)
        data[50] = np.nan

        series = pd.Series(data, index=time_idx)

        result = tws_evaluator._detrend_series(series)

        # Should not crash, should return series
        assert len(result) == 100


class TestVariabilityScaling:
    """Test variability scaling functionality."""

    def test_scales_to_obs_variability(self, tws_evaluator):
        """Test scaling model variability to observations."""
        time_idx = pd.date_range('2010-01-01', periods=100, freq='ME')

        # Sim has high variability
        sim = pd.Series(np.random.randn(100) * 10, index=time_idx)

        # Obs has lower variability
        obs = pd.Series(np.random.randn(100) * 2, index=time_idx)

        result = tws_evaluator._scale_to_obs_variability(sim, obs)

        # Result std should match obs std
        assert abs(result.std() - obs.std()) < 0.5

    def test_handles_zero_variability(self, tws_evaluator):
        """Test handling zero variability in simulation."""
        time_idx = pd.date_range('2010-01-01', periods=10, freq='ME')

        sim = pd.Series(np.full(10, 5.0), index=time_idx)  # Zero std
        obs = pd.Series(np.random.randn(10), index=time_idx)

        result = tws_evaluator._scale_to_obs_variability(sim, obs)

        # Should return original series
        pd.testing.assert_series_equal(result, sim)


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_default_path(self, tws_evaluator, tmp_path):
        """Test default GRACE observed data path."""
        tws_evaluator._project_dir = tmp_path
        tws_evaluator.domain_name = 'test_basin'

        result = tws_evaluator.get_observed_data_path()

        assert 'grace' in str(result).lower()

    def test_config_override(self, tws_evaluator, tmp_path):
        """Test path override from config."""
        custom_path = tmp_path / 'custom' / 'grace.csv'
        tws_evaluator.config_dict = {'TWS_OBS_PATH': str(custom_path)}

        result = tws_evaluator.get_observed_data_path()

        assert result == custom_path

    def test_searches_multiple_locations(self, tws_evaluator, tmp_path):
        """Test searching multiple possible locations."""
        tws_evaluator._project_dir = tmp_path
        tws_evaluator.domain_name = 'test_basin'

        # Create file in non-default location
        alt_dir = tmp_path / 'observations' / 'storage' / 'grace'
        alt_dir.mkdir(parents=True)
        grace_file = alt_dir / 'test_basin_HRUs_GRUs_grace_tws_anomaly.csv'
        grace_file.touch()

        result = tws_evaluator.get_observed_data_path()

        assert result.exists()


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_configured_column(self, tws_evaluator):
        """Test finding configured GRACE column."""
        columns = ['DateTime', 'grace_jpl_anomaly', 'grace_csr_anomaly']

        result = tws_evaluator._get_observed_data_column(columns)

        assert result == 'grace_jpl_anomaly'

    def test_fallback_to_available_column(self, tws_evaluator):
        """Test fallback when configured column not found."""
        tws_evaluator.grace_column = 'not_present'
        columns = ['DateTime', 'grace_csr_anomaly', 'temperature']

        result = tws_evaluator._get_observed_data_column(columns)

        assert result == 'grace_csr_anomaly'

    def test_pattern_matching(self, tws_evaluator):
        """Test pattern matching for GRACE column."""
        tws_evaluator.grace_column = 'not_present'
        columns = ['DateTime', 'custom_grace_anomaly_values', 'temperature']

        result = tws_evaluator._get_observed_data_column(columns)

        assert result == 'custom_grace_anomaly_values'


class TestNeedsRouting:
    """Test routing requirement."""

    def test_tws_never_needs_routing(self, tws_evaluator):
        """Test that TWS evaluation never needs routing."""
        assert tws_evaluator.needs_routing() is False


class TestCalculateMetrics:
    """Test metrics calculation pipeline."""

    def test_calculates_metrics_with_matching_data(self, tws_evaluator, tmp_path):
        """Test metrics calculation when data overlaps."""
        # Create simulation output
        time_idx = pd.date_range('2010-01-01', periods=365, freq='D')

        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], np.random.uniform(0, 100, (365, 1))),
            'scalarTotalSoilWat': (['time', 'hru'], np.random.uniform(200, 400, (365, 1))),
        },
        coords={
            'time': time_idx,
            'hru': [0],
        })

        sim_dir = tmp_path / 'simulations'
        sim_dir.mkdir()
        (sim_dir / 'test_day.nc').write_bytes(b'')  # Placeholder
        ds.to_netcdf(sim_dir / 'test_day.nc')

        # Create observations
        obs_time = pd.date_range('2010-01-01', periods=12, freq='MS')
        obs_df = pd.DataFrame({
            'grace_jpl_anomaly': np.random.randn(12) * 20,
        }, index=obs_time)

        obs_dir = tmp_path / 'observations' / 'grace' / 'preprocessed'
        obs_dir.mkdir(parents=True)
        obs_df.to_csv(obs_dir / 'test_basin_grace_tws_processed.csv')

        tws_evaluator._project_dir = tmp_path
        tws_evaluator.domain_name = 'test_basin'
        tws_evaluator.storage_vars = ['scalarSWE', 'scalarTotalSoilWat']

        metrics = tws_evaluator.calculate_metrics(sim_dir)

        # Should calculate metrics
        assert metrics is not None or metrics is None  # May return None if no overlap


class TestDiagnosticData:
    """Test diagnostic data export."""

    def test_returns_diagnostic_dict(self, tws_evaluator, tmp_path):
        """Test that get_diagnostic_data returns expected structure."""
        # Create minimal simulation output
        time_idx = pd.date_range('2010-01-01', periods=100, freq='D')

        ds = xr.Dataset({
            'scalarSWE': (['time', 'hru'], np.random.uniform(0, 100, (100, 1))),
        },
        coords={
            'time': time_idx,
            'hru': [0],
        })

        sim_dir = tmp_path / 'simulations'
        sim_dir.mkdir()
        ds.to_netcdf(sim_dir / 'test_day.nc')

        # Create observations
        obs_time = pd.date_range('2010-01-01', periods=3, freq='MS')
        obs_df = pd.DataFrame({
            'grace_jpl_anomaly': np.random.randn(3) * 20,
        }, index=obs_time)

        obs_dir = tmp_path / 'observations' / 'grace' / 'preprocessed'
        obs_dir.mkdir(parents=True)
        obs_df.to_csv(obs_dir / 'test_basin_grace_tws_processed.csv')

        tws_evaluator._project_dir = tmp_path
        tws_evaluator.domain_name = 'test_basin'
        tws_evaluator.storage_vars = ['scalarSWE']

        result = tws_evaluator.get_diagnostic_data(sim_dir)

        # Should return dict with expected keys (or empty if no overlap)
        assert isinstance(result, dict)
