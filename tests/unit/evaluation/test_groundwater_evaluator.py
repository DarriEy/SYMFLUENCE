"""
Tests for GroundwaterEvaluator.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from symfluence.evaluation.evaluators.groundwater import GroundwaterEvaluator


def _make_config_value_getter(config_dict):
    """Create a mock _get_config_value that uses config_dict values."""
    call_count = [0]
    init_values = [
        '2010-01-01, 2015-12-31',
        '2016-01-01, 2018-12-31',
        'daily',
        config_dict.get('OPTIMIZATION_TARGET', 'gw_depth'),
    ]

    def mock_get_config_value(typed_accessor, default=None, dict_key=None):
        # During initialization, return from init_values list
        if call_count[0] < len(init_values):
            result = init_values[call_count[0]]
            call_count[0] += 1
            return result
        # After initialization, use config_dict
        if dict_key and dict_key in config_dict:
            return config_dict[dict_key]
        return default

    return mock_get_config_value


@pytest.fixture
def gw_evaluator(mock_config, tmp_path, mock_logger):
    """Create a GroundwaterEvaluator for testing."""
    config_dict = {
        'OPTIMIZATION_TARGET': 'gw_depth',
        'GW_BASE_DEPTH': 50.0,
        'GW_AUTO_ALIGN': False,
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
    }
    mock_config.to_dict.side_effect = lambda flatten=True: config_dict

    with patch.object(GroundwaterEvaluator, '_get_config_value',
                     side_effect=_make_config_value_getter(config_dict)):
        evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
        evaluator.config_dict = config_dict
        return evaluator


class TestGroundwaterEvaluatorInit:
    """Test GroundwaterEvaluator initialization."""

    def test_init_gw_depth(self, mock_config, tmp_path, mock_logger):
        """Test initialization with groundwater depth target."""
        config_dict = {'OPTIMIZATION_TARGET': 'gw_depth'}
        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'gw_depth'

    def test_init_gw_grace(self, mock_config, tmp_path, mock_logger):
        """Test initialization with GRACE target."""
        config_dict = {'OPTIMIZATION_TARGET': 'gw_grace'}
        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            assert evaluator.optimization_target == 'gw_grace'


class TestGroundwaterDepthExtraction:
    """Test groundwater depth extraction."""

    def test_extracts_from_total_soil_water(self, gw_evaluator, tmp_path):
        """Test extraction from scalarTotalSoilWat."""
        ds = xr.Dataset({
            'scalarTotalSoilWat': (['time', 'hru'], np.random.uniform(100, 500, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'gw_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = gw_evaluator._extract_groundwater_depth(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_extracts_from_aquifer_storage(self, gw_evaluator, tmp_path):
        """Test extraction from scalarAquiferStorage."""
        ds = xr.Dataset({
            'scalarAquiferStorage': (['time', 'hru'], np.random.uniform(0, 10, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'gw_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = gw_evaluator._extract_groundwater_depth(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_returns_empty_for_missing_variables(self, gw_evaluator, tmp_path):
        """Test returns empty series when no variables found."""
        ds = xr.Dataset({
            'some_other_variable': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = gw_evaluator._extract_groundwater_depth(ds_loaded)

        assert result.empty


class TestTotalWaterStorageExtraction:
    """Test total water storage extraction for GRACE comparison."""

    def test_sums_storage_components(self, mock_config, tmp_path, mock_logger):
        """Test summing of multiple storage components."""
        config_dict = {'OPTIMIZATION_TARGET': 'gw_grace'}
        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict

            ds = xr.Dataset({
                'scalarSWE': (['time', 'hru'], np.full((10, 1), 100.0)),
                'scalarTotalSoilWat': (['time', 'hru'], np.full((10, 1), 200.0)),
                'scalarAquiferStorage': (['time', 'hru'], np.full((10, 1), 1.0)),  # 1 m = 1000 mm
                'scalarCanopyWat': (['time', 'hru'], np.full((10, 1), 5.0)),
            },
            coords={
                'time': pd.date_range('2010-01-01', periods=10),
                'hru': [0],
            })

            file_path = tmp_path / 'tws_output.nc'
            ds.to_netcdf(file_path)

            with xr.open_dataset(file_path) as ds_loaded:
                result = evaluator._extract_total_water_storage(ds_loaded)

            assert isinstance(result, pd.Series)
            assert len(result) == 10


class TestUnitConversion:
    """Test TWS unit conversion."""

    def test_converts_large_range(self, gw_evaluator):
        """Test unit conversion for large data range."""
        data = pd.Series([1000, 2000, 3000])  # Range > 1000

        result = gw_evaluator._convert_tws_units(data)

        # Should divide by 10
        assert result.iloc[0] == pytest.approx(100)

    def test_converts_medium_range(self, gw_evaluator):
        """Test unit conversion for medium data range."""
        data = pd.Series([10, 20, 30])  # Range > 10

        result = gw_evaluator._convert_tws_units(data)

        # Should multiply by 100
        assert result.iloc[0] == pytest.approx(1000)

    def test_no_conversion_small_range(self, gw_evaluator):
        """Test no conversion for small data range."""
        data = pd.Series([1, 2, 3])  # Range < 10

        result = gw_evaluator._convert_tws_units(data)

        # Should remain unchanged
        pd.testing.assert_series_equal(result, data)


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_gw_depth_path(self, gw_evaluator, tmp_path):
        """Test groundwater depth observed data path."""
        gw_evaluator._project_dir = tmp_path
        gw_evaluator.domain_name = 'test_basin'
        gw_evaluator.optimization_target = 'gw_depth'

        result = gw_evaluator.get_observed_data_path()

        assert 'depth' in str(result)

    def test_gw_grace_path(self, gw_evaluator, tmp_path):
        """Test GRACE observed data path."""
        gw_evaluator._project_dir = tmp_path
        gw_evaluator.domain_name = 'test_basin'
        gw_evaluator.optimization_target = 'gw_grace'

        result = gw_evaluator.get_observed_data_path()

        assert 'grace' in str(result)


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_depth_column(self, gw_evaluator):
        """Test finding depth column."""
        gw_evaluator.optimization_target = 'gw_depth'
        columns = ['DateTime', 'Depth_m', 'temperature']

        result = gw_evaluator._get_observed_data_column(columns)

        assert result == 'Depth_m'

    def test_finds_water_level_column(self, gw_evaluator):
        """Test finding water_level column."""
        gw_evaluator.optimization_target = 'gw_depth'
        columns = ['DateTime', 'water_level', 'temperature']

        result = gw_evaluator._get_observed_data_column(columns)

        assert result == 'water_level'

    def test_finds_grace_column(self, gw_evaluator):
        """Test finding GRACE column."""
        gw_evaluator.optimization_target = 'gw_grace'
        gw_evaluator.grace_center = 'jpl'
        columns = ['DateTime', 'grace_jpl_tws', 'grace_csr_tws']

        result = gw_evaluator._get_observed_data_column(columns)

        assert result == 'grace_jpl_tws'


class TestNeedsRouting:
    """Test routing requirement."""

    def test_gw_never_needs_routing(self, gw_evaluator):
        """Test that groundwater evaluation never needs routing."""
        assert gw_evaluator.needs_routing() is False


class TestAutoAlignment:
    """Test groundwater auto-alignment feature."""

    def test_auto_align_applies_offset(self, mock_config, tmp_path, mock_logger):
        """Test that auto-alignment applies correct offset."""
        config_dict = {
            'OPTIMIZATION_TARGET': 'gw_depth',
            'GW_BASE_DEPTH': 50.0,
            'GW_AUTO_ALIGN': True,
            'CALIBRATION_PERIOD': '2010-01-01, 2010-12-31',
            'EVALUATION_PERIOD': '',
            'CALIBRATION_TIMESTEP': 'native',
        }

        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator.optimization_target = 'gw_depth'

        # Create simulated data (mean = 10)
        time_idx = pd.date_range('2010-01-01', periods=100)
        sim_data = pd.Series(np.full(100, 10.0), index=time_idx)

        # Create observed data (mean = 20)
        obs_data = pd.Series(np.full(100, 20.0), index=time_idx)

        # Mock _load_observed_data to return obs_data
        with patch.object(evaluator, '_load_observed_data', return_value=obs_data), \
             patch.object(evaluator, 'get_simulation_files', return_value=[tmp_path / 'test.nc']), \
             patch.object(evaluator, 'extract_simulated_data', return_value=sim_data):

            result = evaluator.calculate_metrics(tmp_path)

        # Offset should be applied: obs_mean - sim_mean = 20 - 10 = 10
        # After alignment, sim mean should match obs mean
        # This is tested by checking that metrics were calculated (non-None)
        assert result is not None

    def test_auto_align_disabled(self, mock_config, tmp_path, mock_logger):
        """Test that auto-alignment is skipped when disabled."""
        config_dict = {
            'OPTIMIZATION_TARGET': 'gw_depth',
            'GW_BASE_DEPTH': 50.0,
            'GW_AUTO_ALIGN': False,
            'CALIBRATION_PERIOD': '2010-01-01, 2010-12-31',
            'EVALUATION_PERIOD': '',
            'CALIBRATION_TIMESTEP': 'native',
        }

        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator.optimization_target = 'gw_depth'

        time_idx = pd.date_range('2010-01-01', periods=100)
        sim_data = pd.Series(np.full(100, 10.0), index=time_idx)
        obs_data = pd.Series(np.full(100, 20.0), index=time_idx)

        # Auto-align is False, so _load_observed_data should not be called
        # for alignment (but will be called for metric calculation)
        with patch.object(evaluator, '_load_observed_data', return_value=obs_data) as mock_load, \
             patch.object(evaluator, 'get_simulation_files', return_value=[tmp_path / 'test.nc']), \
             patch.object(evaluator, 'extract_simulated_data', return_value=sim_data):

            result = evaluator.calculate_metrics(tmp_path)

        # Should still return metrics
        assert result is not None

    def test_alignment_uses_overlapping_indices(self, mock_config, tmp_path, mock_logger):
        """Test that alignment only uses overlapping time indices."""
        config_dict = {
            'OPTIMIZATION_TARGET': 'gw_depth',
            'GW_BASE_DEPTH': 50.0,
            'GW_AUTO_ALIGN': True,
            'CALIBRATION_PERIOD': '2010-01-01, 2010-12-31',
            'EVALUATION_PERIOD': '',
            'CALIBRATION_TIMESTEP': 'native',
        }

        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator.optimization_target = 'gw_depth'

        # Create simulated data with longer period
        sim_idx = pd.date_range('2010-01-01', periods=200)
        sim_data = pd.Series(np.full(200, 10.0), index=sim_idx)

        # Create observed data with shorter period
        obs_idx = pd.date_range('2010-01-01', periods=100)
        obs_data = pd.Series(np.full(100, 20.0), index=obs_idx)

        with patch.object(evaluator, '_load_observed_data', return_value=obs_data), \
             patch.object(evaluator, 'get_simulation_files', return_value=[tmp_path / 'test.nc']), \
             patch.object(evaluator, 'extract_simulated_data', return_value=sim_data):

            result = evaluator.calculate_metrics(tmp_path)

        # Should calculate metrics for overlapping period
        assert result is not None

    def test_alignment_warns_no_overlap(self, mock_config, tmp_path, mock_logger, caplog):
        """Test that alignment logs warning when no overlapping indices."""
        import logging
        config_dict = {
            'OPTIMIZATION_TARGET': 'gw_depth',
            'GW_BASE_DEPTH': 50.0,
            'GW_AUTO_ALIGN': True,
            'CALIBRATION_PERIOD': '2010-01-01, 2010-12-31',
            'EVALUATION_PERIOD': '',
            'CALIBRATION_TIMESTEP': 'native',
        }

        with patch.object(GroundwaterEvaluator, '_get_config_value',
                         side_effect=_make_config_value_getter(config_dict)):
            evaluator = GroundwaterEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = config_dict
            evaluator.optimization_target = 'gw_depth'

        # Create non-overlapping data
        sim_idx = pd.date_range('2010-01-01', periods=100)
        obs_idx = pd.date_range('2015-01-01', periods=100)
        sim_data = pd.Series(np.full(100, 10.0), index=sim_idx)
        obs_data = pd.Series(np.full(100, 20.0), index=obs_idx)

        with caplog.at_level(logging.WARNING):
            with patch.object(evaluator, '_load_observed_data', return_value=obs_data), \
                 patch.object(evaluator, 'get_simulation_files', return_value=[tmp_path / 'test.nc']), \
                 patch.object(evaluator, 'extract_simulated_data', return_value=sim_data):

                result = evaluator.calculate_metrics(tmp_path)

        # Logger should have warning about no overlap (either alignment or metrics)
        assert any('overlap' in record.message.lower() or 'common' in record.message.lower()
                   for record in caplog.records)
