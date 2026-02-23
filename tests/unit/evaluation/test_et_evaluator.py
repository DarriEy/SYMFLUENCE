"""
Tests for ETEvaluator (Evapotranspiration).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from symfluence.evaluation.evaluators.et import ETEvaluator


@pytest.fixture
def et_evaluator(mock_config, tmp_path, mock_logger):
    """Create an ETEvaluator for testing."""
    # Set up mock config to return proper dict
    mock_config.to_dict.return_value = {
        'OPTIMIZATION_TARGET': 'et',
        'ET_OBS_SOURCE': 'mod16',
        'CALIBRATION_PERIOD': '2010-01-01, 2015-12-31',
        'EVALUATION_PERIOD': '2016-01-01, 2018-12-31',
        'CALIBRATION_TIMESTEP': 'daily',
    }

    with patch.object(ETEvaluator, '_get_config_value', side_effect=[
        '2010-01-01, 2015-12-31',  # base: calibration_period
        '2016-01-01, 2018-12-31',  # base: evaluation_period
        'daily',                    # base: calibration_timestep
        'et',                       # ET: optimization_target
        'mod16',                    # ET: obs_source
        'daily_mean',               # ET: temporal_aggregation
        True,                       # ET: use_quality_control
        2,                          # ET: max_quality_flag
    ]):
        evaluator = ETEvaluator(mock_config, tmp_path, mock_logger)
        evaluator.optimization_target = 'et'
        return evaluator


class TestETEvaluatorInit:
    """Test ETEvaluator initialization."""

    def test_basic_initialization_et(self, mock_config, tmp_path, mock_logger):
        """Test initialization with ET target."""
        with patch.object(ETEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',  # base: calibration_period
            '',                         # base: evaluation_period
            'daily',                    # base: calibration_timestep
            'et',                       # ET: optimization_target
            'mod16',                    # ET: obs_source
            'daily_mean',               # ET: temporal_aggregation
            True,                       # ET: use_quality_control
            2,                          # ET: max_quality_flag
        ]):
            evaluator = ETEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = {'OPTIMIZATION_TARGET': 'et'}

            assert evaluator.optimization_target == 'et'

    def test_initialization_with_latent_heat(self, mock_config, tmp_path, mock_logger):
        """Test initialization with latent heat target."""
        with patch.object(ETEvaluator, '_get_config_value', side_effect=[
            '2010-01-01, 2015-12-31',  # base: calibration_period
            '',                         # base: evaluation_period
            'daily',                    # base: calibration_timestep
            'latent_heat',              # ET: optimization_target
            'mod16',                    # ET: obs_source
            'daily_mean',               # ET: temporal_aggregation
            True,                       # ET: use_quality_control
            2,                          # ET: max_quality_flag
        ]):
            evaluator = ETEvaluator(mock_config, tmp_path, mock_logger)
            evaluator.config_dict = {'OPTIMIZATION_TARGET': 'latent_heat'}

            assert evaluator.optimization_target == 'latent_heat'


class TestETExtraction:
    """Test ET data extraction."""

    def test_extracts_total_et(self, et_evaluator, tmp_path):
        """Test extraction of scalarTotalET variable."""
        # SUMMA outputs ET as negative (water leaving surface)
        ds = xr.Dataset({
            'scalarTotalET': (['time', 'hru'], np.random.uniform(-1e-6, 0, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })
        ds['scalarTotalET'].attrs['units'] = 'kg m-2 s-1'

        file_path = tmp_path / 'et_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = et_evaluator._extract_total_et(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        # After sign flip, should be positive
        assert result.mean() >= 0

    def test_extracts_with_spatial_collapse(self, et_evaluator, tmp_path):
        """Test ET extraction uses _collapse_spatial_dims."""
        ds = xr.Dataset({
            'scalarTotalET': (['time', 'hru'], np.random.uniform(-1e-6, 0, (100, 3))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0, 1, 2],
        })

        file_path = tmp_path / 'et_multi_hru.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = et_evaluator._extract_total_et(ds_loaded)

        # Should have collapsed HRU dimension
        assert len(result) == 100


class TestLatentHeatExtraction:
    """Test latent heat extraction."""

    def test_extracts_latent_heat(self, et_evaluator, tmp_path):
        """Test extraction of scalarLatHeatTotal."""
        et_evaluator.optimization_target = 'latent_heat'

        ds = xr.Dataset({
            'scalarLatHeatTotal': (['time', 'hru'], np.random.uniform(-200, 0, (100, 1))),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=100),
            'hru': [0],
        })

        file_path = tmp_path / 'lh_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            result = et_evaluator._extract_latent_heat(ds_loaded)

        assert isinstance(result, pd.Series)
        assert len(result) == 100
        # After sign flip, should be positive
        assert result.mean() >= 0

    def test_raises_for_missing_latent_heat(self, et_evaluator, tmp_path):
        """Test raises error when latent heat variable not found."""
        ds = xr.Dataset({
            'some_variable': (['time', 'hru'], np.random.rand(10, 1)),
        },
        coords={
            'time': pd.date_range('2010-01-01', periods=10),
            'hru': [0],
        })

        file_path = tmp_path / 'empty_output.nc'
        ds.to_netcdf(file_path)

        with xr.open_dataset(file_path) as ds_loaded:
            with pytest.raises(ValueError, match="scalarLatHeatTotal not found"):
                et_evaluator._extract_latent_heat(ds_loaded)


class TestUnitConversion:
    """Test ET unit conversion."""

    def test_convert_kg_m2_s_to_mm_day(self, et_evaluator):
        """Test conversion from kg m-2 s-1 to mm/day."""
        # 1 kg/m²/s = 86400 mm/day (1 day = 86400 seconds)
        data = pd.Series([1e-6])  # Small value in kg/m²/s

        result = et_evaluator._convert_et_units(data, from_unit='kg_m2_s', to_unit='mm_day')

        # 1e-6 kg/m²/s * 86400 s/day = 0.0864 mm/day
        assert result.iloc[0] == pytest.approx(0.0864, rel=0.01)

    def test_convert_mm_day_to_kg_m2_s(self, et_evaluator):
        """Test conversion from mm/day to kg m-2 s-1."""
        data = pd.Series([1.0])  # 1 mm/day

        result = et_evaluator._convert_et_units(data, from_unit='mm_day', to_unit='kg_m2_s')

        # 1 mm/day / 86400 ≈ 1.16e-5 kg/m²/s
        assert result.iloc[0] == pytest.approx(1.0 / 86400, rel=0.01)


class TestObservedDataPath:
    """Test observed data path resolution."""

    def test_mod16_path(self, et_evaluator, tmp_path):
        """Test MOD16 observed data path."""
        et_evaluator._project_dir = tmp_path
        et_evaluator.domain_name = 'test_basin'
        et_evaluator.config_dict = {'ET_OBS_SOURCE': 'mod16'}

        result = et_evaluator.get_observed_data_path()

        assert 'mod16' in str(result).lower() or 'et' in str(result).lower()

    def test_fluxnet_path(self, et_evaluator, tmp_path):
        """Test FLUXNET observed data path."""
        et_evaluator._project_dir = tmp_path
        et_evaluator.domain_name = 'test_basin'
        et_evaluator.config_dict = {'ET_OBS_SOURCE': 'fluxnet'}

        result = et_evaluator.get_observed_data_path()

        assert 'fluxnet' in str(result).lower() or 'et' in str(result).lower()


class TestObservedDataColumn:
    """Test observed data column selection."""

    def test_finds_et_column(self, et_evaluator):
        """Test finding ET column."""
        columns = ['DateTime', 'et_mm_day', 'temperature']

        result = et_evaluator._get_observed_data_column(columns)

        assert result == 'et_mm_day'

    def test_finds_latent_heat_column(self, et_evaluator):
        """Test finding latent heat column."""
        et_evaluator.optimization_target = 'latent_heat'
        columns = ['DateTime', 'LE_W_m2', 'H_W_m2']

        result = et_evaluator._get_observed_data_column(columns)

        assert result == 'LE_W_m2'

    def test_finds_fluxnet_le_column(self, et_evaluator):
        """Test finding FLUXNET LE_F_MDS column."""
        et_evaluator.optimization_target = 'latent_heat'
        columns = ['DateTime', 'LE_F_MDS', 'H_F_MDS']

        result = et_evaluator._get_observed_data_column(columns)

        assert result == 'LE_F_MDS'


class TestNeedsRouting:
    """Test routing requirement."""

    def test_et_never_needs_routing(self, et_evaluator):
        """Test that ET evaluation never needs routing."""
        assert et_evaluator.needs_routing() is False


class TestQualityControl:
    """Test source-specific quality control methods."""

    def test_apply_quality_control_dispatches_to_fluxnet(self, et_evaluator):
        """Test QC dispatches to FluxNet method when source is fluxnet."""
        et_evaluator.obs_source = 'fluxnet'
        et_evaluator.use_quality_control = True

        obs_df = pd.DataFrame({
            'LE_F_MDS_QC': [0, 1, 2, 3],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        with patch.object(et_evaluator, '_apply_fluxnet_qc', return_value=obs_data) as mock:
            et_evaluator._apply_quality_control(obs_df, obs_data, 'ET')
            mock.assert_called_once()

    def test_apply_quality_control_dispatches_to_modis(self, et_evaluator):
        """Test QC dispatches to MODIS method when source is mod16."""
        et_evaluator.obs_source = 'mod16'
        et_evaluator.use_quality_control = True

        obs_df = pd.DataFrame({
            'ET_QC': [0, 1, 2, 3],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        with patch.object(et_evaluator, '_apply_modis_qc', return_value=obs_data) as mock:
            et_evaluator._apply_quality_control(obs_df, obs_data, 'ET')
            mock.assert_called_once()

    def test_apply_quality_control_dispatches_to_gleam(self, et_evaluator):
        """Test QC dispatches to GLEAM method when source is gleam."""
        et_evaluator.obs_source = 'gleam'
        et_evaluator.use_quality_control = True

        obs_df = pd.DataFrame({
            'E_uncertainty': [0.1, 0.2, 0.3, 0.4],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        with patch.object(et_evaluator, '_apply_gleam_qc', return_value=obs_data) as mock:
            et_evaluator._apply_quality_control(obs_df, obs_data, 'ET')
            mock.assert_called_once()

    def test_apply_quality_control_skips_when_disabled(self, et_evaluator):
        """Test QC is skipped when use_quality_control is False."""
        et_evaluator.use_quality_control = False
        et_evaluator.obs_source = 'fluxnet'

        obs_df = pd.DataFrame({
            'LE_F_MDS_QC': [0, 3, 3, 3],  # Most would be filtered
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_quality_control(obs_df, obs_data, 'ET')

        # All data should be returned unchanged
        pd.testing.assert_series_equal(result, obs_data)


class TestFluxNetQC:
    """Test FluxNet-specific quality control."""

    def test_fluxnet_qc_filters_high_qc_values(self, et_evaluator):
        """Test FluxNet QC filters observations with QC > threshold."""
        et_evaluator.optimization_target = 'et'
        et_evaluator.max_quality_flag = 1

        obs_df = pd.DataFrame({
            'LE_F_MDS_QC': [0, 1, 2, 3],  # 0 and 1 should pass
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_fluxnet_qc(obs_df, obs_data)

        assert len(result) == 2
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 2.0

    def test_fluxnet_qc_returns_unchanged_without_qc_column(self, et_evaluator):
        """Test FluxNet QC returns unchanged when QC column not found."""
        et_evaluator.optimization_target = 'et'

        obs_df = pd.DataFrame({
            'other_column': [1, 2, 3, 4],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_fluxnet_qc(obs_df, obs_data)

        pd.testing.assert_series_equal(result, obs_data)


class TestModisQC:
    """Test MODIS-specific quality control."""

    def test_modis_qc_filters_by_modland_bits(self, et_evaluator):
        """Test MODIS QC extracts MODLAND bits (0-1) for filtering."""
        et_evaluator.config_dict = {'ET_MODIS_MAX_QC': 0}  # Only best quality

        # QC values: 0 (00), 1 (01), 2 (10), 3 (11)
        obs_df = pd.DataFrame({
            'ET_QC': [0, 1, 2, 3],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_modis_qc(obs_df, obs_data)

        # Only QC=0 should pass
        assert len(result) == 1
        assert result.iloc[0] == 1.0

    def test_modis_qc_allows_higher_threshold(self, et_evaluator):
        """Test MODIS QC with higher threshold allows more data."""
        et_evaluator.config_dict = {'ET_MODIS_MAX_QC': 1}

        obs_df = pd.DataFrame({
            'ET_QC': [0, 1, 2, 3],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_modis_qc(obs_df, obs_data)

        # QC=0 and QC=1 should pass
        assert len(result) == 2

    def test_modis_qc_extracts_bits_correctly(self, et_evaluator):
        """Test MODIS QC correctly extracts MODLAND bits from higher values."""
        et_evaluator.config_dict = {'ET_MODIS_MAX_QC': 0}

        # QC value 4 (binary 100) has MODLAND bits = 00, should pass
        # QC value 5 (binary 101) has MODLAND bits = 01, should fail
        obs_df = pd.DataFrame({
            'ET_QC': [4, 5],  # 4 & 0b11 = 0, 5 & 0b11 = 1
        }, index=pd.date_range('2020-01-01', periods=2))
        obs_data = pd.Series([1.0, 2.0], index=obs_df.index)

        result = et_evaluator._apply_modis_qc(obs_df, obs_data)

        assert len(result) == 1
        assert result.iloc[0] == 1.0


class TestGleamQC:
    """Test GLEAM-specific quality control."""

    def test_gleam_qc_filters_by_relative_uncertainty(self, et_evaluator):
        """Test GLEAM QC filters by relative uncertainty threshold."""
        et_evaluator.config_dict = {'ET_GLEAM_MAX_RELATIVE_UNCERTAINTY': 0.3}

        # Relative uncertainty = uncertainty / abs(value)
        # Values: [10, 10, 10, 10], Uncertainty: [1, 3, 4, 6]
        # Relative: [0.1, 0.3, 0.4, 0.6] -> first 2 pass (<=0.3)
        obs_df = pd.DataFrame({
            'E_uncertainty': [1.0, 3.0, 4.0, 6.0],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([10.0, 10.0, 10.0, 10.0], index=obs_df.index)

        result = et_evaluator._apply_gleam_qc(obs_df, obs_data)

        assert len(result) == 2

    def test_gleam_qc_uses_default_threshold(self, et_evaluator):
        """Test GLEAM QC uses default 0.5 threshold."""
        et_evaluator.config_dict = {}  # No threshold set

        obs_df = pd.DataFrame({
            'E_uncertainty': [0.4, 0.6],  # Relative: 0.4, 0.6 for value=1.0
        }, index=pd.date_range('2020-01-01', periods=2))
        obs_data = pd.Series([1.0, 1.0], index=obs_df.index)

        result = et_evaluator._apply_gleam_qc(obs_df, obs_data)

        # 0.4 <= 0.5, 0.6 > 0.5 -> only first passes
        assert len(result) == 1

    def test_gleam_qc_returns_unchanged_without_uncertainty_column(self, et_evaluator):
        """Test GLEAM QC returns unchanged when uncertainty column not found."""
        et_evaluator.config_dict = {}

        obs_df = pd.DataFrame({
            'other_column': [1, 2, 3, 4],
        }, index=pd.date_range('2020-01-01', periods=4))
        obs_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_df.index)

        result = et_evaluator._apply_gleam_qc(obs_df, obs_data)

        pd.testing.assert_series_equal(result, obs_data)
