"""
Unit tests for the BaseModelPostProcessor class.

Tests initialization, path resolution, unit conversions, and helper methods
for the base postprocessor class.
"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from symfluence.utils.models.base import BaseModelPostProcessor
from symfluence.utils.common.constants import UnitConversion


class ConcretePostProcessor(BaseModelPostProcessor):
    """Concrete implementation of BaseModelPostProcessor for testing."""

    def _get_model_name(self) -> str:
        return "TEST"

    def extract_streamflow(self):
        """Dummy implementation."""
        pass


class TestBaseModelPostProcessorInitialization:
    """Test suite for BaseModelPostProcessor initialization."""

    def test_initialization_basic(self, tmp_path):
        """Test basic initialization with minimal config."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()

        processor = ConcretePostProcessor(config, logger)

        assert processor.config_dict == config
        assert processor.logger == logger
        assert processor.domain_name == 'test_domain'
        assert processor.experiment_id == 'exp_001'
        assert processor.model_name == 'TEST'

    def test_initialization_creates_directories(self, tmp_path):
        """Test that initialization creates necessary directories."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()

        processor = ConcretePostProcessor(config, logger)

        # Check that results directory was created
        assert processor.results_dir.exists()
        assert processor.results_dir.is_dir()

    def test_initialization_paths(self, tmp_path):
        """Test that all paths are correctly set."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()

        processor = ConcretePostProcessor(config, logger)

        expected_project_dir = tmp_path / 'domain_test_domain'
        expected_sim_dir = expected_project_dir / 'simulations' / 'exp_001' / 'TEST'
        expected_results_dir = expected_project_dir / 'results'

        assert processor.project_dir == expected_project_dir
        assert processor.sim_dir == expected_sim_dir
        assert processor.results_dir == expected_results_dir

    def test_initialization_default_experiment_id(self, tmp_path):
        """Test initialization with default experiment ID."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain'
            # No EXPERIMENT_ID provided
        }
        logger = Mock()

        processor = ConcretePostProcessor(config, logger)

        assert processor.experiment_id == 'default_experiment'


class TestBaseModelPostProcessorUnitConversions:
    """Test suite for unit conversion helper methods."""

    def test_convert_mm_per_day_to_cms_basic(self, tmp_path):
        """Test basic mm/day to cms conversion."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create test series: 10 mm/day
        q_mm_day = pd.Series([10.0, 20.0, 30.0])
        area_km2 = 100.0

        q_cms = processor.convert_mm_per_day_to_cms(q_mm_day, area_km2)

        # Expected: q_mm_day * area_km2 / 86.4
        expected = q_mm_day * area_km2 / UnitConversion.MM_DAY_TO_CMS

        pd.testing.assert_series_equal(q_cms, expected)

    def test_convert_mm_per_day_to_cms_with_auto_area(self, tmp_path):
        """Test mm/day to cms conversion with automatic area detection."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Mock get_catchment_area_km2 to return a known value
        processor.get_catchment_area_km2 = Mock(return_value=50.0)

        q_mm_day = pd.Series([10.0, 20.0, 30.0])

        # Call without specifying area (should use auto-detection)
        q_cms = processor.convert_mm_per_day_to_cms(q_mm_day)

        # Should have called get_catchment_area_km2
        processor.get_catchment_area_km2.assert_called_once()

        # Check values
        expected = q_mm_day * 50.0 / UnitConversion.MM_DAY_TO_CMS
        pd.testing.assert_series_equal(q_cms, expected)

    def test_convert_cms_to_mm_per_day_basic(self, tmp_path):
        """Test basic cms to mm/day conversion."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create test series: streamflow in cms
        q_cms = pd.Series([1.0, 2.0, 3.0])
        area_km2 = 100.0

        q_mm_day = processor.convert_cms_to_mm_per_day(q_cms, area_km2)

        # Expected: q_cms * 86.4 / area_km2
        expected = q_cms * UnitConversion.MM_DAY_TO_CMS / area_km2

        pd.testing.assert_series_equal(q_mm_day, expected)

    def test_roundtrip_conversion(self, tmp_path):
        """Test roundtrip conversion: mm/day -> cms -> mm/day."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Original values
        q_mm_day_original = pd.Series([5.0, 10.0, 15.0])
        area_km2 = 75.0

        # Convert to cms
        q_cms = processor.convert_mm_per_day_to_cms(q_mm_day_original, area_km2)

        # Convert back to mm/day
        q_mm_day_back = processor.convert_cms_to_mm_per_day(q_cms, area_km2)

        # Should match original
        pd.testing.assert_series_equal(q_mm_day_original, q_mm_day_back)


class TestBaseModelPostProcessorNetCDFReader:
    """Test suite for read_netcdf_streamflow method."""

    def test_read_netcdf_streamflow_basic(self, tmp_path):
        """Test basic NetCDF streamflow reading."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create a test NetCDF file
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        data = np.random.rand(10, 2, 2, 2)  # time, param_set, lat, lon

        ds = xr.Dataset(
            {'streamflow': (['time', 'param_set', 'latitude', 'longitude'], data)},
            coords={
                'time': time,
                'param_set': [0, 1],
                'latitude': [0, 1],
                'longitude': [0, 1]
            }
        )

        nc_path = tmp_path / 'test_output.nc'
        ds.to_netcdf(nc_path)

        # Read streamflow
        q_sim = processor.read_netcdf_streamflow(
            nc_path,
            'streamflow',
            param_set=0,
            latitude=0,
            longitude=0
        )

        # Should return a pandas Series
        assert isinstance(q_sim, pd.Series)
        assert len(q_sim) == 10
        assert q_sim.name == 'streamflow'

    def test_read_netcdf_streamflow_missing_variable(self, tmp_path):
        """Test error handling for missing variable."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create a test NetCDF file without the expected variable
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        data = np.random.rand(10)

        ds = xr.Dataset(
            {'other_var': (['time'], data)},
            coords={'time': time}
        )

        nc_path = tmp_path / 'test_output.nc'
        ds.to_netcdf(nc_path)

        # Should raise ValueError for missing variable
        with pytest.raises(ValueError):
            processor.read_netcdf_streamflow(nc_path, 'streamflow')


class TestBaseModelPostProcessorSaveStreamflow:
    """Test suite for save_streamflow_to_results method."""

    def test_save_streamflow_new_file(self, tmp_path):
        """Test saving streamflow to new results file."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create test streamflow data
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        streamflow = pd.Series(np.random.rand(10), index=time, name='discharge')

        # Save streamflow
        output_path = processor.save_streamflow_to_results(
            streamflow,
            model_column_name='TEST_discharge_cms'
        )

        # Check file was created
        assert output_path.exists()

        # Read and verify
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert 'TEST_discharge_cms' in df.columns
        assert len(df) == 10

    def test_save_streamflow_append_to_existing(self, tmp_path):
        """Test appending streamflow to existing results file."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create existing results file
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        existing_data = pd.DataFrame({
            'observed_discharge': np.random.rand(10)
        }, index=time)

        results_file = processor.results_dir / f"{processor.experiment_id}_results.csv"
        existing_data.to_csv(results_file)

        # Create new streamflow data
        streamflow = pd.Series(np.random.rand(10), index=time, name='discharge')

        # Save (should append)
        output_path = processor.save_streamflow_to_results(
            streamflow,
            model_column_name='TEST_discharge_cms'
        )

        # Read and verify both columns exist
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert 'observed_discharge' in df.columns
        assert 'TEST_discharge_cms' in df.columns

    def test_save_streamflow_custom_output_file(self, tmp_path):
        """Test saving streamflow to custom output file."""
        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create test streamflow data
        time = pd.date_range('2020-01-01', periods=10, freq='D')
        streamflow = pd.Series(np.random.rand(10), index=time)

        # Custom output path
        custom_path = tmp_path / 'custom_output.csv'

        # Save to custom path
        output_path = processor.save_streamflow_to_results(
            streamflow,
            output_file=custom_path
        )

        assert output_path == custom_path
        assert custom_path.exists()


class TestBaseModelPostProcessorCatchmentArea:
    """Test suite for get_catchment_area_km2 method."""

    def test_get_catchment_area_from_shapefile(self, tmp_path):
        """Test reading catchment area from river basins shapefile."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'DOMAIN_DEFINITION_METHOD': 'default'  # Required for shapefile naming
        }
        logger = Mock()
        processor = ConcretePostProcessor(config, logger)

        # Create mock shapefile with GRU_area attribute
        shapefiles_dir = processor.project_dir / 'shapefiles' / 'river_basins'
        shapefiles_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple polygon
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame({
            'geometry': [polygon],
            'GRU_area': [150_500_000.0]  # Area in m² (will be converted to km²)
        }, crs='EPSG:4326')

        # Shapefile name matches expected pattern: {domain_name}_riverBasins_{method}.shp
        shapefile_path = shapefiles_dir / 'test_riverBasins_default.shp'
        gdf.to_file(shapefile_path)

        # Get area (should convert from m² to km²)
        area = processor.get_catchment_area_km2()

        # Expected: 150_500_000 m² / 1e6 = 150.5 km²
        assert abs(area - 150.5) < 0.01


class TestBaseModelPostProcessorCustomization:
    """Test suite for customization hooks."""

    def test_custom_simulation_dir(self, tmp_path):
        """Test overriding _get_simulation_dir."""

        class CustomPostProcessor(BaseModelPostProcessor):
            def _get_model_name(self):
                return "CUSTOM"

            def _get_simulation_dir(self):
                # Custom path
                return self.project_dir / 'custom_sims' / self.experiment_id

            def extract_streamflow(self):
                pass

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()

        processor = CustomPostProcessor(config, logger)

        expected_sim_dir = processor.project_dir / 'custom_sims' / 'exp_001'
        assert processor.sim_dir == expected_sim_dir

    def test_setup_model_specific_paths(self, tmp_path):
        """Test _setup_model_specific_paths hook."""

        class CustomPostProcessor(BaseModelPostProcessor):
            def _get_model_name(self):
                return "CUSTOM"

            def _setup_model_specific_paths(self):
                # Add custom paths
                self.custom_path = self.project_dir / 'custom_data'

            def extract_streamflow(self):
                pass

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001'
        }
        logger = Mock()

        processor = CustomPostProcessor(config, logger)

        # Check custom path was set
        assert hasattr(processor, 'custom_path')
        assert processor.custom_path == processor.project_dir / 'custom_data'


class TestBaseModelPostProcessorAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_missing_get_model_name_raises_error(self, tmp_path):
        """Test that missing _get_model_name raises TypeError."""

        class IncompletePostProcessor(BaseModelPostProcessor):
            # Missing _get_model_name
            def extract_streamflow(self):
                pass

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test'
        }
        logger = Mock()

        with pytest.raises(TypeError):
            IncompletePostProcessor(config, logger)

    def test_missing_extract_streamflow_raises_error(self, tmp_path):
        """Test that missing extract_streamflow raises TypeError."""

        class IncompletePostProcessor(BaseModelPostProcessor):
            def _get_model_name(self):
                return "INCOMPLETE"
            # Missing extract_streamflow

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test'
        }
        logger = Mock()

        with pytest.raises(TypeError):
            IncompletePostProcessor(config, logger)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
