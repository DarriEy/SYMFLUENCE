"""
Unit tests for BaseModelPreProcessor.

Tests the base class functionality shared across all model preprocessors.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from symfluence.models.base import BaseModelPreProcessor
from symfluence.core.exceptions import (
    ConfigurationError,
    FileOperationError,
    ModelExecutionError
)
from symfluence.core.config.models import SymfluenceConfig


class ConcretePreProcessor(BaseModelPreProcessor):
    """Concrete implementation for testing."""

    def _get_model_name(self) -> str:
        return "TEST"

    def run_preprocessing(self):
        pass


def _create_config(tmp_path, overrides=None):
    """Create a valid SymfluenceConfig with required fields."""
    base = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'exp_001',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
    }
    if overrides:
        base.update(overrides)
    return SymfluenceConfig(**base)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    return _create_config(tmp_path)


@pytest.fixture
def logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def preprocessor(config, logger, tmp_path):
    """Create a concrete preprocessor instance."""
    # Create necessary directory structure
    project_dir = tmp_path / 'domain_test_domain'
    (project_dir / 'settings' / 'TEST').mkdir(parents=True, exist_ok=True)
    (project_dir / 'forcing' / 'TEST_input').mkdir(parents=True, exist_ok=True)
    # Make sure local base settings exist in the fake code directory
    code_dir = Path(config.system.code_dir)
    (code_dir / 'src' / 'symfluence' / 'resources' / 'base_settings' / 'TEST').mkdir(parents=True, exist_ok=True)

    return ConcretePreProcessor(config, logger)


class TestBaseModelPreProcessorInitialization:
    """Test initialization and basic properties."""

    def test_initialization(self, preprocessor, config, tmp_path):
        """Test basic initialization."""
        # Use Path() for comparison since to_dict converts Path to string
        assert Path(preprocessor.config_dict['SYMFLUENCE_DATA_DIR']) == Path(tmp_path)
        assert preprocessor.domain_name == 'test_domain'
        assert preprocessor.data_dir == Path(tmp_path).resolve() # SymfluenceConfig resolves paths
        assert preprocessor.project_dir == Path(tmp_path).resolve() / 'domain_test_domain'
        assert preprocessor.model_name == 'TEST'

    def test_setup_dir_creation(self, preprocessor, tmp_path):
        """Test that setup_dir is set correctly."""
        expected_setup_dir = Path(tmp_path).resolve() / 'domain_test_domain' / 'settings' / 'TEST'
        assert preprocessor.setup_dir == expected_setup_dir

    def test_forcing_dir_creation(self, preprocessor, tmp_path):
        """Test that forcing_dir is set correctly."""
        expected_forcing_dir = Path(tmp_path).resolve() / 'domain_test_domain' / 'forcing' / 'TEST_input'
        assert preprocessor.forcing_dir == expected_forcing_dir

    def test_forcing_basin_path(self, preprocessor, tmp_path):
        """Test that forcing_basin_path is set correctly."""
        expected_path = Path(tmp_path).resolve() / 'domain_test_domain' / 'forcing' / 'basin_averaged_data'
        assert preprocessor.forcing_basin_path == expected_path


class TestPathResolution:
    """Test path resolution methods."""

    def test_get_default_path_with_default(self, preprocessor):
        """Test _get_default_path with 'default' value."""
        result = preprocessor._get_default_path('SOME_PATH', 'some/subpath')
        expected = preprocessor.project_dir / 'some/subpath'
        assert result == expected

    def test_get_default_path_with_none(self, preprocessor):
        """Test _get_default_path with None value."""
        result = preprocessor._get_default_path('NONEXISTENT_KEY', 'fallback/path')
        expected = preprocessor.project_dir / 'fallback/path'
        assert result == expected

    def test_get_default_path_with_custom(self, preprocessor, tmp_path):
        """Test _get_default_path with custom path."""
        custom_path = (tmp_path / 'custom' / 'path').resolve()
        # Re-initialize with custom path
        config = _create_config(tmp_path, {'CUSTOM_PATH': str(custom_path)})
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        result = preprocessor._get_default_path('CUSTOM_PATH', 'default/path')
        assert result.resolve() == custom_path

    def test_get_catchment_path_default(self, preprocessor):
        """Test get_catchment_path with defaults."""
        result = preprocessor.get_catchment_path()
        expected = preprocessor.project_dir / 'shapefiles' / 'catchment' / 'test_domain_HRUs_lumped.shp'
        assert result == expected

    def test_get_river_network_path_default(self, preprocessor):
        """Test get_river_network_path with defaults."""
        result = preprocessor.get_river_network_path()
        expected = preprocessor.project_dir / 'shapefiles' / 'river_network' / 'test_domain_riverNetwork_lumped.shp'
        assert result == expected


class TestDirectoryCreation:
    """Test directory creation methods."""

    def test_create_directories_basic(self, preprocessor, tmp_path):
        """Test basic directory creation."""
        # Remove directories if they exist
        if preprocessor.setup_dir.exists():
            import shutil
            shutil.rmtree(preprocessor.setup_dir)
        if preprocessor.forcing_dir.exists():
            import shutil
            shutil.rmtree(preprocessor.forcing_dir)

        preprocessor.create_directories()

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()

    def test_create_directories_with_additional(self, preprocessor, tmp_path):
        """Test directory creation with additional directories."""
        additional_dir = preprocessor.project_dir / 'extra' / 'directory'

        preprocessor.create_directories(additional_dirs=[additional_dir])

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()
        assert additional_dir.exists()

    def test_create_directories_idempotent(self, preprocessor):
        """Test that create_directories can be called multiple times."""
        preprocessor.create_directories()
        preprocessor.create_directories()  # Should not raise error

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_dir.exists()


class TestFilePathResolution:
    """Test file path resolution."""

    def test_get_file_path_with_defaults(self, preprocessor):
        """Test _get_file_path with all defaults."""
        result = preprocessor._get_file_path(
            'test_file',
            'FILE_PATH',
            'FILE_NAME',
            'default.txt'
        )

        # When path is default/None, uses project_dir
        assert result == preprocessor.project_dir / 'default.txt'

    def test_get_file_path_with_custom_path(self, preprocessor, tmp_path):
        """Test _get_file_path with custom path."""
        custom_dir = (tmp_path / 'custom').resolve()
        custom_dir.mkdir(exist_ok=True)
        
        # Re-initialize with custom settings
        config = _create_config(tmp_path, {
            'FILE_PATH': str(custom_dir),
            'FILE_NAME': 'custom_file.txt'
        })
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)

        result = preprocessor._get_file_path(
            'test_file',
            'FILE_PATH',
            'FILE_NAME',
            'default.txt'
        )

        assert result.resolve() == custom_dir / 'custom_file.txt'


class TestHelperMethods:
    """Test helper methods."""

    def test_is_lumped_true(self, preprocessor):
        """Test _is_lumped returns True for lumped configuration."""
        assert preprocessor._is_lumped() is True

    def test_is_lumped_false(self, preprocessor, tmp_path):
        """Test _is_lumped returns False for distributed configuration."""
        config = _create_config(tmp_path, {'DOMAIN_DEFINITION_METHOD': 'distributed'})
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        assert preprocessor._is_lumped() is False


class TestCopyBaseSettings:
    """Test base settings copying."""

    def test_copy_base_settings_directory_not_exist(self, preprocessor, tmp_path, logger):
        """Test copy_base_settings handles missing source directory gracefully."""
        non_existent = tmp_path / 'does_not_exist'

        preprocessor.copy_base_settings(source_dir=non_existent)

        # Should log warning but not raise error
        assert logger.warning.called

    def test_copy_base_settings_with_source_dir(self, preprocessor, tmp_path):
        """Test copy_base_settings with valid source directory."""
        source_dir = tmp_path / 'source'
        source_dir.mkdir(exist_ok=True)
        (source_dir / 'test_file.txt').write_text('test content')

        preprocessor.copy_base_settings(source_dir=source_dir)

        # Check if file was copied
        dest_file = preprocessor.setup_dir / 'test_file.txt'
        assert dest_file.exists()
        assert dest_file.read_text() == 'test content'

    def test_copy_base_settings_with_patterns(self, preprocessor, tmp_path):
        """Test copy_base_settings with file patterns."""
        source_dir = tmp_path / 'source'
        source_dir.mkdir(exist_ok=True)
        (source_dir / 'test.txt').write_text('text file')
        (source_dir / 'test.nc').write_text('netcdf file')
        (source_dir / 'ignore.dat').write_text('ignore this')

        # Clear setup dir
        if preprocessor.setup_dir.exists():
            import shutil
            shutil.rmtree(preprocessor.setup_dir)
        preprocessor.setup_dir.mkdir(parents=True, exist_ok=True)

        preprocessor.copy_base_settings(
            source_dir=source_dir,
            file_patterns=['*.txt', '*.nc']
        )

        # Check that only .txt and .nc files were copied
        assert (preprocessor.setup_dir / 'test.txt').exists()
        assert (preprocessor.setup_dir / 'test.nc').exists()
        assert not (preprocessor.setup_dir / 'ignore.dat').exists()


class TestAbstractMethods:
    """Test abstract method requirements."""

    def test_requires_get_model_name(self, config, logger):
        """Test that _get_model_name must be implemented."""
        class IncompletePreProcessor(BaseModelPreProcessor):
            def run_preprocessing(self):
                pass
            # Missing _get_model_name

        with pytest.raises(TypeError):
            IncompletePreProcessor(config, logger)

    def test_requires_run_preprocessing(self, config, logger):
        """Test that run_preprocessing must be implemented."""
        class IncompletePreProcessor(BaseModelPreProcessor):
            def _get_model_name(self) -> str:
                return "TEST"
            # Missing run_preprocessing

        with pytest.raises(TypeError):
            IncompletePreProcessor(config, logger)


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_missing_required_config_keys(self, logger):
        """Test that missing required config keys raise ValueError (from SymfluenceConfig)."""
        incomplete_config = {
            'DOMAIN_NAME': 'test'
            # Missing SYMFLUENCE_DATA_DIR and FORCING_DATASET
        }

        # SymfluenceConfig raises ValueError during initialization if required keys are missing
        with pytest.raises(ValueError) as exc_info:
            SymfluenceConfig(**incomplete_config)

        assert 'Missing required configuration keys' in str(exc_info.value)

    def test_valid_config_passes_validation(self, config, logger):
        """Test that valid configuration passes validation."""
        # Should not raise any exception
        preprocessor = ConcretePreProcessor(config, logger)
        assert preprocessor is not None


class TestNewMethods:
    """Test newly added methods in refactoring."""

    def test_get_dem_path_default(self, preprocessor):
        """Test get_dem_path with default DEM name."""
        result = preprocessor.get_dem_path()
        expected = preprocessor.project_dir / 'attributes' / 'elevation' / 'dem' / 'domain_test_domain_elv.tif'
        assert result == expected

    def test_get_dem_path_custom(self, preprocessor, tmp_path):
        """Test get_dem_path with custom DEM name."""
        # Re-initialize with custom settings
        config = _create_config(tmp_path, {'DEM_NAME': 'custom_dem.tif'})
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        
        result = preprocessor.get_dem_path()
        expected = preprocessor.project_dir / 'attributes' / 'elevation' / 'dem' / 'custom_dem.tif'
        assert result == expected

    def test_get_timestep_config_hourly(self, preprocessor, tmp_path):
        """Test get_timestep_config for hourly data."""
        config = _create_config(tmp_path, {'FORCING_TIME_STEP_SIZE': 3600})
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        ts_config = preprocessor.get_timestep_config()

        assert ts_config['resample_freq'] == 'h'
        assert ts_config['time_units'] == 'hours since 1970-01-01'
        assert ts_config['time_unit'] == 'h'
        assert ts_config['conversion_factor'] == 3.6
        assert ts_config['time_label'] == 'hourly'
        assert ts_config['timestep_seconds'] == 3600

    def test_get_timestep_config_daily(self, preprocessor, tmp_path):
        """Test get_timestep_config for daily data."""
        config = _create_config(tmp_path, {'FORCING_TIME_STEP_SIZE': 86400})
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        ts_config = preprocessor.get_timestep_config()

        assert ts_config['resample_freq'] == 'D'
        assert ts_config['time_units'] == 'days since 1970-01-01'
        assert ts_config['time_unit'] == 'D'
        assert ts_config['conversion_factor'] == 86.4
        assert ts_config['time_label'] == 'daily'
        assert ts_config['timestep_seconds'] == 86400

    def test_get_timestep_config_custom(self, preprocessor, tmp_path):
        """Test get_timestep_config for custom timestep."""
        config = _create_config(tmp_path, {'FORCING_TIME_STEP_SIZE': 10800}) # 3 hours
        preprocessor = ConcretePreProcessor(config, preprocessor.logger)
        ts_config = preprocessor.get_timestep_config()

        assert ts_config['resample_freq'] == '3h'
        assert ts_config['time_label'] == '3-hourly'
        assert ts_config['conversion_factor'] == 3.6 * 3

    def test_get_base_settings_source_dir(self, preprocessor, tmp_path):
        """Test get_base_settings_source_dir."""
        result = preprocessor.get_base_settings_source_dir()
        # config_dict contains strings for paths
        code_dir = Path(preprocessor.config_dict['SYMFLUENCE_CODE_DIR'])
        expected = (
            code_dir
            / 'src'
            / 'symfluence'
            / 'resources'
            / 'base_settings'
            / 'TEST'
        )
        assert result == expected


class TestNewAttributes:
    """Test newly added attributes."""

    def test_forcing_raw_path(self, preprocessor):
        """Test that forcing_raw_path is set correctly."""
        expected = preprocessor.project_dir / 'forcing' / 'raw_data'
        assert preprocessor.forcing_raw_path == expected

    def test_merged_forcing_path(self, preprocessor):
        """Test that merged_forcing_path is set correctly."""
        expected = preprocessor.project_dir / 'forcing' / 'merged_data'
        assert preprocessor.merged_forcing_path == expected

    def test_shapefile_path(self, preprocessor):
        """Test that shapefile_path is set correctly."""
        expected = preprocessor.project_dir / 'shapefiles' / 'forcing'
        assert preprocessor.shapefile_path == expected

    def test_intersect_path(self, preprocessor):
        """Test that intersect_path is set correctly."""
        expected = preprocessor.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        assert preprocessor.intersect_path == expected

    def test_forcing_dataset(self, preprocessor):
        """Test that forcing_dataset is set correctly."""
        assert preprocessor.forcing_dataset == 'era5'

    def test_forcing_time_step_size(self, preprocessor):
        """Test that forcing_time_step_size is set correctly."""
        assert preprocessor.forcing_time_step_size == 3600


class TestErrorHandling:
    """Test error handling improvements."""

    def test_create_directories_raises_on_failure(self, preprocessor, logger):
        """Test that create_directories raises FileOperationError on failure."""
        # Make setup_dir point to a file (not a directory) to cause mkdir to fail
        bad_file = preprocessor.project_dir / 'bad_file'
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text('This is a file, not a directory')

        preprocessor.setup_dir = bad_file

        with pytest.raises(FileOperationError) as exc_info:
            preprocessor.create_directories()

        assert 'Failed to create directory' in str(exc_info.value)

    def test_copy_base_settings_raises_on_missing_dir(self, preprocessor, tmp_path, logger):
        """Test that copy_base_settings logs warning but doesn't raise error for missing source."""
        non_existent = tmp_path / 'does_not_exist'

        # Should not raise error now, just log warning
        preprocessor.copy_base_settings(source_dir=non_existent)

        assert logger.warning.called
        assert 'Base settings source directory does not exist' in logger.warning.call_args[0][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
