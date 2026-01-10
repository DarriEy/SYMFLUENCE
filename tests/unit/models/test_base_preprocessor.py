"""
Unit tests for BaseModelPreProcessor.

Tests the base class functionality shared across all model preprocessors.
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from symfluence.models.base.base_preprocessor import BaseModelPreProcessor


class ConcretePreProcessor(BaseModelPreProcessor):
    """Concrete implementation for testing."""

    def _get_model_name(self) -> str:
        return "TEST"

    def run_preprocessing(self):
        pass


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    return {
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'EXPERIMENT_ID': 'test_experiment',
        'EXPERIMENT_TIME_START': '2020-01-01',
        'EXPERIMENT_TIME_END': '2020-01-02',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'TEST',
        'DOMAIN_NAME': 'test_domain',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'CATCHMENT_PATH': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'RIVER_NETWORK_SHP_PATH': 'default',
        'RIVER_NETWORK_SHP_NAME': 'default'
    }


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

    return ConcretePreProcessor(config, logger)


class TestBaseModelPreProcessorInitialization:
    """Test initialization and basic properties."""

    def test_initialization(self, preprocessor, config, tmp_path):
        """Test basic initialization."""
        # Check that all keys in the input config are present and correct in the result
        actual_config = preprocessor.config.to_dict(flatten=True)
        for key, value in config.items():
            assert actual_config[key] == value
        
        assert preprocessor.domain_name == 'test_domain'
        assert preprocessor.data_dir == tmp_path
        assert preprocessor.project_dir == tmp_path / 'domain_test_domain'
        assert preprocessor.model_name == 'TEST'

    def test_setup_dir_creation(self, preprocessor, tmp_path):
        """Test that setup_dir is set correctly."""
        expected_setup_dir = tmp_path / 'domain_test_domain' / 'settings' / 'TEST'
        assert preprocessor.setup_dir == expected_setup_dir

    def test_forcing_dir_creation(self, preprocessor, tmp_path):
        """Test that forcing_dir is set correctly."""
        expected_forcing_dir = tmp_path / 'domain_test_domain' / 'forcing' / 'TEST_input'
        assert preprocessor.forcing_dir == expected_forcing_dir

    def test_forcing_basin_path(self, preprocessor, tmp_path):
        """Test that forcing_basin_path is set correctly."""
        expected_path = tmp_path / 'domain_test_domain' / 'forcing' / 'basin_averaged_data'
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

    def test_get_default_path_with_custom(self, config, logger, tmp_path):
        """Test _get_default_path with custom path."""
        custom_path = tmp_path / 'custom' / 'path'
        config['CUSTOM_PATH'] = str(custom_path)
        # Re-initialize since config is frozen in preprocessor
        proc = ConcretePreProcessor(config, logger)
        result = proc._get_default_path('CUSTOM_PATH', 'default/path')
        assert result == custom_path

    def test_get_catchment_path_default(self, preprocessor):
        """Test get_catchment_path with defaults."""
        result = preprocessor.get_catchment_path()
        expected = preprocessor.project_dir / 'shapefiles' / 'catchment' / 'test_domain_HRUs_lumped.shp'
        assert result == expected

    def test_get_river_network_path_default(self, preprocessor):
        """Test get_river_network_path with defaults."""
        result = preprocessor.get_river_network_path()
        # Since domain_definition_method is 'lumped', the suffix is 'lumped'
        expected = preprocessor.project_dir / 'shapefiles' / 'river_network' / 'test_domain_riverNetwork_lumped.shp'
        assert result == expected


class TestDirectoryCreation:
    """Test directory creation methods."""

    def test_create_directories_basic(self, preprocessor, tmp_path):
        """Test basic directory creation."""
        # Remove directories if they exist
        if preprocessor.setup_dir.exists():
            shutil.rmtree(preprocessor.setup_dir)
        if preprocessor.forcing_dir.exists():
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

    def test_get_file_path_with_custom_path(self, config, logger, tmp_path):
        """Test _get_file_path with custom path."""
        custom_dir = tmp_path / 'custom'
        custom_dir.mkdir(exist_ok=True)
        config['FILE_PATH'] = str(custom_dir)
        config['FILE_NAME'] = 'custom_file.txt'
        
        # Re-initialize since config is frozen
        proc = ConcretePreProcessor(config, logger)

        result = proc._get_file_path(
            'test_file',
            'FILE_PATH',
            'FILE_NAME',
            'default.txt'
        )

        assert result == custom_dir / 'custom_file.txt'


class TestHelperMethods:
    """Test helper methods."""

    def test_is_lumped_true(self, preprocessor):
        """Test _is_lumped returns True for lumped configuration."""
        assert preprocessor._is_lumped() is True

    def test_is_lumped_false(self, config, logger):
        """Test _is_lumped returns False for distributed configuration."""
        config['DOMAIN_DEFINITION_METHOD'] = 'delineate'
        proc = ConcretePreProcessor(config, logger)
        assert proc._is_lumped() is False


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
