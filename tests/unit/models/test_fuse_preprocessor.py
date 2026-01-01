"""
Unit tests for FUSE preprocessor.

Tests FUSE-specific preprocessing functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.symfluence.utils.models.fuse_utils import FUSEPreProcessor
from src.symfluence.utils.exceptions import ModelExecutionError


class TestFUSEPreProcessorInitialization:
    """Test FUSE preprocessor initialization."""

    def test_initialization_with_valid_config(self, fuse_config, mock_logger, setup_test_directories):
        """Test FUSE preprocessor initializes correctly."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        assert preprocessor.model_name == "FUSE"
        assert preprocessor.domain_name == fuse_config['DOMAIN_NAME']
        assert preprocessor.forcing_dataset == fuse_config['FORCING_DATASET'].lower()

    def test_fuse_specific_paths(self, fuse_config, mock_logger, setup_test_directories):
        """Test FUSE-specific path initialization."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Check FUSE-specific paths
        assert preprocessor.forcing_fuse_path == preprocessor.project_dir / 'forcing' / 'FUSE_input'
        assert preprocessor.catchment_path is not None

    def test_uses_base_class_paths(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE uses base class paths."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # These should come from BaseModelPreProcessor
        assert hasattr(preprocessor, 'merged_forcing_path')
        assert hasattr(preprocessor, 'forcing_raw_path')
        assert hasattr(preprocessor, 'shapefile_path')


class TestFUSETimestepConfiguration:
    """Test FUSE timestep configuration."""

    def test_get_timestep_config_hourly(self, fuse_config, mock_logger, setup_test_directories):
        """Test timestep config for hourly data."""
        fuse_config['FORCING_TIME_STEP_SIZE'] = 3600
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        config = preprocessor.get_timestep_config()

        assert config['resample_freq'] == 'h'
        assert config['time_units'] == 'hours since 1970-01-01'
        assert config['time_unit'] == 'h'
        assert config['conversion_factor'] == 3.6
        assert config['time_label'] == 'hourly'

    def test_get_timestep_config_daily(self, fuse_config, mock_logger, setup_test_directories):
        """Test timestep config for daily data."""
        fuse_config['FORCING_TIME_STEP_SIZE'] = 86400
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        config = preprocessor.get_timestep_config()

        assert config['resample_freq'] == 'D'
        assert config['time_units'] == 'days since 1970-01-01'
        assert config['time_unit'] == 'D'
        assert config['conversion_factor'] == 86.4
        assert config['time_label'] == 'daily'

    def test_get_timestep_config_custom_3hourly(self, fuse_config, mock_logger, setup_test_directories):
        """Test timestep config for 3-hourly data."""
        fuse_config['FORCING_TIME_STEP_SIZE'] = 10800  # 3 hours
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        config = preprocessor.get_timestep_config()

        assert config['resample_freq'] == '3h'
        assert config['time_label'] == '3-hourly'
        assert config['conversion_factor'] == 3.6 * 3


class TestFUSEDirectoryCreation:
    """Test FUSE directory creation."""

    def test_create_directories(self, fuse_config, mock_logger, setup_test_directories, temp_dir):
        """Test FUSE directory creation."""
        fuse_config['SYMFLUENCE_DATA_DIR'] = str(temp_dir)
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Remove directories if they exist
        import shutil
        if preprocessor.setup_dir.exists():
            shutil.rmtree(preprocessor.setup_dir)
        if preprocessor.forcing_fuse_path.exists():
            shutil.rmtree(preprocessor.forcing_fuse_path)

        preprocessor.create_directories()

        assert preprocessor.setup_dir.exists()
        assert preprocessor.forcing_fuse_path.exists()


class TestFUSECopyBaseSettings:
    """Test FUSE base settings copying."""

    def test_copy_base_settings_uses_correct_source(self, fuse_config, mock_logger, setup_test_directories):
        """Test that copy_base_settings uses correct source directory."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Create source directory with dummy files
        source_dir = setup_test_directories['code_dir'] / '0_base_settings' / 'FUSE'
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / 'test_settings.txt').write_text('test content')
        (source_dir / 'fuse_zDecisions_902.txt').write_text('decisions')

        # Update config to point to source
        preprocessor.config['SYMFLUENCE_CODE_DIR'] = str(setup_test_directories['code_dir'])

        # Call copy_base_settings
        preprocessor.copy_base_settings()

        # Verify settings directory exists
        assert preprocessor.setup_dir.exists()


class TestFUSEPreprocessingWorkflow:
    """Test FUSE preprocessing workflow."""

    @patch.object(FUSEPreProcessor, 'create_directories')
    @patch.object(FUSEPreProcessor, 'copy_base_settings')
    @patch.object(FUSEPreProcessor, 'prepare_forcing_data')
    @patch.object(FUSEPreProcessor, 'create_elevation_bands')
    @patch.object(FUSEPreProcessor, 'create_filemanager')
    def test_run_preprocessing_calls_all_steps(
        self,
        mock_filemanager,
        mock_elevation,
        mock_forcing,
        mock_copy,
        mock_dirs,
        fuse_config,
        mock_logger,
        setup_test_directories
    ):
        """Test that run_preprocessing calls all required steps in order."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Run preprocessing
        preprocessor.run_preprocessing()

        # Verify all steps were called
        mock_dirs.assert_called_once()
        mock_copy.assert_called_once()
        mock_forcing.assert_called_once()
        mock_elevation.assert_called_once()
        mock_filemanager.assert_called_once()

    @patch.object(FUSEPreProcessor, 'create_directories', side_effect=Exception('Test error'))
    @patch.object(FUSEPreProcessor, 'copy_base_settings')
    def test_run_preprocessing_error_handling(
        self,
        mock_copy,
        mock_dirs,
        fuse_config,
        mock_logger,
        setup_test_directories
    ):
        """Test that run_preprocessing handles errors properly."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Should raise ModelExecutionError
        with pytest.raises(ModelExecutionError) as exc_info:
            preprocessor.run_preprocessing()

        assert 'FUSE preprocessing' in str(exc_info.value)

        # First step should have been called
        mock_dirs.assert_called_once()

        # Subsequent steps should not be called
        mock_copy.assert_not_called()


class TestFUSECatchmentPath:
    """Test FUSE catchment path handling."""

    def test_catchment_path_uses_base_class(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE uses base class get_catchment_path."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Should use base class method
        expected = preprocessor.project_dir / 'shapefiles' / 'catchment' / f"{fuse_config['DOMAIN_NAME']}_HRUs_{fuse_config['DOMAIN_DISCRETIZATION']}.shp"
        assert preprocessor.catchment_path == expected


class TestFUSESpatialMode:
    """Test FUSE spatial mode configuration."""

    def test_default_lumped_mode(self, fuse_config, mock_logger, setup_test_directories):
        """Test default FUSE spatial mode is lumped."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Config should have lumped mode
        assert preprocessor.config.get('FUSE_SPATIAL_MODE') == 'lumped'


class TestFUSERegistration:
    """Test FUSE model registration."""

    def test_fuse_registered_as_preprocessor(self):
        """Test that FUSE is registered in the model registry."""
        from src.symfluence.utils.models.registry import ModelRegistry

        # FUSE should be registered
        assert 'FUSE' in ModelRegistry._preprocessors


class TestFUSEPETCalculator:
    """Test FUSE PET calculator mixin."""

    def test_has_pet_calculator_mixin(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE preprocessor has PET calculator functionality."""
        from src.symfluence.utils.models.mixins import PETCalculatorMixin

        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Should be instance of PETCalculatorMixin
        assert isinstance(preprocessor, PETCalculatorMixin)


class TestFUSEBaseClassIntegration:
    """Test FUSE integration with base class."""

    def test_inherits_from_base_preprocessor(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE inherits from BaseModelPreProcessor."""
        from src.symfluence.utils.models.base import BaseModelPreProcessor

        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        assert isinstance(preprocessor, BaseModelPreProcessor)

    def test_uses_base_class_attributes(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE uses base class attributes."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Should have base class attributes
        assert hasattr(preprocessor, 'forcing_time_step_size')
        assert hasattr(preprocessor, 'forcing_dataset')
        assert preprocessor.forcing_time_step_size == fuse_config['FORCING_TIME_STEP_SIZE']

    def test_uses_base_class_methods(self, fuse_config, mock_logger, setup_test_directories):
        """Test that FUSE can use base class methods."""
        preprocessor = FUSEPreProcessor(fuse_config, mock_logger)

        # Should be able to call base class methods
        dem_path = preprocessor.get_dem_path()
        assert dem_path is not None

        base_settings_dir = preprocessor.get_base_settings_source_dir()
        assert base_settings_dir.name == 'FUSE'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
