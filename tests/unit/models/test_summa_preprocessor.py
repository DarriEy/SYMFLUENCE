"""
Unit tests for SUMMA preprocessor.

Tests SUMMA-specific preprocessing functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from symfluence.utils.models.summa import SummaPreProcessor
from symfluence.utils.exceptions import ModelExecutionError


class TestSummaPreProcessorInitialization:
    """Test SUMMA preprocessor initialization."""

    def test_initialization_with_valid_config(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA preprocessor initializes correctly."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.model_name == "SUMMA"
        assert preprocessor.domain_name == summa_config['DOMAIN_NAME']
        assert preprocessor.forcing_dataset == summa_config['FORCING_DATASET'].lower()

    def test_summa_specific_paths(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA-specific path initialization."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Check SUMMA-specific paths
        assert preprocessor.forcing_summa_path == preprocessor.project_dir / 'forcing' / 'SUMMA_input'
        assert preprocessor.dem_path.exists() or preprocessor.dem_path.name.endswith('.tif')

    def test_summa_configuration_attributes(self, summa_config, mock_logger, setup_test_directories):
        """Test SUMMA configuration attributes are set."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.hruId == summa_config['CATCHMENT_SHP_HRUID']
        assert preprocessor.gruId == summa_config['CATCHMENT_SHP_GRUID']
        assert preprocessor.data_step == summa_config['FORCING_TIME_STEP_SIZE']
        assert preprocessor.forcing_measurement_height == float(summa_config['FORCING_MEASUREMENT_HEIGHT'])

    def test_uses_base_class_forcing_paths(self, summa_config, mock_logger, setup_test_directories):
        """Test that SUMMA uses base class forcing paths."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # These should come from BaseModelPreProcessor
        assert hasattr(preprocessor, 'merged_forcing_path')
        assert hasattr(preprocessor, 'shapefile_path')
        assert hasattr(preprocessor, 'intersect_path')


class TestSummaPathResolution:
    """Test SUMMA path resolution methods."""

    def test_dem_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test DEM path with default name."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        expected_name = f"domain_{summa_config['DOMAIN_NAME']}_elv.tif"
        assert preprocessor.dem_path.name == expected_name

    def test_dem_path_custom(self, summa_config, mock_logger, setup_test_directories):
        """Test DEM path with custom name."""
        summa_config['DEM_NAME'] = 'custom_dem.tif'
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.dem_path.name == 'custom_dem.tif'

    def test_catchment_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test catchment path with defaults."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        discretization = summa_config['DOMAIN_DISCRETIZATION']
        expected_name = f"{summa_config['DOMAIN_NAME']}_HRUs_{discretization}.shp"
        assert preprocessor.catchment_name == expected_name

    def test_river_network_path_default(self, summa_config, mock_logger, setup_test_directories):
        """Test river network path with defaults."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        expected_name = f"{summa_config['DOMAIN_NAME']}_riverNetwork_delineate.shp"
        assert preprocessor.river_network_name == expected_name


class TestSummaCopyBaseSettings:
    """Test SUMMA base settings copying."""

    def test_copy_base_settings_uses_correct_source(self, summa_config, mock_logger, setup_test_directories):
        """Test that copy_base_settings uses correct source directory."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Create source directory with dummy file
        source_dir = setup_test_directories['code_dir'] / 'src' / 'symfluence' / 'data' / 'base_settings' / 'SUMMA'
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / 'test_settings.txt').write_text('test content')

        # Update config to point to source
        preprocessor.config_dict['SYMFLUENCE_CODE_DIR'] = str(setup_test_directories['code_dir'])

        # Call copy_base_settings
        preprocessor.copy_base_settings()

        # Verify file was copied
        settings_path = preprocessor.setup_dir
        assert settings_path.exists()


class TestSummaPreprocessingWorkflow:
    """Test SUMMA preprocessing workflow."""

    @patch.object(SummaPreProcessor, 'apply_datastep_and_lapse_rate')
    @patch.object(SummaPreProcessor, 'copy_base_settings')
    @patch.object(SummaPreProcessor, 'create_file_manager')
    @patch.object(SummaPreProcessor, 'create_forcing_file_list')
    @patch.object(SummaPreProcessor, 'create_initial_conditions')
    @patch.object(SummaPreProcessor, 'create_trial_parameters')
    @patch.object(SummaPreProcessor, 'create_attributes_file')
    def test_run_preprocessing_calls_all_steps(
        self,
        mock_attrs,
        mock_params,
        mock_initial,
        mock_forcing_list,
        mock_file_mgr,
        mock_copy,
        mock_lapse,
        summa_config,
        mock_logger,
        setup_test_directories
    ):
        """Test that run_preprocessing calls all required steps in order."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Run preprocessing
        preprocessor.run_preprocessing()

        # Verify all steps were called
        mock_lapse.assert_called_once()
        mock_copy.assert_called_once()
        mock_file_mgr.assert_called_once()
        mock_forcing_list.assert_called_once()
        mock_initial.assert_called_once()
        mock_params.assert_called_once()
        mock_attrs.assert_called_once()

    @patch.object(SummaPreProcessor, 'apply_datastep_and_lapse_rate', side_effect=Exception('Test error'))
    @patch.object(SummaPreProcessor, 'copy_base_settings')
    def test_run_preprocessing_error_handling(
        self,
        mock_copy,
        mock_lapse,
        summa_config,
        mock_logger,
        setup_test_directories
    ):
        """Test that run_preprocessing handles errors properly."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Should raise ModelExecutionError
        with pytest.raises(ModelExecutionError) as exc_info:
            preprocessor.run_preprocessing()

        assert 'SUMMA preprocessing' in str(exc_info.value)

        # First step should have been called
        mock_lapse.assert_called_once()

        # Subsequent steps should not be called
        mock_copy.assert_not_called()


class TestSummaTimestepHandling:
    """Test SUMMA timestep configuration."""

    def test_data_step_set_from_config(self, summa_config, mock_logger, setup_test_directories):
        """Test that data_step is set from forcing timestep size."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        assert preprocessor.data_step == summa_config['FORCING_TIME_STEP_SIZE']

    def test_uses_base_class_timestep_config(self, summa_config, mock_logger, setup_test_directories):
        """Test that SUMMA can use base class timestep config."""
        preprocessor = SummaPreProcessor(summa_config, mock_logger)

        # Should have access to base class method
        timestep_config = preprocessor.get_timestep_config()

        assert timestep_config['timestep_seconds'] == summa_config['FORCING_TIME_STEP_SIZE']
        assert 'time_label' in timestep_config


class TestSummaRegistration:
    """Test SUMMA model registration."""

    def test_summa_registered_as_preprocessor(self):
        """Test that SUMMA is registered in the model registry."""
        from symfluence.utils.models.registry import ModelRegistry

        # SUMMA should be registered
        assert 'SUMMA' in ModelRegistry._preprocessors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
