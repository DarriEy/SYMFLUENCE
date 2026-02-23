"""
Tests for MESH postprocessor.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest


class TestMESHPostProcessorInitialization:
    """Tests for MESH postprocessor initialization."""

    def test_postprocessor_can_be_imported(self):
        """Test that MESHPostProcessor can be imported."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor
        assert MESHPostProcessor is not None

    def test_postprocessor_initialization(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test postprocessor initializes with config."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)
        assert postprocessor is not None
        assert postprocessor.domain_name == 'test_domain'

    def test_postprocessor_model_name(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test postprocessor returns correct model name."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)
        assert postprocessor._get_model_name() == 'MESH'


class TestMESHPostProcessorPaths:
    """Tests for MESH postprocessor path setup."""

    def test_mesh_setup_dir_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test MESH setup directory is configured."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)
        assert postprocessor.mesh_setup_dir is not None
        assert 'MESH' in str(postprocessor.mesh_setup_dir)

    def test_forcing_mesh_path_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test forcing MESH path is configured."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)
        assert postprocessor.forcing_mesh_path is not None
        assert 'MESH_input' in str(postprocessor.forcing_mesh_path)

    def test_catchment_path_configured(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test catchment path is configured."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)
        assert postprocessor.catchment_path is not None


class TestMESHStreamflowExtraction:
    """Tests for MESH streamflow extraction."""

    def test_extract_streamflow_returns_none_when_missing(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test streamflow extraction returns None when file missing."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        result = postprocessor.extract_streamflow()

        assert result is None

    def test_extract_streamflow_with_valid_output(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test streamflow extraction with valid output file."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        # Mock save_streamflow_to_results to avoid file operations
        with patch.object(postprocessor, 'save_streamflow_to_results') as mock_save:
            mock_save.return_value = Path('/fake/path/streamflow.csv')

            result = postprocessor.extract_streamflow()

            assert result is not None
            mock_save.assert_called_once()

    def test_extract_streamflow_uses_qosim1(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test streamflow extraction uses QOSIM1 column."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        with patch.object(postprocessor, 'save_streamflow_to_results') as mock_save:
            mock_save.return_value = Path('/fake/path/streamflow.csv')

            postprocessor.extract_streamflow()

            # Check that the call was made with the correct column name
            call_args = mock_save.call_args
            assert call_args is not None
            # Check the series passed has correct values
            series_arg = call_args[0][0]
            assert abs(series_arg.iloc[0] - 11.2) < 0.01  # First QOSIM1 value


class TestMESHPostProcessorRegistry:
    """Tests for MESH postprocessor registry integration."""

    def test_postprocessor_registered_with_registry(self):
        """Test MESH postprocessor is registered with model registry."""
        from symfluence.models.registry import ModelRegistry

        # Check if MESH postprocessor is registered
        postprocessors = ModelRegistry._postprocessors
        assert 'MESH' in postprocessors

    def test_postprocessor_is_correct_class(self):
        """Test registered postprocessor is MESHPostProcessor."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor
        from symfluence.models.registry import ModelRegistry

        postprocessor_class = ModelRegistry._postprocessors.get('MESH')
        assert postprocessor_class == MESHPostProcessor


class TestMESHTimeConversion:
    """Tests for MESH time conversion in postprocessor."""

    def test_julian_to_datetime_conversion(self, mesh_config, mock_logger, setup_mesh_directories, sample_mesh_output_csv):
        """Test Julian day conversion in postprocessor."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        # Read the CSV manually to check conversion
        df = pd.read_csv(sample_mesh_output_csv, skipinitialspace=True)

        from datetime import datetime, timedelta

        def julian_to_datetime(row):
            return datetime(int(row['YEAR']), 1, 1) + timedelta(days=int(row['DAY']) - 1)

        df['datetime'] = df.apply(julian_to_datetime, axis=1)

        # Check first row is Jan 1, 2020
        assert df['datetime'].iloc[0].year == 2020
        assert df['datetime'].iloc[0].month == 1
        assert df['datetime'].iloc[0].day == 1


class TestMESHErrorHandling:
    """Tests for MESH postprocessor error handling."""

    def test_extract_streamflow_handles_missing_qosim(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test extraction handles missing QOSIM columns."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        # Create a CSV without QOSIM columns
        bad_csv = setup_mesh_directories['forcing_dir'] / 'MESH_output_streamflow.csv'
        bad_csv.write_text("DAY, YEAR, OTHER_COL\n1, 2020, 10.5\n")

        result = postprocessor.extract_streamflow()

        assert result is None

    def test_extract_streamflow_handles_parse_error(self, mesh_config, mock_logger, setup_mesh_directories):
        """Test extraction handles CSV parse errors."""
        from symfluence.models.mesh.postprocessor import MESHPostProcessor

        postprocessor = MESHPostProcessor(mesh_config, mock_logger)

        # Create a malformed CSV
        bad_csv = setup_mesh_directories['forcing_dir'] / 'MESH_output_streamflow.csv'
        bad_csv.write_text("not,a,valid,csv\n{{bad data}}\n")

        result = postprocessor.extract_streamflow()

        assert result is None
