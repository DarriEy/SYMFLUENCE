"""
Unit Tests for BaseAcquisitionHandler.

Tests the abstract base class functionality:
- Configuration parsing
- Bounding box handling
- Date range parsing
- Skip-if-exists logic
- Credential resolution
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import tempfile

import pandas as pd
import pytest

from fixtures.acquisition_fixtures import MockConfigFactory


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def concrete_handler_class():
    """
    Create a concrete implementation of BaseAcquisitionHandler for testing.

    Returns the class, not an instance.
    """
    from symfluence.data.acquisition.base import BaseAcquisitionHandler

    class ConcreteHandler(BaseAcquisitionHandler):
        """Concrete handler for testing base class functionality."""

        def download(self, output_dir: Path) -> Path:
            """Minimal implementation that creates a test file."""
            output_file = output_dir / "test_output.nc"
            output_file.touch()
            return output_file

    return ConcreteHandler


@pytest.fixture
def handler_instance(concrete_handler_class, mock_config, mock_logger):
    """Create an instance of the concrete handler."""
    return concrete_handler_class(mock_config, mock_logger)


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.acquisition
class TestBaseHandlerInitialization:
    """Tests for BaseAcquisitionHandler.__init__()."""

    def test_init_with_dict_config(self, concrete_handler_class, mock_logger):
        """Handler can be initialized with dict config."""
        config = MockConfigFactory.create()
        handler = concrete_handler_class(config, mock_logger)

        assert handler is not None
        assert handler.logger == mock_logger

    def test_init_with_symfluence_config(self, concrete_handler_class, mock_logger):
        """Handler can be initialized with SymfluenceConfig object."""
        from symfluence.core.config.models import SymfluenceConfig

        config_dict = MockConfigFactory.create()
        typed_config = SymfluenceConfig(**config_dict)

        handler = concrete_handler_class(typed_config, mock_logger)

        assert handler is not None
        assert handler._config is typed_config

    def test_init_parses_bbox(self, handler_instance):
        """Handler correctly parses bounding box from config."""
        bbox = handler_instance.bbox

        assert bbox is not None
        assert 'lat_min' in bbox
        assert 'lat_max' in bbox
        assert 'lon_min' in bbox
        assert 'lon_max' in bbox

        # Default bbox is "46.0/8.0/47.0/9.0"
        assert bbox['lat_min'] == 46.0
        assert bbox['lon_min'] == 8.0
        assert bbox['lat_max'] == 47.0
        assert bbox['lon_max'] == 9.0

    def test_init_parses_dates(self, handler_instance):
        """Handler correctly parses date range from config."""
        assert handler_instance.start_date is not None
        assert handler_instance.end_date is not None

        # Default dates are 2020-01-01 to 2020-01-31
        assert handler_instance.start_date == pd.Timestamp('2020-01-01')
        assert handler_instance.end_date == pd.Timestamp('2020-01-31')

    def test_init_with_reporting_manager(self, concrete_handler_class, mock_config, mock_logger):
        """Handler accepts optional reporting_manager."""
        mock_reporting_manager = MagicMock()

        handler = concrete_handler_class(
            mock_config, mock_logger, reporting_manager=mock_reporting_manager
        )

        assert handler.reporting_manager is mock_reporting_manager

    def test_init_without_reporting_manager(self, handler_instance):
        """Handler works without reporting_manager."""
        assert handler_instance.reporting_manager is None


# =============================================================================
# Bounding Box Parsing Tests
# =============================================================================

@pytest.mark.acquisition
class TestBoundingBoxParsing:
    """Tests for bounding box parsing."""

    def test_bbox_with_standard_format(self, concrete_handler_class, mock_logger):
        """Parse standard bbox format: north/west/south/east."""
        config = MockConfigFactory.create(bbox="46.0/10.0/45.0/11.0")
        handler = concrete_handler_class(config, mock_logger)

        assert handler.bbox['lat_min'] == 45.0
        assert handler.bbox['lon_min'] == 10.0
        assert handler.bbox['lat_max'] == 46.0
        assert handler.bbox['lon_max'] == 11.0

    def test_bbox_with_negative_longitude(self, concrete_handler_class, mock_logger):
        """Parse bbox with negative longitude."""
        # Format: north/west/south/east -> 41.0/-75.0/40.0/-74.0
        config = MockConfigFactory.create(bbox="41.0/-75.0/40.0/-74.0")
        handler = concrete_handler_class(config, mock_logger)

        assert handler.bbox['lon_min'] == -75.0
        assert handler.bbox['lon_max'] == -74.0

    def test_bbox_with_negative_latitude(self, concrete_handler_class, mock_logger):
        """Parse bbox with negative latitude (Southern hemisphere)."""
        # Format: north/west/south/east -> -34.0/18.0/-35.0/19.0
        config = MockConfigFactory.create(bbox="-34.0/18.0/-35.0/19.0")
        handler = concrete_handler_class(config, mock_logger)

        assert handler.bbox['lat_min'] == -35.0
        assert handler.bbox['lat_max'] == -34.0

    def test_bbox_values_are_floats(self, handler_instance):
        """Bbox values should be floats."""
        for key, value in handler_instance.bbox.items():
            assert isinstance(value, float), f"bbox[{key}] should be float"


# =============================================================================
# Skip-If-Exists Tests
# =============================================================================

@pytest.mark.acquisition
class TestSkipIfExists:
    """Tests for _skip_if_exists() method."""

    def test_skip_if_file_exists_and_not_forced(self, handler_instance, tmp_path):
        """Should skip if file exists and FORCE_DOWNLOAD is False."""
        test_file = tmp_path / "existing_file.nc"
        test_file.touch()

        # Default FORCE_DOWNLOAD is False
        result = handler_instance._skip_if_exists(test_file)

        assert result is True

    def test_no_skip_if_file_missing(self, handler_instance, tmp_path):
        """Should not skip if file doesn't exist."""
        test_file = tmp_path / "missing_file.nc"

        result = handler_instance._skip_if_exists(test_file)

        assert result is False

    def test_no_skip_if_forced(self, concrete_handler_class, mock_logger, tmp_path):
        """Should not skip if FORCE_DOWNLOAD is True."""
        config = MockConfigFactory.create(force_download=True)
        handler = concrete_handler_class(config, mock_logger)

        test_file = tmp_path / "existing_file.nc"
        test_file.touch()

        result = handler._skip_if_exists(test_file)

        assert result is False

    def test_skip_respects_force_override(self, handler_instance, tmp_path):
        """Force parameter should override config setting."""
        test_file = tmp_path / "existing_file.nc"
        test_file.touch()

        # Override with force=True
        result = handler_instance._skip_if_exists(test_file, force=True)

        assert result is False

    def test_skip_logs_message(self, concrete_handler_class, tmp_path, capturing_logger):
        """Should log info message when skipping."""
        logger, captured = capturing_logger
        config = MockConfigFactory.create()
        handler = concrete_handler_class(config, logger)

        test_file = tmp_path / "existing_file.nc"
        test_file.touch()

        handler._skip_if_exists(test_file)

        # Check that an info message was logged
        info_messages = [r for r in captured if r.levelno == 20]  # INFO level
        assert len(info_messages) > 0


# =============================================================================
# Download Method Tests
# =============================================================================

@pytest.mark.acquisition
class TestDownloadMethod:
    """Tests for download() method."""

    def test_download_creates_output(self, handler_instance, tmp_path):
        """Download should create output file and return path."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = handler_instance.download(output_dir)

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()

    def test_download_returns_path_type(self, handler_instance, tmp_path):
        """Download should return Path object."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = handler_instance.download(output_dir)

        assert isinstance(result, Path)


# =============================================================================
# Plot Diagnostics Tests
# =============================================================================

@pytest.mark.acquisition
class TestPlotDiagnostics:
    """Tests for plot_diagnostics() method."""

    def test_plot_diagnostics_without_manager(self, handler_instance, tmp_path):
        """plot_diagnostics should be no-op without reporting_manager."""
        test_file = tmp_path / "test.nc"
        test_file.touch()

        # Should not raise
        handler_instance.plot_diagnostics(test_file)

    def test_plot_diagnostics_with_manager(
        self, concrete_handler_class, mock_config, mock_logger, tmp_path
    ):
        """plot_diagnostics should call reporting_manager methods."""
        mock_manager = MagicMock()
        handler = concrete_handler_class(
            mock_config, mock_logger, reporting_manager=mock_manager
        )

        test_file = tmp_path / "test.tif"
        test_file.touch()

        handler.plot_diagnostics(test_file)

        # Should have called visualize_spatial_coverage
        mock_manager.visualize_spatial_coverage.assert_called_once()

    def test_plot_diagnostics_handles_csv(
        self, concrete_handler_class, mock_config, mock_logger, tmp_path
    ):
        """plot_diagnostics should handle CSV files."""
        import pandas as pd

        mock_manager = MagicMock()
        handler = concrete_handler_class(
            mock_config, mock_logger, reporting_manager=mock_manager
        )

        # Create a test CSV
        test_file = tmp_path / "test.csv"
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
        df.to_csv(test_file, index=False)

        handler.plot_diagnostics(test_file)

        # Should have called visualize_data_distribution
        mock_manager.visualize_data_distribution.assert_called()

    def test_plot_diagnostics_handles_errors_gracefully(
        self, concrete_handler_class, mock_config, mock_logger, tmp_path
    ):
        """plot_diagnostics should log warning on error, not raise."""
        mock_manager = MagicMock()
        mock_manager.visualize_spatial_coverage.side_effect = Exception("Test error")

        handler = concrete_handler_class(
            mock_config, mock_logger, reporting_manager=mock_manager
        )

        test_file = tmp_path / "test.tif"
        test_file.touch()

        # Should not raise
        handler.plot_diagnostics(test_file)


# =============================================================================
# Domain Directory Tests
# =============================================================================

@pytest.mark.acquisition
class TestDomainDirectory:
    """Tests for domain_dir property."""

    def test_domain_dir_returns_path(self, handler_instance):
        """domain_dir should return a Path."""
        result = handler_instance.domain_dir

        assert isinstance(result, Path)

    def test_domain_dir_creates_directory(self, concrete_handler_class, mock_logger, tmp_path):
        """domain_dir should create the directory if needed."""
        config = MockConfigFactory.create(data_dir=str(tmp_path / "data"))
        handler = concrete_handler_class(config, mock_logger)

        domain_dir = handler.domain_dir

        assert domain_dir.exists()


# =============================================================================
# Credential Resolution Tests
# =============================================================================

@pytest.mark.acquisition
class TestCredentialResolution:
    """Tests for _get_earthdata_credentials() method."""

    def test_credentials_from_environment(
        self, concrete_handler_class, mock_logger, mock_earthdata_env, tmp_path
    ):
        """Should get credentials from environment variables."""
        config = MockConfigFactory.create()

        # Patch home to avoid reading system .netrc
        with patch('pathlib.Path.home', return_value=tmp_path):
            handler = concrete_handler_class(config, mock_logger)
            username, password = handler._get_earthdata_credentials()

        assert username == "test_user"
        assert password == "test_password"

    def test_credentials_from_config(
        self, concrete_handler_class, mock_logger, clean_environment, tmp_path
    ):
        """Should get credentials from config when env vars not set."""
        config = MockConfigFactory.create_with_credentials(
            earthdata_user="config_user",
            earthdata_pass="config_pass"
        )

        # Patch home to avoid reading system .netrc
        with patch('pathlib.Path.home', return_value=tmp_path):
            handler = concrete_handler_class(config, mock_logger)
            username, password = handler._get_earthdata_credentials()

        assert username == "config_user"
        assert password == "config_pass"

    def test_credentials_returns_none_when_missing(
        self, concrete_handler_class, mock_logger, clean_environment, tmp_path
    ):
        """Should return (None, None) when no credentials available."""
        config = MockConfigFactory.create()

        # Patch home to avoid reading system .netrc
        with patch('pathlib.Path.home', return_value=tmp_path):
            handler = concrete_handler_class(config, mock_logger)
            username, password = handler._get_earthdata_credentials()

        # Both should be None when no credentials available
        assert username is None
        assert password is None


# =============================================================================
# Configuration Mixin Integration Tests
# =============================================================================

@pytest.mark.acquisition
class TestConfigurableMixinIntegration:
    """Tests for ConfigurableMixin integration."""

    def test_config_dict_property(self, handler_instance):
        """Handler should have config_dict property from mixin."""
        assert hasattr(handler_instance, 'config_dict')

        config_dict = handler_instance.config_dict

        assert isinstance(config_dict, dict)
        assert 'DOMAIN_NAME' in config_dict

    def test_project_dir_property(self, handler_instance):
        """Handler should have project_dir property from mixin."""
        assert hasattr(handler_instance, 'project_dir')

        project_dir = handler_instance.project_dir

        assert isinstance(project_dir, Path)
