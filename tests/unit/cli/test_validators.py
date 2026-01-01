"""Unit tests for CLI validators."""

import pytest
from pathlib import Path

from symfluence.utils.cli.validators import (
    validate_coordinates,
    validate_bounding_box,
    validate_config_exists,
    validate_file_exists,
    validate_directory_exists
)

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestCoordinateValidation:
    """Test coordinate validation."""

    @pytest.mark.parametrize("coords,expected_valid", [
        ("51.1722/-115.5717", True),
        ("51.1722/115.5717", True),
        ("-51.1722/-115.5717", True),
        ("0/0", True),
        ("90/180", True),
        ("-90/-180", True),
        ("91/0", False),  # Lat out of range
        ("0/181", False),  # Lon out of range
        ("51.17", False),  # Missing longitude
        ("lat/lon", False),  # Non-numeric
        ("51.17/115.57/extra", False),  # Too many parts
    ])
    def test_coordinate_formats(self, coords, expected_valid):
        """Test various coordinate formats."""
        is_valid, error_msg = validate_coordinates(coords)
        assert is_valid == expected_valid
        if not is_valid:
            assert error_msg is not None

    def test_coordinate_error_messages(self):
        """Test that error messages are informative."""
        is_valid, error_msg = validate_coordinates("91/0")
        assert "out of range" in error_msg.lower()

        is_valid, error_msg = validate_coordinates("lat/lon")
        assert "numeric" in error_msg.lower()


class TestBoundingBoxValidation:
    """Test bounding box validation."""

    @pytest.mark.parametrize("bbox,expected_valid", [
        ("55.0/10.0/45.0/20.0", True),  # lat_max/lon_min/lat_min/lon_max
        ("90/-180/-90/180", True),  # Maximum extents
        ("50/0/40/10", True),
        ("40/10/50/20", False),  # lat_min > lat_max
        ("50/20/40/10", False),  # lon_min > lon_max
        ("91/0/45/10", False),  # lat_max out of range
        ("50/0/-91/10", False),  # lat_min out of range
        ("50/0/45", False),  # Missing component
    ])
    def test_bounding_box_formats(self, bbox, expected_valid):
        """Test various bounding box formats."""
        is_valid, error_msg = validate_bounding_box(bbox)
        assert is_valid == expected_valid
        if not is_valid:
            assert error_msg is not None


class TestFileValidation:
    """Test file existence validation."""

    def test_valid_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("test content")

        is_valid, error_msg = validate_file_exists(str(test_file))
        assert is_valid is True
        assert error_msg is None

    def test_missing_file(self):
        """Test validation of missing file."""
        is_valid, error_msg = validate_file_exists("/nonexistent/file.yaml")
        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_directory_not_file(self, tmp_path):
        """Test that directory is not validated as file."""
        is_valid, error_msg = validate_file_exists(str(tmp_path))
        assert is_valid is False
        assert "not a file" in error_msg.lower()


class TestConfigValidation:
    """Test config file validation."""

    def test_valid_config(self, tmp_path):
        """Test validation of existing config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("DOMAIN_NAME: test")

        is_valid, error_msg = validate_config_exists(str(config_file))
        assert is_valid is True
        assert error_msg is None

    def test_missing_config(self):
        """Test validation of missing config file."""
        is_valid, error_msg = validate_config_exists("/nonexistent/config.yaml")
        assert is_valid is False
        assert "not found" in error_msg.lower()


class TestDirectoryValidation:
    """Test directory existence validation."""

    def test_valid_directory(self, tmp_path):
        """Test validation of existing directory."""
        is_valid, error_msg = validate_directory_exists(str(tmp_path))
        assert is_valid is True
        assert error_msg is None

    def test_missing_directory(self):
        """Test validation of missing directory."""
        is_valid, error_msg = validate_directory_exists("/nonexistent/directory")
        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_file_not_directory(self, tmp_path):
        """Test that file is not validated as directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        is_valid, error_msg = validate_directory_exists(str(test_file))
        assert is_valid is False
        assert "not a directory" in error_msg.lower()
