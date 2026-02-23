"""Unit tests for CLI validators."""

import pytest
from symfluence.cli.validators import (
    validate_bounding_box,
    validate_config_exists,
    validate_coordinates,
    validate_directory_exists,
    validate_file_exists,
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
        result = validate_coordinates(coords)
        assert result.is_ok == expected_valid
        if not result.is_ok:
            assert result.first_error() is not None

    def test_coordinate_error_messages(self):
        """Test that error messages are informative."""
        result = validate_coordinates("91/0")
        assert "out of range" in result.first_error().message.lower()

        result = validate_coordinates("lat/lon")
        assert "numeric" in result.first_error().message.lower()


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
        result = validate_bounding_box(bbox)
        assert result.is_ok == expected_valid
        if not result.is_ok:
            assert result.first_error() is not None


class TestFileValidation:
    """Test file existence validation."""

    def test_valid_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("test content")

        result = validate_file_exists(str(test_file))
        assert result.is_ok is True
        assert result.errors == ()

    def test_missing_file(self):
        """Test validation of missing file."""
        result = validate_file_exists("/nonexistent/file.yaml")
        assert result.is_ok is False
        assert "not found" in result.first_error().message.lower()

    def test_directory_not_file(self, tmp_path):
        """Test that directory is not validated as file."""
        result = validate_file_exists(str(tmp_path))
        assert result.is_ok is False
        assert "not a file" in result.first_error().message.lower()


class TestConfigValidation:
    """Test config file validation."""

    def test_valid_config(self, tmp_path):
        """Test validation of existing config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("DOMAIN_NAME: test")

        result = validate_config_exists(str(config_file))
        assert result.is_ok is True
        assert result.errors == ()

    def test_missing_config(self):
        """Test validation of missing config file."""
        result = validate_config_exists("/nonexistent/config.yaml")
        assert result.is_ok is False
        assert "not found" in result.first_error().message.lower()


class TestDirectoryValidation:
    """Test directory existence validation."""

    def test_valid_directory(self, tmp_path):
        """Test validation of existing directory."""
        result = validate_directory_exists(str(tmp_path))
        assert result.is_ok is True
        assert result.errors == ()

    def test_missing_directory(self):
        """Test validation of missing directory."""
        result = validate_directory_exists("/nonexistent/directory")
        assert result.is_ok is False
        assert "not found" in result.first_error().message.lower()

    def test_file_not_directory(self, tmp_path):
        """Test that file is not validated as directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = validate_directory_exists(str(test_file))
        assert result.is_ok is False
        assert "not a directory" in result.first_error().message.lower()
