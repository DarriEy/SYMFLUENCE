"""Tests for symfluence.core.path_resolver module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.path_resolver import resolve_file_path, resolve_path

# =============================================================================
# resolve_path
# =============================================================================

class TestResolvePath:
    """Tests for resolve_path standalone function."""

    def test_default_keyword_uses_default_path(self, tmp_path):
        config = {"MY_PATH": "default"}
        result = resolve_path(config, "MY_PATH", tmp_path, "settings/model")
        assert result == tmp_path / "settings" / "model"

    def test_none_uses_default_path(self, tmp_path):
        config = {}
        result = resolve_path(config, "MY_PATH", tmp_path, "settings/model")
        assert result == tmp_path / "settings" / "model"

    def test_explicit_path_overrides_default(self, tmp_path):
        config = {"MY_PATH": "/custom/path"}
        result = resolve_path(config, "MY_PATH", tmp_path, "settings/model")
        assert result == Path("/custom/path")

    @patch("symfluence.core.path_resolver.resolve_data_subdir")
    def test_forcing_subpath_uses_data_subdir(self, mock_resolve, tmp_path):
        mock_resolve.return_value = tmp_path / "data" / "forcing"
        config = {"FORCING_PATH": "default"}
        result = resolve_path(config, "FORCING_PATH", tmp_path, "forcing/merged_data")
        mock_resolve.assert_called_once_with(tmp_path, "forcing")
        assert result == tmp_path / "data" / "forcing" / "merged_data"

    @patch("symfluence.core.path_resolver.resolve_data_subdir")
    def test_attributes_subpath_uses_data_subdir(self, mock_resolve, tmp_path):
        mock_resolve.return_value = tmp_path / "data" / "attributes"
        config = {}
        result = resolve_path(config, "ATTR_PATH", tmp_path, "attributes/elevation")
        mock_resolve.assert_called_once_with(tmp_path, "attributes")

    def test_non_data_subpath_uses_project_dir(self, tmp_path):
        config = {}
        result = resolve_path(config, "SETTINGS", tmp_path, "settings/SUMMA")
        assert result == tmp_path / "settings" / "SUMMA"

    def test_must_exist_raises_when_missing(self, tmp_path):
        config = {}
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_path(config, "X", tmp_path, "missing/path", must_exist=True)

    def test_must_exist_passes_when_exists(self, tmp_path):
        target = tmp_path / "exists"
        target.mkdir(parents=True)
        config = {}
        result = resolve_path(config, "X", tmp_path, "exists", must_exist=True)
        assert result == target

    def test_logs_default_path(self, tmp_path):
        logger = MagicMock()
        resolve_path({}, "X", tmp_path, "sub", logger=logger)
        logger.debug.assert_called()

    def test_logs_configured_path(self, tmp_path):
        logger = MagicMock()
        resolve_path({"X": "/custom"}, "X", tmp_path, "sub", logger=logger)
        logger.debug.assert_called()


# =============================================================================
# resolve_file_path
# =============================================================================

class TestResolveFilePath:
    """Tests for resolve_file_path standalone function."""

    def test_both_defaults(self, tmp_path):
        config = {"DEM_PATH": "default", "DEM_NAME": "default"}
        result = resolve_file_path(
            config, tmp_path, "DEM_PATH", "DEM_NAME",
            "settings/dem", "elevation.tif"
        )
        assert result == tmp_path / "settings" / "dem" / "elevation.tif"

    def test_explicit_dir_and_name(self, tmp_path):
        config = {"DEM_PATH": "/custom/dem", "DEM_NAME": "my_dem.tif"}
        result = resolve_file_path(
            config, tmp_path, "DEM_PATH", "DEM_NAME",
            "settings/dem", "elevation.tif"
        )
        assert result == Path("/custom/dem/my_dem.tif")

    def test_default_dir_custom_name(self, tmp_path):
        config = {"DEM_NAME": "custom.tif"}
        result = resolve_file_path(
            config, tmp_path, "DEM_PATH", "DEM_NAME",
            "settings/dem", "elevation.tif"
        )
        assert result.name == "custom.tif"

    def test_must_exist_raises_when_missing(self, tmp_path):
        config = {}
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_file_path(
                config, tmp_path, "P", "N", "sub", "file.txt",
                must_exist=True
            )

    def test_must_exist_passes_when_exists(self, tmp_path):
        target_dir = tmp_path / "sub"
        target_dir.mkdir()
        (target_dir / "file.txt").write_text("data")
        config = {}
        result = resolve_file_path(
            config, tmp_path, "P", "N", "sub", "file.txt",
            must_exist=True
        )
        assert result.exists()
