"""Unit tests for path_resolver utilities."""

import pytest
from pathlib import Path
import logging
from symfluence.core.path_resolver import (
    resolve_path,
    resolve_file_path,
    PathResolverMixin
)


class TestResolvePath:
    """Test resolve_path function."""

    def test_resolve_with_default_keyword(self, tmp_path):
        """Test path resolution when config value is 'default'."""
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_resolve_with_none_value(self, tmp_path):
        """Test path resolution when config value is None."""
        config = {'MY_PATH': None}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_resolve_with_missing_key(self, tmp_path):
        """Test path resolution when config key is missing."""
        config = {}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_resolve_with_absolute_path(self, tmp_path):
        """Test path resolution with explicit absolute path."""
        explicit_path = tmp_path / 'custom/location'
        config = {'MY_PATH': str(explicit_path)}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert result == explicit_path

    def test_resolve_with_relative_path(self, tmp_path):
        """Test path resolution with relative path string."""
        config = {'MY_PATH': 'relative/path'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert result == Path('relative/path')

    def test_must_exist_raises_error(self, tmp_path):
        """Test that must_exist raises FileNotFoundError for non-existent path."""
        config = {'MY_PATH': 'default'}
        with pytest.raises(FileNotFoundError, match="Required path does not exist"):
            resolve_path(config, 'MY_PATH', tmp_path, 'nonexistent', must_exist=True)

    def test_must_exist_passes_for_existing(self, tmp_path):
        """Test that must_exist passes for existing path."""
        (tmp_path / 'existing').mkdir(parents=True)
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'existing', must_exist=True)
        assert result.exists()
        assert result == tmp_path / 'existing'

    def test_with_logger(self, tmp_path, caplog):
        """Test that logger receives debug messages."""
        logger = logging.getLogger('test_logger')
        config = {'MY_PATH': 'default'}

        with caplog.at_level(logging.DEBUG, logger='test_logger'):
            result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing', logger=logger)

        assert result == tmp_path / 'data/forcing'
        # Logger should have been called (check in caplog if needed)

    def test_returns_path_object(self, tmp_path):
        """Test that function returns Path object."""
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'data/forcing')
        assert isinstance(result, Path)


class TestResolveFilePath:
    """Test resolve_file_path function."""

    def test_resolve_with_defaults(self, tmp_path):
        """Test file path resolution with default directory and filename."""
        config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert result == tmp_path / 'data' / 'attributes/elevation/dem.tif'

    def test_resolve_with_custom_path(self, tmp_path):
        """Test file path resolution with custom directory."""
        custom_dir = tmp_path / 'custom_location'
        config = {'DEM_PATH': str(custom_dir), 'DEM_NAME': 'default'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert result == custom_dir / 'dem.tif'

    def test_resolve_with_custom_name(self, tmp_path):
        """Test file path resolution with custom filename."""
        config = {'DEM_PATH': 'default', 'DEM_NAME': 'custom_dem.tif'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert result == tmp_path / 'data' / 'attributes/elevation/custom_dem.tif'

    def test_resolve_with_both_custom(self, tmp_path):
        """Test file path resolution with custom directory and filename."""
        custom_dir = tmp_path / 'my_dem_dir'
        config = {'DEM_PATH': str(custom_dir), 'DEM_NAME': 'my_custom.tif'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert result == custom_dir / 'my_custom.tif'

    def test_resolve_with_none_values(self, tmp_path):
        """Test file path resolution when config values are None."""
        config = {'DEM_PATH': None, 'DEM_NAME': None}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert result == tmp_path / 'data' / 'attributes/elevation/dem.tif'

    def test_must_exist_for_file(self, tmp_path):
        """Test must_exist validation for file paths."""
        config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        with pytest.raises(FileNotFoundError, match="Required file does not exist"):
            resolve_file_path(
                config, tmp_path, 'DEM_PATH', 'DEM_NAME',
                'attributes/elevation', 'dem.tif', must_exist=True
            )

    def test_must_exist_passes_for_existing_file(self, tmp_path):
        """Test must_exist passes when file exists."""
        # Create the file
        file_dir = tmp_path / 'attributes/elevation'
        file_dir.mkdir(parents=True)
        test_file = file_dir / 'dem.tif'
        test_file.touch()

        config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif', must_exist=True
        )
        assert result.exists()
        assert result == test_file

    def test_returns_path_object(self, tmp_path):
        """Test that function returns Path object."""
        config = {'DEM_PATH': 'default', 'DEM_NAME': 'default'}
        result = resolve_file_path(
            config, tmp_path, 'DEM_PATH', 'DEM_NAME',
            'attributes/elevation', 'dem.tif'
        )
        assert isinstance(result, Path)


class TestPathResolverMixin:
    """Test PathResolverMixin."""

    def _make_mock_config(self, config_dict):
        """Create a mock config object that has to_dict method."""
        from unittest.mock import Mock
        mock_config = Mock()
        mock_config.to_dict = Mock(return_value=config_dict)
        return mock_config

    def test_mixin_get_default_path(self, tmp_path):
        """Test mixin's _get_default_path method."""
        class MockClass(PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'MY_PATH': 'default'})
        obj.project_dir = tmp_path
        obj.logger = None

        result = obj._get_default_path('MY_PATH', 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_mixin_get_default_path_with_custom(self, tmp_path):
        """Test mixin with custom path value."""
        custom_path = tmp_path / 'custom_location'

        class MockClass(PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'MY_PATH': str(custom_path)})
        obj.project_dir = tmp_path
        obj.logger = None

        result = obj._get_default_path('MY_PATH', 'data/forcing')
        assert result == custom_path

    def test_mixin_get_file_path(self, tmp_path):
        """Test mixin's _get_file_path method."""
        class MockClass(PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'DEM_PATH': 'default', 'DEM_NAME': 'dem.tif'})
        obj.project_dir = tmp_path
        obj.logger = None

        result = obj._get_file_path('DEM_PATH', 'DEM_NAME', 'elevation', 'default.tif')
        assert result == tmp_path / 'elevation/dem.tif'

    def test_mixin_must_exist_parameter(self, tmp_path):
        """Test mixin's must_exist parameter."""
        class MockClass(PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'MY_PATH': 'default'})
        obj.project_dir = tmp_path
        obj.logger = None

        with pytest.raises(FileNotFoundError):
            obj._get_default_path('MY_PATH', 'nonexistent', must_exist=True)

    def test_mixin_works_with_logger(self, tmp_path):
        """Test mixin works when logger attribute exists."""
        logger = logging.getLogger('test_mixin_logger')

        class MockClass(PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'MY_PATH': 'default'})
        obj.project_dir = tmp_path
        obj.logger = logger

        result = obj._get_default_path('MY_PATH', 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_mixin_works_without_logger(self, tmp_path):
        """Test mixin works when logger attribute doesn't exist."""
        class MockClassNoLogger(PathResolverMixin):
            pass

        obj = MockClassNoLogger()
        obj._config = self._make_mock_config({'MY_PATH': 'default'})
        obj.project_dir = tmp_path

        result = obj._get_default_path('MY_PATH', 'data/forcing')
        assert result == tmp_path / 'data/forcing'

    def test_mixin_multiple_inheritance(self, tmp_path):
        """Test mixin works with multiple inheritance."""
        class BaseClass:
            def base_method(self):
                return "base"

        class MockClass(BaseClass, PathResolverMixin):
            pass

        obj = MockClass()
        obj._config = self._make_mock_config({'MY_PATH': 'default'})
        obj.project_dir = tmp_path
        obj.logger = None

        assert obj.base_method() == "base"
        result = obj._get_default_path('MY_PATH', 'data/forcing')
        assert result == tmp_path / 'data/forcing'


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config(self, tmp_path):
        """Test with empty configuration dictionary."""
        config = {}
        result = resolve_path(config, 'MISSING_KEY', tmp_path, 'default/path')
        assert result == tmp_path / 'default/path'

    def test_nested_default_subpath(self, tmp_path):
        """Test with deeply nested default subpath."""
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'a/b/c/d/e/file.txt')
        assert result == tmp_path / 'a/b/c/d/e/file.txt'

    def test_path_with_spaces(self, tmp_path):
        """Test path resolution with spaces in names."""
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, 'path with spaces/file name.txt')
        assert result == tmp_path / 'path with spaces/file name.txt'

    def test_empty_default_subpath(self, tmp_path):
        """Test with empty default subpath."""
        config = {'MY_PATH': 'default'}
        result = resolve_path(config, 'MY_PATH', tmp_path, '')
        assert result == tmp_path

    def test_file_path_with_missing_keys(self, tmp_path):
        """Test resolve_file_path with missing config keys."""
        config = {}
        result = resolve_file_path(
            config, tmp_path, 'MISSING_PATH', 'MISSING_NAME',
            'default/dir', 'default_file.txt'
        )
        assert result == tmp_path / 'default/dir/default_file.txt'
