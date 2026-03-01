"""Tests for HYPE model configuration."""

import pytest

from symfluence.models.base import ConfigValidationError


class TestHYPEConfigAdapter:
    """Tests for HYPEConfigAdapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        assert HYPEConfigAdapter is not None

    def test_adapter_creation(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        assert adapter is not None

    def test_get_config_schema_returns_hype_config(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema is not None
        assert schema.__name__ == "HYPEConfig"

    def test_required_keys_includes_settings_path(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        keys = adapter.get_required_keys()
        assert "SETTINGS_HYPE_PATH" in keys

    def test_validate_passes_with_required_keys(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        config = {"SETTINGS_HYPE_PATH": "/path/to/hype"}
        adapter.validate(config)

    def test_validate_raises_on_missing_keys(self):
        from symfluence.models.hype.config import HYPEConfigAdapter
        adapter = HYPEConfigAdapter()
        config = {"SETTINGS_HYPE_PATH": None}
        with pytest.raises(ConfigValidationError, match="Missing required"):
            adapter.validate(config)
