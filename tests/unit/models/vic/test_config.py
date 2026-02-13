"""Tests for VIC configuration adapter."""

import pytest


class TestVICConfigAdapter:
    """Tests for VIC configuration adapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.vic.config import VICConfigAdapter
        assert VICConfigAdapter is not None

    def test_adapter_initialization(self):
        from symfluence.models.vic.config import VICConfigAdapter
        adapter = VICConfigAdapter()
        assert adapter is not None
        assert adapter.model_name == 'VIC'

    def test_adapter_returns_config_schema(self):
        from symfluence.models.vic.config import VICConfigAdapter
        from symfluence.core.config.models.model_configs import VICConfig
        adapter = VICConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema == VICConfig

    def test_adapter_registered_with_registry(self):
        from symfluence.models.registry import ModelRegistry
        assert 'VIC' in ModelRegistry._config_adapters


class TestVICConfigValidation:
    """Tests for VIC config validation."""

    def test_valid_driver_accepted(self):
        from symfluence.models.vic.config import VICConfigAdapter
        adapter = VICConfigAdapter()
        # Should not raise
        adapter.validate({'driver': 'image'})
        adapter.validate({'driver': 'classic'})

    def test_invalid_driver_rejected(self):
        from symfluence.models.vic.config import VICConfigAdapter
        adapter = VICConfigAdapter()
        with pytest.raises(ValueError, match="Invalid VIC driver"):
            adapter.validate({'driver': 'invalid'})

    def test_invalid_snow_bands_rejected(self):
        from symfluence.models.vic.config import VICConfigAdapter
        adapter = VICConfigAdapter()
        with pytest.raises(ValueError, match="n_snow_bands"):
            adapter.validate({'snow_band': True, 'n_snow_bands': 30})

    def test_snow_band_disabled_ignores_count(self):
        from symfluence.models.vic.config import VICConfigAdapter
        adapter = VICConfigAdapter()
        # Should not raise when snow_band is disabled
        adapter.validate({'snow_band': False, 'n_snow_bands': 30})


class TestVICConfigDefaults:
    """Tests for VIC config default values."""

    def test_vic_config_has_driver_field(self):
        from symfluence.core.config.models.model_configs import VICConfig
        config = VICConfig()
        assert config.driver == 'image'

    def test_vic_config_has_exe_field(self):
        from symfluence.core.config.models.model_configs import VICConfig
        config = VICConfig()
        assert config.exe == 'vic_image.exe'

    def test_vic_config_has_snow_band_field(self):
        from symfluence.core.config.models.model_configs import VICConfig
        config = VICConfig()
        assert config.snow_band is False
        assert config.n_snow_bands == 10
