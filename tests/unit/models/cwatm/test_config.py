"""Tests for CWatM configuration adapter."""
from __future__ import annotations

import pytest


class TestCWatMConfigAdapter:
    """Tests for CWatM configuration adapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.cwatm.config import CWatMConfigAdapter
        assert CWatMConfigAdapter is not None

    def test_adapter_initialization(self):
        from symfluence.models.cwatm.config import CWatMConfigAdapter
        adapter = CWatMConfigAdapter()
        assert adapter is not None
        assert adapter.model_name == 'CWATM'

    def test_adapter_returns_config_schema(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        from symfluence.models.cwatm.config import CWatMConfigAdapter
        adapter = CWatMConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema == CWatMConfig


class TestCWatMConfigValidation:
    """Tests for CWatM config validation."""

    def test_valid_ini_extension_accepted(self):
        from symfluence.models.cwatm.config import CWatMConfigAdapter
        adapter = CWatMConfigAdapter()
        adapter.validate({'config_file': 'settings.ini'})

    def test_invalid_config_extension_rejected(self):
        from symfluence.models.cwatm.config import CWatMConfigAdapter
        adapter = CWatMConfigAdapter()
        with pytest.raises(ValueError, match=".ini"):
            adapter.validate({'config_file': 'settings.yaml'})


class TestCWatMConfigDefaults:
    """Tests for CWatM config default values."""

    def test_config_default_exe(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert config.exe == 'run_cwatm.py'

    def test_config_default_resolution(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert config.resolution == '30min'

    def test_config_default_timeout(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert config.timeout == 14400

    def test_config_default_spatial_mode(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert config.spatial_mode == 'distributed'

    def test_config_default_pet_method(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert config.pet_method == 'hamon'

    def test_config_params_to_calibrate(self):
        from symfluence.core.config.models.model_configs import CWatMConfig
        config = CWatMConfig()
        assert 'SnowMeltCoef' in config.params_to_calibrate
        assert 'manningsN' in config.params_to_calibrate
