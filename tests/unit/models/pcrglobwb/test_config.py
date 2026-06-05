"""Tests for PCR-GLOBWB configuration adapter."""
from __future__ import annotations

import pytest


class TestPCRGLOBWBConfigAdapter:
    """Tests for PCR-GLOBWB configuration adapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        assert PCRGLOBWBConfigAdapter is not None

    def test_adapter_initialization(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        assert adapter is not None
        assert adapter.model_name == 'PCRGLOBWB'

    def test_adapter_returns_config_schema(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema == PCRGLOBWBConfig


class TestPCRGLOBWBConfigValidation:
    """Tests for PCR-GLOBWB config validation."""

    def test_valid_ini_extension_accepted(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        adapter.validate({'config_file': 'setup.ini'})

    def test_invalid_config_extension_rejected(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        with pytest.raises(ValueError, match=".ini"):
            adapter.validate({'config_file': 'setup.toml'})

    def test_valid_resolution_accepted(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        adapter.validate({'resolution': '05min'})
        adapter.validate({'resolution': '30min'})

    def test_invalid_resolution_rejected(self):
        from symfluence.models.pcrglobwb.config import PCRGLOBWBConfigAdapter
        adapter = PCRGLOBWBConfigAdapter()
        with pytest.raises(ValueError, match="resolution"):
            adapter.validate({'resolution': '10min'})


class TestPCRGLOBWBConfigDefaults:
    """Tests for PCR-GLOBWB config default values."""

    def test_config_has_python_exe_field(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.python_exe == 'python'

    def test_config_has_resolution_field(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.resolution == '05min'

    def test_config_has_timeout_field(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.timeout == 14400

    def test_config_has_spinup_fields(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.spinup_years == 0
        assert config.spinup_convergence is False

    def test_config_has_opendap_field(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.use_opendap is False

    def test_config_default_exe(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.exe == 'deterministic_runner.py'

    def test_config_default_spatial_mode(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.spatial_mode == 'distributed'

    def test_config_default_pet_method(self):
        from symfluence.core.config.models.model_configs import PCRGLOBWBConfig
        config = PCRGLOBWBConfig()
        assert config.pet_method == 'hamon'
