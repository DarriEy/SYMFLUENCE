"""Tests for Noah-MP configuration adapter."""

import pytest


class TestNoahMPConfigAdapter:
    """Tests for Noah-MP configuration adapter."""

    def test_adapter_can_be_imported(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        assert NoahMPConfigAdapter is not None

    def test_adapter_initialization(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        assert adapter is not None
        assert adapter.model_name == 'NOAHMP'

    def test_adapter_returns_config_schema(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema == NoahMPConfig


class TestNoahMPConfigValidation:
    """Tests for Noah-MP config validation."""

    def test_valid_namelist_extension_accepted(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        adapter.validate({'namelist_file': 'namelist.input'})

    def test_invalid_namelist_extension_rejected(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        with pytest.raises(ValueError, match=".input"):
            adapter.validate({'namelist_file': 'namelist.txt'})

    def test_valid_timestep_accepted(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        adapter.validate({'namelist_file': 'namelist.input', 'timestep': 1800})

    def test_invalid_timestep_rejected(self):
        from symfluence.models.noahmp.config import NoahMPConfigAdapter
        adapter = NoahMPConfigAdapter()
        with pytest.raises(ValueError, match="timestep"):
            adapter.validate({'namelist_file': 'namelist.input', 'timestep': 600})


class TestNoahMPConfigDefaults:
    """Tests for Noah-MP config default values."""

    def test_config_default_exe(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.exe == 'noah_owp_modular.exe'

    def test_config_default_namelist(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.namelist_file == 'namelist.input'

    def test_config_default_timestep(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.timestep == 3600

    def test_config_default_nsoil(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.nsoil == 4

    def test_config_default_nsnow(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.nsnow == 3

    def test_config_default_timeout(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.timeout == 7200

    def test_config_default_runoff_option(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.runoff_option == 1

    def test_config_default_dynamic_veg_option(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.dynamic_veg_option == 4

    def test_config_default_spinup_loops(self):
        from symfluence.core.config.models.model_configs import NoahMPConfig
        config = NoahMPConfig()
        assert config.spinup_loops == 0
