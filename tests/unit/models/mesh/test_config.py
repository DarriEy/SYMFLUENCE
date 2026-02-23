"""
Tests for MESH configuration.
"""

import pytest

from symfluence.core.config.models import SymfluenceConfig


class TestMESHConfig:
    """Tests for MESH configuration."""

    def test_mesh_spatial_mode_configured(self, mesh_config):
        """Test MESH spatial mode configuration."""
        assert mesh_config.model.mesh.spatial_mode == 'distributed'

    def test_mesh_exe_default(self, mesh_config):
        """Test MESH executable default value."""
        assert mesh_config.model.mesh.exe == 'mesh.exe'

    def test_mesh_params_to_calibrate(self, mesh_config):
        """Test MESH calibration parameters."""
        assert mesh_config.model.mesh.params_to_calibrate == 'ZSNL,MANN,RCHARG'

    def test_mesh_install_path_default(self, mesh_config):
        """Test MESH install path default."""
        assert mesh_config.model.mesh.install_path == 'default'


class TestMESHConfigValidation:
    """Tests for MESH configuration validation."""

    def test_valid_mesh_config_passes(self, temp_dir):
        """Test that valid MESH config passes validation."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'MESH',
            'FORCING_DATASET': 'ERA5',
            'MESH_SPATIAL_MODE': 'lumped',
            'MESH_SPINUP_DAYS': 730,
        }
        config = SymfluenceConfig(**config_dict)
        assert config.model.mesh.spatial_mode == 'lumped'

    def test_mesh_supports_lumped_mode(self, temp_dir):
        """Test MESH supports lumped spatial mode."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'MESH',
            'FORCING_DATASET': 'ERA5',
            'MESH_SPATIAL_MODE': 'lumped',
        }
        config = SymfluenceConfig(**config_dict)
        assert config.model.mesh.spatial_mode == 'lumped'

    def test_mesh_auto_spatial_mode(self, temp_dir):
        """Test MESH auto spatial mode determination."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(temp_dir / 'data'),
            'SYMFLUENCE_CODE_DIR': str(temp_dir / 'code'),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'MESH',
            'FORCING_DATASET': 'ERA5',
            'MESH_SPATIAL_MODE': 'auto',
        }
        config = SymfluenceConfig(**config_dict)
        assert config.model.mesh.spatial_mode == 'auto'


class TestMESHConfigAdapter:
    """Tests for MESH configuration adapter."""

    def test_adapter_can_be_imported(self):
        """Test that MESHConfigAdapter can be imported."""
        from symfluence.models.mesh.config import MESHConfigAdapter
        assert MESHConfigAdapter is not None

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        from symfluence.models.mesh.config import MESHConfigAdapter
        adapter = MESHConfigAdapter()
        assert adapter is not None
        assert adapter.model_name == 'MESH'

    def test_adapter_returns_config_schema(self):
        """Test adapter returns config schema."""
        from symfluence.core.config.models.model_configs import MESHConfig
        from symfluence.models.mesh.config import MESHConfigAdapter

        adapter = MESHConfigAdapter()
        schema = adapter.get_config_schema()
        assert schema == MESHConfig
