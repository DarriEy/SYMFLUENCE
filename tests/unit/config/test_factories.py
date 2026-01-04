"""
Unit tests for configuration factory methods.

Tests the three factory methods for creating SymfluenceConfig instances:
- from_file(): Load from YAML with 5-layer hierarchy
- from_preset(): Load from named preset
- from_minimal(): Create minimal config with smart defaults
"""

import pytest
from pathlib import Path
import tempfile
import os
from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.exceptions import ConfigurationError


class TestFromMinimalFactory:
    """Test from_minimal() factory method"""

    def test_minimal_config_creation(self):
        """Test creating minimal config with required overrides"""
        config = SymfluenceConfig.from_minimal(
            domain_name='test_basin',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # Verify basic fields
        assert config.domain.name == 'test_basin'
        assert config.model.hydrological_model == 'SUMMA'
        assert config.domain.time_start == '2020-01-01 00:00'
        assert config.domain.time_end == '2020-12-31 23:00'

        # Verify defaults were applied
        assert config.forcing.dataset == 'ERA5'  # Default forcing
        assert config.domain.experiment_id == 'run_1'  # Default experiment ID

    def test_minimal_config_with_additional_overrides(self):
        """Test minimal config with additional overrides"""
        config = SymfluenceConfig.from_minimal(
            domain_name='test_basin',
            model='FUSE',
            forcing_dataset='NLDAS',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00',
            POUR_POINT_COORDS='40.5/-111.0',
            MPI_PROCESSES=8
        )

        assert config.domain.name == 'test_basin'
        assert config.model.hydrological_model == 'FUSE'
        assert config.forcing.dataset == 'NLDAS'
        assert config.domain.pour_point_coords == '40.5/-111.0'
        assert config.system.mpi_processes == 8

    def test_minimal_config_missing_required_fields(self):
        """Test that minimal config raises error if required fields missing"""
        # Missing EXPERIMENT_TIME_START and EXPERIMENT_TIME_END
        with pytest.raises(ConfigurationError, match="Missing required fields"):
            SymfluenceConfig.from_minimal(
                domain_name='test_basin',
                model='SUMMA'
            )

    def test_minimal_config_model_specific_defaults(self):
        """Test that model-specific defaults are applied"""
        summa_config = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # SUMMA should have its defaults
        assert summa_config.model.summa is not None
        assert summa_config.model.summa.exe == 'summa_sundials.exe'

        fuse_config = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='FUSE',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # FUSE should have its defaults
        assert fuse_config.model.fuse is not None
        assert fuse_config.model.fuse.exe == 'fuse.exe'


class TestFromFileFactory:
    """Test from_file() factory method"""

    def create_temp_config(self, content: str) -> Path:
        """Helper to create temporary config file"""
        fd, path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return Path(path)

    def test_from_file_basic(self):
        """Test loading config from YAML file"""
        yaml_content = """
SYMFLUENCE_DATA_DIR: /data
SYMFLUENCE_CODE_DIR: /code
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
DOMAIN_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            config = SymfluenceConfig.from_file(config_path)

            assert config.domain.name == 'test_basin'
            assert config.domain.experiment_id == 'run_1'
            assert config.model.hydrological_model == 'SUMMA'
            assert config.forcing.dataset == 'ERA5'
        finally:
            config_path.unlink()

    def test_from_file_with_overrides(self):
        """Test loading config with CLI overrides"""
        yaml_content = """
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
DOMAIN_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
MPI_PROCESSES: 1
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            config = SymfluenceConfig.from_file(
                config_path,
                overrides={'MPI_PROCESSES': 8, 'DEBUG_MODE': True}
            )

            # Override should take precedence
            assert config.system.mpi_processes == 8
            assert config.system.debug_mode is True

            # File values should still be present
            assert config.domain.name == 'test_basin'
        finally:
            config_path.unlink()

    def test_from_file_missing_file(self):
        """Test that from_file raises error for missing file"""
        with pytest.raises(FileNotFoundError):
            SymfluenceConfig.from_file(Path('/nonexistent/config.yaml'))

    def test_from_file_environment_variables(self, monkeypatch):
        """Test that environment variables override file values"""
        yaml_content = """
DOMAIN_NAME: test_basin
EXPERIMENT_ID: run_1
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: lumped
DOMAIN_DISCRETIZATION: lumped
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
MPI_PROCESSES: 1
"""
        config_path = self.create_temp_config(yaml_content)

        try:
            # Set environment variable
            monkeypatch.setenv('SYMFLUENCE_MPI_PROCESSES', '16')

            config = SymfluenceConfig.from_file(config_path, use_env=True)

            # Environment variable should override file value
            assert config.system.mpi_processes == 16
        finally:
            config_path.unlink()


class TestFromPresetFactory:
    """Test from_preset() factory method"""

    def test_from_preset_basic(self):
        """Test loading config from preset with required overrides"""
        # Presets don't include all required fields - user must provide them
        try:
            config = SymfluenceConfig.from_preset(
                'fuse-basic',
                DOMAIN_NAME='test_basin',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00'
            )

            # Should have FUSE model configured
            assert 'FUSE' in config.model.hydrological_model or config.model.hydrological_model == 'FUSE'

            # Should have all required fields
            assert config.domain.name == 'test_basin'
            assert config.domain.time_start == '2020-01-01 00:00'
            assert config.domain.time_end == '2020-12-31 23:00'

        except (ConfigurationError, ValueError) as e:
            if "not found" in str(e).lower():
                pytest.skip("Preset 'fuse-basic' not available")
            else:
                raise

    def test_from_preset_with_overrides(self):
        """Test loading preset with overrides"""
        try:
            config = SymfluenceConfig.from_preset(
                'fuse-basic',
                DOMAIN_NAME='custom_basin',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00',
                MPI_PROCESSES=16
            )

            # Overrides should take precedence
            assert config.domain.name == 'custom_basin'
            assert config.system.mpi_processes == 16

        except (ConfigurationError, ValueError) as e:
            if "not found" in str(e).lower():
                pytest.skip("Preset 'fuse-basic' not available")
            else:
                raise

    def test_from_preset_invalid_preset(self):
        """Test that invalid preset name raises error"""
        with pytest.raises(ConfigurationError, match="not found"):
            SymfluenceConfig.from_preset('nonexistent_preset')


class TestFactoryRoundTrip:
    """Test round-trip compatibility between factories and to_dict()"""

    def test_minimal_to_dict_round_trip(self):
        """Test that minimal config can be converted to dict and back"""
        config1 = SymfluenceConfig.from_minimal(
            domain_name='test',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00'
        )

        # Convert to flat dict
        flat = config1.to_dict(flatten=True)

        # Verify dict-like access works
        assert flat['DOMAIN_NAME'] == 'test'
        assert flat['HYDROLOGICAL_MODEL'] == 'SUMMA'

        # Verify bracket access works
        assert config1['DOMAIN_NAME'] == flat['DOMAIN_NAME']
        assert config1['EXPERIMENT_ID'] == flat['EXPERIMENT_ID']
