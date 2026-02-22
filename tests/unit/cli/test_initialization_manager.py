"""Unit tests for InitializationManager."""

import pytest
from unittest.mock import patch
import yaml

from symfluence.cli.services import InitializationManager

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


@pytest.fixture
def init_manager():
    """Create InitializationManager instance for testing.

    Injects a capture-friendly Console instance.
    """
    import sys
    from symfluence.cli.console import Console, ConsoleConfig

    # Configure console to use stdout/stderr which capsys captures
    console_config = ConsoleConfig(
        output_stream=sys.stdout,
        error_stream=sys.stderr,
        use_colors=False,
        show_progress=False
    )
    console = Console(console_config)

    return InitializationManager(console=console)


@pytest.fixture
def sample_preset():
    """Sample preset for testing."""
    return {
        'description': 'Test preset',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'run_1',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FORCING_DATASET': 'ERA5',
            'EXPERIMENT_TIME_START': '2010-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00'
        },
        'fuse_decisions': {
            'RFERR': ['multiplc_e'],
            'SNOWM': ['temp_index']
        }
    }


@pytest.fixture
def sample_config():
    """Sample config dict for testing."""
    return {
        'SYMFLUENCE_DATA_DIR': '/test/data',
        'SYMFLUENCE_CODE_DIR': '/test/code',
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'run_1',
        'EXPERIMENT_TIME_START': '2010-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-12-31 23:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'HYDROLOGICAL_MODEL': 'FUSE',
        'FORCING_DATASET': 'ERA5',
        'NUM_PROCESSES': 1
    }


class TestInitialization:
    """Test InitializationManager initialization."""

    def test_initialization(self, init_manager):
        """Test InitializationManager initializes correctly."""
        assert init_manager.presets is not None
        assert isinstance(init_manager.presets, dict)
        assert len(init_manager.presets) > 0

    def test_model_defaults_loaded(self, init_manager):
        """Test that model defaults are loaded correctly."""
        assert 'FUSE' in init_manager.model_defaults
        assert 'SUMMA' in init_manager.model_defaults

    def test_forcing_defaults_loaded(self, init_manager):
        """Test forcing defaults are loaded."""
        assert 'ERA5' in init_manager.forcing_defaults
        assert 'CONUS404' in init_manager.forcing_defaults


class TestListPresets:
    """Test list_presets functionality."""

    def test_list_presets_prints_output(self, init_manager, capsys):
        """Test list_presets prints preset information."""
        import sys
        from symfluence.cli.console import Console, ConsoleConfig

        init_manager._console = Console(ConsoleConfig(
            output_stream=sys.stdout,
            error_stream=sys.stderr,
            use_colors=False
        ))

        init_manager.list_presets()
        captured = capsys.readouterr()

        assert 'Available Presets:' in captured.out
        assert 'fuse-provo' in captured.out
        assert 'summa-basic' in captured.out


class TestShowPreset:
    """Test show_preset functionality."""

    def test_show_preset_valid_name(self, init_manager, capsys):
        """Test show_preset displays preset information."""
        import sys
        from symfluence.cli.console import Console, ConsoleConfig

        init_manager._console = Console(ConsoleConfig(
            output_stream=sys.stdout,
            error_stream=sys.stderr,
            use_colors=False
        ))

        init_manager.show_preset('fuse-provo')
        captured = capsys.readouterr()

        assert 'Preset: fuse-provo' in captured.out
        assert 'Description:' in captured.out
        assert 'Key Settings:' in captured.out

    def test_show_preset_invalid_name(self, init_manager, capsys):
        """Test show_preset handles invalid preset name."""
        import sys
        from symfluence.cli.console import Console, ConsoleConfig

        init_manager._console = Console(ConsoleConfig(
            output_stream=sys.stdout,
            error_stream=sys.stderr,
            use_colors=False
        ))

        init_manager.show_preset('nonexistent')
        captured = capsys.readouterr()

        # The manager writes errors to its error_console which defaults to stderr
        # Since we redirected error_stream to sys.stderr (captured by capsys),
        # we check captured.err
        assert 'Unknown preset' in captured.err or 'Unknown preset' in captured.out


class TestParseCliOverrides:
    """Test _parse_cli_overrides functionality."""

    def test_parse_cli_overrides_domain(self, init_manager):
        """Test parsing domain override."""
        overrides = {'domain': 'my_domain'}
        result = init_manager._parse_cli_overrides(overrides)

        assert result['DOMAIN_NAME'] == 'my_domain'

    def test_parse_cli_overrides_model(self, init_manager):
        """Test parsing model override."""
        overrides = {'model': 'SUMMA'}
        result = init_manager._parse_cli_overrides(overrides)

        assert result['HYDROLOGICAL_MODEL'] == 'SUMMA'

    def test_parse_cli_overrides_dates(self, init_manager):
        """Test parsing date overrides."""
        overrides = {
            'start_date': '2015-01-01',
            'end_date': '2020-12-31'
        }
        result = init_manager._parse_cli_overrides(overrides)

        assert result['EXPERIMENT_TIME_START'] == '2015-01-01 00:00'
        assert result['EXPERIMENT_TIME_END'] == '2020-12-31 23:00'

    def test_parse_cli_overrides_forcing(self, init_manager):
        """Test parsing forcing override."""
        overrides = {'forcing': 'CONUS404'}
        result = init_manager._parse_cli_overrides(overrides)

        assert result['FORCING_DATASET'] == 'CONUS404'

    def test_parse_cli_overrides_discretization(self, init_manager):
        """Test parsing discretization override."""
        overrides = {'discretization': 'GRUs'}
        result = init_manager._parse_cli_overrides(overrides)

        assert result['SUB_GRID_DISCRETIZATION'] == 'GRUs'

    def test_parse_cli_overrides_definition_method(self, init_manager):
        """Test parsing definition method override."""
        overrides = {'definition_method': 'delineate'}
        result = init_manager._parse_cli_overrides(overrides)

        assert result['DOMAIN_DEFINITION_METHOD'] == 'delineate'

    def test_parse_cli_overrides_empty(self, init_manager):
        """Test parsing empty overrides."""
        overrides = {}
        result = init_manager._parse_cli_overrides(overrides)

        assert result == {}

    def test_parse_cli_overrides_none_values(self, init_manager):
        """Test parsing overrides with None values."""
        overrides = {
            'domain': None,
            'model': None
        }
        result = init_manager._parse_cli_overrides(overrides)

        assert 'DOMAIN_NAME' not in result
        assert 'HYDROLOGICAL_MODEL' not in result


class TestApplySmartDefaults:
    """Test _apply_smart_defaults functionality."""

    def test_apply_smart_defaults_fuse(self, init_manager):
        """Test applying FUSE smart defaults."""
        config = {'HYDROLOGICAL_MODEL': 'FUSE'}
        init_manager._apply_smart_defaults(config)

        assert config['FUSE_SPATIAL_MODE'] == 'lumped'
        assert config['ROUTING_MODEL'] == 'none'
        assert config['FUSE_INSTALL_PATH'] == 'default'

    def test_apply_smart_defaults_summa(self, init_manager):
        """Test applying SUMMA smart defaults."""
        config = {'HYDROLOGICAL_MODEL': 'SUMMA'}
        init_manager._apply_smart_defaults(config)

        assert config['ROUTING_MODEL'] == 'mizuRoute'
        assert config['SUMMA_INSTALL_PATH'] == 'default'

    def test_apply_smart_defaults_forcing(self, init_manager):
        """Test applying forcing smart defaults."""
        config = {'FORCING_DATASET': 'ERA5'}
        init_manager._apply_smart_defaults(config)

        assert config['FORCING_TIME_STEP_SIZE'] == 3600
        assert config['DATA_ACCESS'] == 'cloud'

    def test_apply_smart_defaults_common(self, init_manager):
        """Test applying common smart defaults."""
        config = {}
        init_manager._apply_smart_defaults(config)

        assert config['NUM_PROCESSES'] == 1
        assert config['FORCE_RUN_ALL_STEPS'] is False
        assert config['DATA_ACCESS'] == 'cloud'

    def test_apply_smart_defaults_preserves_existing(self, init_manager):
        """Test smart defaults don't override existing values."""
        config = {
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FUSE_SPATIAL_MODE': 'distributed',
            'NUM_PROCESSES': 4
        }
        init_manager._apply_smart_defaults(config)

        assert config['FUSE_SPATIAL_MODE'] == 'distributed'  # Preserved
        assert config['NUM_PROCESSES'] == 4  # Preserved
        assert config['ROUTING_MODEL'] == 'none'  # Added


class TestAutoSetPaths:
    """Test _auto_set_paths functionality."""

    def test_auto_set_paths_default_data_dir(self, init_manager):
        """Test auto-setting default data directory."""
        config = {'SYMFLUENCE_DATA_DIR': 'default'}
        init_manager._auto_set_paths(config)

        assert config['SYMFLUENCE_DATA_DIR'] != 'default'
        assert 'symfluence_data' in config['SYMFLUENCE_DATA_DIR']

    def test_auto_set_paths_missing_data_dir(self, init_manager):
        """Test auto-setting missing data directory."""
        config = {}
        init_manager._auto_set_paths(config)

        assert 'SYMFLUENCE_DATA_DIR' in config
        assert 'symfluence_data' in config['SYMFLUENCE_DATA_DIR']

    def test_auto_set_paths_default_code_dir(self, init_manager):
        """Test auto-setting default code directory."""
        config = {'SYMFLUENCE_CODE_DIR': 'default'}
        init_manager._auto_set_paths(config)

        assert config['SYMFLUENCE_CODE_DIR'] != 'default'
        assert isinstance(config['SYMFLUENCE_CODE_DIR'], str)

    def test_auto_set_paths_preserves_explicit_paths(self, init_manager):
        """Test explicit paths are preserved."""
        config = {
            'SYMFLUENCE_DATA_DIR': '/custom/data',
            'SYMFLUENCE_CODE_DIR': '/custom/code'
        }
        init_manager._auto_set_paths(config)

        assert config['SYMFLUENCE_DATA_DIR'] == '/custom/data'
        assert config['SYMFLUENCE_CODE_DIR'] == '/custom/code'


class TestValidateConfig:
    """Test _validate_config functionality."""

    def test_validate_config_valid(self, init_manager, sample_config):
        """Test validating a valid config."""
        # Should not raise
        init_manager._validate_config(sample_config)

    def test_validate_config_missing_required_field(self, init_manager):
        """Test validation fails for missing required field."""
        config = {
            'SYMFLUENCE_DATA_DIR': '/test',
            # Missing DOMAIN_NAME and other required fields
        }

        with pytest.raises(ValueError, match="Missing required field"):
            init_manager._validate_config(config)

    def test_validate_config_invalid_dates(self, init_manager, sample_config):
        """Test validation fails for invalid date range."""
        sample_config['EXPERIMENT_TIME_START'] = '2020-01-01 00:00'
        sample_config['EXPERIMENT_TIME_END'] = '2010-01-01 00:00'

        with pytest.raises(ValueError, match="End date must be after start date"):
            init_manager._validate_config(sample_config)

    def test_validate_config_invalid_model(self, init_manager, sample_config):
        """Test validation fails for invalid model."""
        sample_config['HYDROLOGICAL_MODEL'] = 'INVALID_MODEL'

        with pytest.raises(ValueError, match="Invalid model"):
            init_manager._validate_config(sample_config)


class TestGenerateConfig:
    """Test generate_config functionality."""

    @patch.object(InitializationManager, '_load_yaml')
    def test_generate_config_with_preset(self, mock_load_yaml, init_manager, sample_preset):
        """Test generating config with preset."""
        # Mock template loading
        mock_load_yaml.return_value = {'BASE_KEY': 'base_value'}

        # Mock get_preset from init_presets module
        with patch('symfluence.cli.init_presets.get_preset', return_value=sample_preset):
            config = init_manager.generate_config(
                preset_name='fuse-provo',
                cli_overrides={},
                minimal=False,
                comprehensive=False
            )

        assert config['DOMAIN_NAME'] == 'test_domain'
        assert config['HYDROLOGICAL_MODEL'] == 'FUSE'
        assert 'FUSE_DECISION_OPTIONS' in config

    @patch.object(InitializationManager, '_load_yaml')
    def test_generate_config_with_cli_overrides(self, mock_load_yaml, init_manager, sample_preset):
        """Test CLI overrides take precedence over preset."""
        mock_load_yaml.return_value = {}

        with patch('symfluence.cli.init_presets.get_preset', return_value=sample_preset):
            config = init_manager.generate_config(
                preset_name='fuse-provo',
                cli_overrides={'domain': 'custom_domain'},
                minimal=False,
                comprehensive=False
            )

        assert config['DOMAIN_NAME'] == 'custom_domain'  # Override wins

    def test_generate_config_minimal(self, init_manager):
        """Test generating minimal config."""
        config = init_manager.generate_config(
            preset_name=None,
            cli_overrides={'domain': 'test', 'model': 'FUSE', 'forcing': 'ERA5'},
            minimal=True,
            comprehensive=False
        )

        # Should have basic required fields
        assert 'DOMAIN_NAME' in config
        assert 'HYDROLOGICAL_MODEL' in config


class TestWriteConfig:
    """Test write_config functionality."""

    def test_write_config_creates_file(self, init_manager, sample_config, tmp_path):
        """Test write_config creates config file."""
        output_file = tmp_path / "config_test.yaml"

        result = init_manager.write_config(sample_config, output_file)

        assert result == output_file
        assert output_file.exists()

    def test_write_config_creates_directory(self, init_manager, sample_config, tmp_path):
        """Test write_config creates parent directory if needed."""
        output_file = tmp_path / "subdir" / "config_test.yaml"

        init_manager.write_config(sample_config, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_write_config_valid_yaml(self, init_manager, sample_config, tmp_path):
        """Test written config is valid YAML."""
        output_file = tmp_path / "config_test.yaml"

        init_manager.write_config(sample_config, output_file)

        # Should be able to load as YAML
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded['DOMAIN_NAME'] == sample_config['DOMAIN_NAME']

    def test_write_config_includes_header(self, init_manager, sample_config, tmp_path):
        """Test written config includes header comments."""
        output_file = tmp_path / "config_test.yaml"

        init_manager.write_config(sample_config, output_file)

        with open(output_file, 'r') as f:
            content = f.read()

        assert 'SYMFLUENCE Configuration File' in content
        assert 'generated using the --init command' in content


class TestCreateScaffold:
    """Test create_scaffold functionality."""

    def test_create_scaffold_creates_directories(self, init_manager, sample_config, tmp_path):
        """Test create_scaffold creates expected directories."""
        # Set data dir to tmp_path
        sample_config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

        result = init_manager.create_scaffold(sample_config)

        # Check domain directory was created
        domain_dir = tmp_path / f"domain_{sample_config['DOMAIN_NAME']}"
        assert domain_dir.exists()

        # Check subdirectories
        assert (domain_dir / 'shapefiles' / 'pour_point').exists()
        assert (domain_dir / 'shapefiles' / 'catchment').exists()
        assert (domain_dir / 'data' / 'forcing' / 'raw_data').exists()
        assert (domain_dir / 'settings').exists()

    def test_create_scaffold_fuse_model(self, init_manager, tmp_path):
        """Test create_scaffold creates FUSE directories."""
        config = {
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'run_1',
            'HYDROLOGICAL_MODEL': 'FUSE',
            'SYMFLUENCE_DATA_DIR': str(tmp_path)
        }

        init_manager.create_scaffold(config)

        domain_dir = tmp_path / 'domain_test'
        assert (domain_dir / 'settings' / 'FUSE').exists()

    def test_create_scaffold_summa_model(self, init_manager, tmp_path):
        """Test create_scaffold creates SUMMA directories."""
        config = {
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'run_1',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'ROUTING_MODEL': 'mizuRoute',
            'SYMFLUENCE_DATA_DIR': str(tmp_path)
        }

        init_manager.create_scaffold(config)

        domain_dir = tmp_path / 'domain_test'
        assert (domain_dir / 'settings' / 'SUMMA').exists()
        assert (domain_dir / 'settings' / 'mizuRoute').exists()

    def test_create_scaffold_raises_on_existing_without_force(self, init_manager, sample_config, tmp_path):
        """Test create_scaffold raises error if directory exists."""
        sample_config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

        # Create domain directory
        domain_dir = tmp_path / f"domain_{sample_config['DOMAIN_NAME']}"
        domain_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="Domain directory already exists"):
            init_manager.create_scaffold(sample_config, force=False)

    def test_create_scaffold_overwrites_with_force(self, init_manager, sample_config, tmp_path):
        """Test create_scaffold overwrites with force=True."""
        sample_config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

        # Create domain directory
        domain_dir = tmp_path / f"domain_{sample_config['DOMAIN_NAME']}"
        domain_dir.mkdir(parents=True)

        # Should not raise
        result = init_manager.create_scaffold(sample_config, force=True)
        assert result.exists()
