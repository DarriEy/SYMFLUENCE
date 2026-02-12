"""Unit tests for CLI ArgumentParser."""

import pytest

from symfluence.cli.argument_parser import CLIParser

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestParserInitialization:
    """Test CLIParser initialization."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = CLIParser()
        assert parser.parser is not None

    def test_parser_has_subparsers(self):
        """Test that parser has subcommand structure."""
        parser = CLIParser()
        # Parser should require a subcommand
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestGlobalOptions:
    """Test global options available to all commands."""

    def test_config_option(self):
        """Test --config global option."""
        parser = CLIParser()
        args = parser.parse_args(['--config', 'test.yaml', 'workflow', 'list-steps'])
        assert args.config == 'test.yaml'

    def test_debug_option(self):
        """Test --debug global option."""
        parser = CLIParser()
        args = parser.parse_args(['--debug', 'workflow', 'list-steps'])
        assert args.debug is True

    def test_dry_run_option(self):
        """Test --dry-run global option."""
        parser = CLIParser()
        args = parser.parse_args(['--dry-run', 'workflow', 'run'])
        assert args.dry_run is True


class TestWorkflowCommands:
    """Test workflow category commands."""

    def test_workflow_run(self):
        """Test workflow run command."""
        parser = CLIParser()
        args = parser.parse_args(['workflow', 'run'])
        assert args.category == 'workflow'
        assert args.action == 'run'
        assert hasattr(args, 'func')

    def test_workflow_step(self):
        """Test workflow step command."""
        parser = CLIParser()
        args = parser.parse_args(['workflow', 'step', 'calibrate_model'])
        assert args.category == 'workflow'
        assert args.action == 'step'
        assert args.step_name == 'calibrate_model'

    def test_workflow_steps(self):
        """Test workflow steps command."""
        parser = CLIParser()
        args = parser.parse_args(['workflow', 'steps', 'setup_project', 'run_model'])
        assert args.category == 'workflow'
        assert args.action == 'steps'
        assert args.step_names == ['setup_project', 'run_model']

    def test_workflow_list_steps(self):
        """Test workflow list-steps command."""
        parser = CLIParser()
        args = parser.parse_args(['workflow', 'list-steps'])
        assert args.action == 'list-steps'


class TestProjectCommands:
    """Test project category commands."""

    def test_project_init(self):
        """Test project init command."""
        parser = CLIParser()
        args = parser.parse_args(['project', 'init', 'fuse-provo'])
        assert args.category == 'project'
        assert args.action == 'init'
        assert args.preset == 'fuse-provo'

    def test_project_list_presets(self):
        """Test project list-presets command."""
        parser = CLIParser()
        args = parser.parse_args(['project', 'list-presets'])
        assert args.action == 'list-presets'

    def test_project_pour_point(self):
        """Test project pour-point command."""
        parser = CLIParser()
        args = parser.parse_args([
            'project', 'pour-point', '51.17/-115.57',
            '--domain-name', 'Bow',
            '--definition', 'delineate'
        ])
        assert args.coordinates == '51.17/-115.57'
        assert args.domain_name == 'Bow'
        assert args.domain_def == 'delineate'


class TestBinaryCommands:
    """Test binary category commands."""

    def test_binary_install(self):
        """Test binary install command."""
        parser = CLIParser()
        args = parser.parse_args(['binary', 'install', 'summa', 'mizuroute'])
        assert args.category == 'binary'
        assert args.action == 'install'
        assert args.tools == ['summa', 'mizuroute']

    def test_binary_validate(self):
        """Test binary validate command."""
        parser = CLIParser()
        args = parser.parse_args(['binary', 'validate'])
        assert args.action == 'validate'

    def test_binary_doctor(self):
        """Test binary doctor command."""
        parser = CLIParser()
        args = parser.parse_args(['binary', 'doctor'])
        assert args.action == 'doctor'


class TestConfigCommands:
    """Test config category commands."""

    def test_config_validate(self):
        """Test config validate command."""
        parser = CLIParser()
        args = parser.parse_args(['config', 'validate'])
        assert args.category == 'config'
        assert args.action == 'validate'

    def test_config_validate_env(self):
        """Test config validate-env command."""
        parser = CLIParser()
        args = parser.parse_args(['config', 'validate-env'])
        assert args.action == 'validate-env'


class TestAgentCommands:
    """Test agent category commands."""

    def test_agent_start(self):
        """Test agent start command."""
        parser = CLIParser()
        args = parser.parse_args(['agent', 'start'])
        assert args.category == 'agent'
        assert args.action == 'start'

    def test_agent_run(self):
        """Test agent run command."""
        parser = CLIParser()
        args = parser.parse_args(['agent', 'run', 'calibrate the model'])
        assert args.action == 'run'
        assert args.prompt == 'calibrate the model'


class TestExampleCommands:
    """Test example category commands."""

    def test_example_launch(self):
        """Test example launch command."""
        parser = CLIParser()
        args = parser.parse_args(['example', 'launch', '1a'])
        assert args.category == 'example'
        assert args.action == 'launch'
        assert args.example_id == '1a'

    def test_example_list(self):
        """Test example list command."""
        parser = CLIParser()
        args = parser.parse_args(['example', 'list'])
        assert args.action == 'list'
