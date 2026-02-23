"""Unit tests for agent command handlers."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from symfluence.cli.commands.agent_commands import AgentCommands
from symfluence.cli.exit_codes import ExitCode

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestAgentStart:
    """Test agent start command."""

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_start_success(self, mock_agent_manager_class, temp_config_dir):
        """Test successful agent start."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_agent = MagicMock()
        mock_agent.run_interactive_mode.return_value = ExitCode.SUCCESS
        mock_agent_manager_class.return_value = mock_agent

        args = Namespace(
            config=str(config_file),
            verbose=False,
            debug=False
        )

        result = AgentCommands.start(args)

        assert result == ExitCode.SUCCESS
        mock_agent.run_interactive_mode.assert_called_once()

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_start_import_error(self, mock_agent_manager_class):
        """Test agent start with import error."""
        mock_agent_manager_class.side_effect = ImportError("Cannot import agent")

        args = Namespace(
            config='/some/config.yaml',
            verbose=False,
            debug=False
        )

        result = AgentCommands.start(args)

        assert result == ExitCode.DEPENDENCY_ERROR

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_start_connection_error(self, mock_agent_manager_class):
        """Test agent start with connection error."""
        mock_agent = MagicMock()
        mock_agent.run_interactive_mode.side_effect = ConnectionError("API unavailable")
        mock_agent_manager_class.return_value = mock_agent

        args = Namespace(
            config='/some/config.yaml',
            verbose=False,
            debug=False
        )

        result = AgentCommands.start(args)

        assert result == ExitCode.NETWORK_ERROR

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_start_file_not_found(self, mock_agent_manager_class):
        """Test agent start with file not found."""
        mock_agent_manager_class.side_effect = FileNotFoundError("Config not found")

        args = Namespace(
            config='/nonexistent/config.yaml',
            verbose=False,
            debug=False
        )

        result = AgentCommands.start(args)

        assert result == ExitCode.FILE_NOT_FOUND


class TestAgentRun:
    """Test agent run command."""

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_run_success(self, mock_agent_manager_class, temp_config_dir):
        """Test successful prompt execution."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_agent = MagicMock()
        mock_agent.run_single_prompt.return_value = ExitCode.SUCCESS
        mock_agent_manager_class.return_value = mock_agent

        args = Namespace(
            config=str(config_file),
            prompt='Run the model',
            verbose=False,
            debug=False
        )

        result = AgentCommands.run(args)

        assert result == ExitCode.SUCCESS
        mock_agent.run_single_prompt.assert_called_once_with('Run the model')

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_run_timeout_error(self, mock_agent_manager_class, temp_config_dir):
        """Test prompt execution with timeout."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_agent = MagicMock()
        mock_agent.run_single_prompt.side_effect = TimeoutError("Request timed out")
        mock_agent_manager_class.return_value = mock_agent

        args = Namespace(
            config=str(config_file),
            prompt='Complex analysis',
            verbose=False,
            debug=False
        )

        result = AgentCommands.run(args)

        assert result == ExitCode.TIMEOUT_ERROR

    @patch('symfluence.agent.agent_manager.AgentManager')
    def test_run_connection_error(self, mock_agent_manager_class, temp_config_dir):
        """Test prompt execution with connection error."""
        config_file = temp_config_dir / "config_files" / "config_template.yaml"

        mock_agent = MagicMock()
        mock_agent.run_single_prompt.side_effect = ConnectionError("Lost connection")
        mock_agent_manager_class.return_value = mock_agent

        args = Namespace(
            config=str(config_file),
            prompt='Run the model',
            verbose=False,
            debug=False
        )

        result = AgentCommands.run(args)

        assert result == ExitCode.NETWORK_ERROR
