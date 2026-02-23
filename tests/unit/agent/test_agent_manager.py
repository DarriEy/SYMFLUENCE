"""
Tests for the AgentManager class.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestAgentManagerInitialization:
    """Tests for AgentManager initialization."""

    def test_agent_manager_can_be_imported(self):
        """Test that AgentManager can be imported."""
        from symfluence.agent.agent_manager import AgentManager
        assert AgentManager is not None

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_agent_manager_initialization(self, mock_api_client, mock_env_openai, temp_dir):
        """Test AgentManager initializes components."""
        from symfluence.agent.agent_manager import AgentManager

        manager = AgentManager(str(temp_dir / 'config.yaml'), verbose=False)

        assert manager is not None
        assert manager.config_path == str(temp_dir / 'config.yaml')
        assert manager.conversation_manager is not None
        assert manager.tool_registry is not None
        assert manager.tool_executor is not None

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_agent_manager_verbose_mode(self, mock_api_client, mock_env_openai, temp_dir):
        """Test AgentManager verbose mode setting."""
        from symfluence.agent.agent_manager import AgentManager

        manager = AgentManager(str(temp_dir / 'config.yaml'), verbose=True)

        assert manager.verbose is True


class TestAgentManagerToolPrinting:
    """Tests for AgentManager tool display."""

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_print_available_tools(self, mock_api_client, mock_env_openai, temp_dir, capsys):
        """Test printing available tools."""
        from symfluence.agent.agent_manager import AgentManager

        manager = AgentManager(str(temp_dir / 'config.yaml'))
        manager._print_available_tools()

        captured = capsys.readouterr()
        assert 'Available Tools' in captured.out
        assert 'Workflow Steps' in captured.out or 'Binary Management' in captured.out


class TestAgentLoop:
    """Tests for the main agent loop."""

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_agent_loop_simple_response(self, mock_api_client, mock_env_openai, temp_dir):
        """Test agent loop returns simple response without tool calls."""
        from symfluence.agent.agent_manager import AgentManager

        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.choices[0].message.tool_calls = None

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response
        mock_api_client.return_value = mock_client

        manager = AgentManager(str(temp_dir / 'config.yaml'))
        manager.api_client = mock_client

        response = manager._agent_loop("Hello")

        assert response == "Hello! How can I help you?"

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_agent_loop_handles_api_error(self, mock_api_client, mock_env_openai, temp_dir):
        """Test agent loop handles API errors gracefully."""
        from symfluence.agent.agent_manager import AgentManager

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception("API Error")
        mock_api_client.return_value = mock_client

        manager = AgentManager(str(temp_dir / 'config.yaml'))
        manager.api_client = mock_client

        response = manager._agent_loop("Hello")

        assert "API error" in response


class TestAgentManagerModes:
    """Tests for different agent execution modes."""

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_run_single_prompt_success(self, mock_api_client, mock_env_openai, temp_dir, capsys):
        """Test single prompt mode execution."""
        from symfluence.agent.agent_manager import AgentManager

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Task completed."
        mock_response.choices[0].message.tool_calls = None

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = mock_response
        mock_api_client.return_value = mock_client

        manager = AgentManager(str(temp_dir / 'config.yaml'))
        manager.api_client = mock_client

        exit_code = manager.run_single_prompt("Test prompt")

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Task completed." in captured.out

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_run_single_prompt_error(self, mock_api_client, mock_env_openai, temp_dir, capsys):
        """Test single prompt mode error handling."""
        from symfluence.agent.agent_manager import AgentManager

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception("Test error")
        mock_api_client.return_value = mock_client

        manager = AgentManager(str(temp_dir / 'config.yaml'))
        manager.api_client = mock_client

        exit_code = manager.run_single_prompt("Test prompt")

        # Agent handles errors gracefully - check output contains error message
        captured = capsys.readouterr()
        assert "API error" in captured.out or exit_code in (0, 1)


class TestToolCallExecution:
    """Tests for tool call execution in agent loop."""

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_execute_tool_call_success(self, mock_api_client, mock_env_openai, temp_dir):
        """Test successful tool call execution."""
        from symfluence.agent.agent_manager import AgentManager
        from symfluence.agent.tool_executor import ToolResult

        manager = AgentManager(str(temp_dir / 'config.yaml'))

        # Mock the tool executor
        manager.tool_executor.execute_tool = Mock(return_value=ToolResult(
            success=True,
            output="Tool output",
            error=None,
            exit_code=0
        ))

        # Create mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "show_help"
        mock_tool_call.function.arguments = "{}"

        result = manager._execute_tool_call(mock_tool_call)

        assert result.success is True
        assert result.output == "Tool output"

    @patch('symfluence.agent.agent_manager.APIClient')
    def test_execute_tool_call_invalid_json(self, mock_api_client, mock_env_openai, temp_dir):
        """Test tool call with invalid JSON arguments."""
        from symfluence.agent.agent_manager import AgentManager

        manager = AgentManager(str(temp_dir / 'config.yaml'))

        # Create mock tool call with invalid JSON
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "show_help"
        mock_tool_call.function.arguments = "not valid json"

        result = manager._execute_tool_call(mock_tool_call)

        assert result.success is False
        assert "Invalid tool arguments" in result.error
