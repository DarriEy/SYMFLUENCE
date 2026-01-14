"""
Tests for the ToolExecutor class.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestToolExecutorInitialization:
    """Tests for ToolExecutor initialization."""

    def test_tool_executor_can_be_imported(self):
        """Test that ToolExecutor can be imported."""
        from symfluence.agent.tool_executor import ToolExecutor
        assert ToolExecutor is not None

    def test_tool_executor_initialization(self):
        """Test ToolExecutor initializes without registry."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        assert executor is not None
        assert executor.tool_registry is None

    def test_tool_executor_with_registry(self):
        """Test ToolExecutor initializes with registry."""
        from symfluence.agent.tool_executor import ToolExecutor
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        assert executor.tool_registry is registry


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_can_be_imported(self):
        """Test that ToolResult can be imported."""
        from symfluence.agent.tool_executor import ToolResult
        assert ToolResult is not None

    def test_tool_result_creation(self):
        """Test ToolResult creation."""
        from symfluence.agent.tool_executor import ToolResult

        result = ToolResult(
            success=True,
            output="Test output",
            error=None,
            exit_code=0
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.error is None
        assert result.exit_code == 0

    def test_tool_result_to_string_success(self):
        """Test ToolResult string formatting for success."""
        from symfluence.agent.tool_executor import ToolResult

        result = ToolResult(
            success=True,
            output="Task completed",
            error=None,
            exit_code=0
        )

        formatted = result.to_string()
        assert "Success" in formatted
        assert "Task completed" in formatted

    def test_tool_result_to_string_failure(self):
        """Test ToolResult string formatting for failure."""
        from symfluence.agent.tool_executor import ToolResult

        result = ToolResult(
            success=False,
            output="",
            error="Something went wrong",
            exit_code=1
        )

        formatted = result.to_string()
        assert "Failed" in formatted
        assert "Something went wrong" in formatted


class TestToolExecution:
    """Tests for tool execution."""

    def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor.execute_tool("nonexistent_tool", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_workflow_step_requires_config(self):
        """Test workflow step requires config_path."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor._execute_workflow_step("setup_project", {})

        assert result.success is False
        assert "config_path" in result.error.lower()


class TestMetaOperations:
    """Tests for meta operation execution."""

    def test_execute_show_help(self):
        """Test show_help meta operation."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor._execute_meta_operation("show_help", {})

        assert result.success is True
        assert len(result.output) > 0

    def test_execute_list_available_tools_without_registry(self):
        """Test list_available_tools without registry."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor._execute_meta_operation("list_available_tools", {})

        assert result.success is False
        assert "registry" in result.error.lower()

    def test_execute_list_available_tools_with_registry(self):
        """Test list_available_tools with registry."""
        from symfluence.agent.tool_executor import ToolExecutor
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)
        result = executor._execute_meta_operation("list_available_tools", {})

        assert result.success is True
        assert "Available Tools" in result.output

    def test_execute_explain_workflow(self):
        """Test explain_workflow meta operation."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor._execute_meta_operation("explain_workflow", {})

        assert result.success is True
        assert "Workflow" in result.output
        assert "setup_project" in result.output or "run_model" in result.output


class TestCodeOperations:
    """Tests for code operation execution."""

    def test_execute_read_file(self, temp_dir):
        """Test read_file code operation."""
        from symfluence.agent.tool_executor import ToolExecutor

        # Create a test file
        test_file = temp_dir / 'test.py'
        test_file.write_text("print('hello')\n")

        with patch('symfluence.agent.file_operations.FileOperations') as mock_ops:
            mock_instance = MagicMock()
            mock_instance.read_file.return_value = (True, "1: print('hello')")
            mock_ops.return_value = mock_instance

            executor = ToolExecutor()
            result = executor._execute_code_operations("read_file", {
                "file_path": str(test_file)
            })

            assert result.success is True

    def test_execute_list_directory(self, temp_dir):
        """Test list_directory code operation."""
        from symfluence.agent.tool_executor import ToolExecutor

        with patch('symfluence.agent.file_operations.FileOperations') as mock_ops:
            mock_instance = MagicMock()
            mock_instance.list_directory.return_value = (True, "dir1/\nfile1.py")
            mock_ops.return_value = mock_instance

            executor = ToolExecutor()
            result = executor._execute_code_operations("list_directory", {
                "directory": str(temp_dir)
            })

            assert result.success is True


class TestSlurmOperations:
    """Tests for SLURM operation execution."""

    def test_slurm_operations_not_implemented(self):
        """Test SLURM operations return not implemented."""
        from symfluence.agent.tool_executor import ToolExecutor

        executor = ToolExecutor()
        result = executor._execute_slurm_operation("submit_slurm_job", {})

        assert result.success is False
        assert "not yet implemented" in result.error.lower()


class TestSymfluenceInstance:
    """Tests for SYMFLUENCE instance management."""

    def test_get_symfluence_instance_creates_new(self, temp_dir, sample_config_yaml):
        """Test creating new SYMFLUENCE instance."""
        from symfluence.agent.tool_executor import ToolExecutor

        # Patch SYMFLUENCE in the core module where it's imported from
        with patch('symfluence.core.SYMFLUENCE') as mock_sf:
            mock_instance = MagicMock()
            mock_sf.return_value = mock_instance

            executor = ToolExecutor()
            sf = executor._get_symfluence_instance(str(sample_config_yaml))

            assert sf is mock_instance
            mock_sf.assert_called_once()

    def test_get_symfluence_instance_uses_cache(self, temp_dir, sample_config_yaml):
        """Test SYMFLUENCE instance caching."""
        from symfluence.agent.tool_executor import ToolExecutor

        with patch('symfluence.core.SYMFLUENCE') as mock_sf:
            mock_instance = MagicMock()
            mock_sf.return_value = mock_instance

            executor = ToolExecutor()

            # First call
            sf1 = executor._get_symfluence_instance(str(sample_config_yaml))
            # Second call should use cache
            sf2 = executor._get_symfluence_instance(str(sample_config_yaml))

            assert sf1 is sf2
            # Should only be called once due to caching
            mock_sf.assert_called_once()
