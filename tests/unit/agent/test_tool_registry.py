"""
Tests for the ToolRegistry class.
"""

from unittest.mock import Mock, patch

import pytest


class TestToolRegistryInitialization:
    """Tests for ToolRegistry initialization."""

    def test_tool_registry_can_be_imported(self):
        """Test that ToolRegistry can be imported."""
        from symfluence.agent.tool_registry import ToolRegistry
        assert ToolRegistry is not None

    def test_tool_registry_initialization(self):
        """Test ToolRegistry initializes with tools."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()

        assert registry is not None
        assert len(registry.tools) > 0

    def test_tool_registry_has_categories(self):
        """Test ToolRegistry organizes tools by category."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()

        assert 'Workflow Steps' in registry.tools_by_category
        assert 'Binary Management' in registry.tools_by_category
        assert 'Configuration' in registry.tools_by_category


class TestToolDefinitions:
    """Tests for tool definition retrieval."""

    def test_get_tool_definitions(self):
        """Test getting all tool definitions."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tool_definitions()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all('type' in tool for tool in tools)
        assert all('function' in tool for tool in tools)

    def test_tool_definitions_format(self):
        """Test tool definitions are in OpenAI format."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tool_definitions()

        for tool in tools:
            assert tool['type'] == 'function'
            assert 'name' in tool['function']
            assert 'description' in tool['function']
            assert 'parameters' in tool['function']

    def test_get_tools_by_category(self):
        """Test getting tools organized by category."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tools_by_category = registry.get_tools_by_category()

        assert isinstance(tools_by_category, dict)
        for category, tools in tools_by_category.items():
            assert isinstance(tools, list)
            for tool in tools:
                assert 'function' in tool


class TestWorkflowStepTools:
    """Tests for workflow step tool definitions."""

    def test_workflow_step_tools_built(self):
        """Test workflow step tools are built from WorkflowCommands."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        workflow_tools = registry.tools_by_category.get('Workflow Steps', [])

        assert len(workflow_tools) > 0

    def test_workflow_tools_require_config_path(self):
        """Test workflow tools require config_path parameter."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        workflow_tools = registry.tools_by_category.get('Workflow Steps', [])

        for tool in workflow_tools:
            params = tool['function']['parameters']
            assert 'config_path' in params.get('properties', {})
            assert 'config_path' in params.get('required', [])


class TestBinaryManagementTools:
    """Tests for binary management tool definitions."""

    def test_binary_tools_exist(self):
        """Test binary management tools exist."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        binary_tools = registry.tools_by_category.get('Binary Management', [])

        assert len(binary_tools) > 0

    def test_install_executables_tool(self):
        """Test install_executables tool definition."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        binary_tools = registry.tools_by_category.get('Binary Management', [])

        install_tool = None
        for tool in binary_tools:
            if tool['function']['name'] == 'install_executables':
                install_tool = tool
                break

        assert install_tool is not None
        assert 'tools' in install_tool['function']['parameters']['properties']


class TestConfigurationTools:
    """Tests for configuration tool definitions."""

    def test_config_tools_exist(self):
        """Test configuration tools exist."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        config_tools = registry.tools_by_category.get('Configuration', [])

        assert len(config_tools) > 0

    def test_validate_config_tool(self):
        """Test validate_config_file tool definition."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        config_tools = registry.tools_by_category.get('Configuration', [])

        validate_tool = None
        for tool in config_tools:
            if tool['function']['name'] == 'validate_config_file':
                validate_tool = tool
                break

        assert validate_tool is not None
        assert 'config_file' in validate_tool['function']['parameters'].get('required', [])


class TestCodeOperationTools:
    """Tests for code operation tool definitions."""

    def test_code_tools_exist(self):
        """Test code operation tools exist."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        code_tools = registry.tools_by_category.get('Code Operations', [])

        assert len(code_tools) > 0

    def test_read_file_tool(self):
        """Test read_file tool definition."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        code_tools = registry.tools_by_category.get('Code Operations', [])

        read_tool = None
        for tool in code_tools:
            if tool['function']['name'] == 'read_file':
                read_tool = tool
                break

        assert read_tool is not None
        assert 'file_path' in read_tool['function']['parameters'].get('required', [])

    def test_propose_code_change_tool(self):
        """Test propose_code_change tool definition."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        code_tools = registry.tools_by_category.get('Code Operations', [])

        change_tool = None
        for tool in code_tools:
            if tool['function']['name'] == 'propose_code_change':
                change_tool = tool
                break

        assert change_tool is not None
        params = change_tool['function']['parameters']
        required = params.get('required', [])
        assert 'file_path' in required
        assert 'old_code' in required
        assert 'new_code' in required


class TestMetaTools:
    """Tests for meta tool definitions."""

    def test_meta_tools_exist(self):
        """Test meta tools exist."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        meta_tools = registry.tools_by_category.get('Meta Tools', [])

        assert len(meta_tools) > 0

    def test_show_help_tool(self):
        """Test show_help tool definition."""
        from symfluence.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        meta_tools = registry.tools_by_category.get('Meta Tools', [])

        help_tool = None
        for tool in meta_tools:
            if tool['function']['name'] == 'show_help':
                help_tool = tool
                break

        assert help_tool is not None
        # show_help has no required parameters
        assert len(help_tool['function']['parameters'].get('required', [])) == 0
