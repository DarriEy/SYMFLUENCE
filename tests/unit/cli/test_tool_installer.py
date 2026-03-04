"""Unit tests for ToolInstaller behavior and result reporting."""

from unittest.mock import MagicMock


def test_install_fails_fast_when_required_tool_missing(mock_external_tools, tmp_path):
    """A tool with unmet `requires` should not proceed to build."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)
    installer._verify_installation = MagicMock(return_value=True)

    result = installer.install(specific_tools=["summa"], force=True)

    assert all(call.args[0] != "summa" for call in installer._run_build_commands.call_args_list)
    assert "summa" in result["failed"]
    assert "summa" not in result["successful"]


def test_install_marks_verification_failure_as_failed(mock_external_tools, tmp_path):
    """Verification failure should remove a tool from successful installs."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)
    installer._verify_installation = MagicMock(return_value=False)

    result = installer.install(specific_tools=["taudem"], force=True)

    assert "taudem" in result["failed"]
    assert "taudem" not in result["successful"]
    assert any("taudem: installation verification failed" in e for e in result["errors"])
