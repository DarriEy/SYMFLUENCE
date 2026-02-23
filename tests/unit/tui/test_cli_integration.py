"""Tests for TUI CLI integration — argument parser and command handler."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("textual", reason="textual not installed")

pytestmark = [pytest.mark.unit, pytest.mark.tui, pytest.mark.cli, pytest.mark.quick]


# ============================================================================
# Argument Parser Registration
# ============================================================================

class TestTUIArgumentParser:
    """Tests for TUI subcommand registration in the CLI."""

    def test_tui_subcommand_registered(self):
        """The 'tui' subcommand is available."""
        from symfluence.cli.argument_parser import CLIParser

        parser = CLIParser()
        # Parse 'tui launch' — should not raise
        args = parser.parse_args(["tui", "launch"])
        assert hasattr(args, "func")

    def test_tui_launch_has_func(self):
        """'tui launch' sets the correct handler function."""
        from symfluence.cli.argument_parser import CLIParser
        from symfluence.cli.commands.tui_commands import TUICommands

        parser = CLIParser()
        args = parser.parse_args(["tui", "launch"])
        assert args.func == TUICommands.launch

    def test_tui_launch_demo_argument(self):
        """'--demo' argument is parsed correctly."""
        from symfluence.cli.argument_parser import CLIParser

        parser = CLIParser()
        args = parser.parse_args(["tui", "launch", "--demo", "bow"])
        assert args.demo == "bow"

    def test_tui_launch_config_argument(self):
        """'--config' argument is inherited from common parser."""
        from symfluence.cli.argument_parser import CLIParser

        parser = CLIParser()
        args = parser.parse_args(["tui", "launch", "--config", "/path/to/config.yaml"])
        assert args.config == "/path/to/config.yaml"

    def test_tui_without_action_fails(self):
        """'tui' without an action shows error."""
        from symfluence.cli.argument_parser import CLIParser

        parser = CLIParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["tui"])


# ============================================================================
# TUICommands Handler
# ============================================================================

class TestTUICommands:
    """Tests for TUICommands.launch() handler."""

    @patch("symfluence.cli.commands.tui_commands.BaseCommand._console")
    @patch("symfluence.tui.launch_tui")
    def test_launch_calls_launch_tui(self, mock_launch, mock_console):
        """launch() delegates to symfluence.tui.launch_tui."""
        from symfluence.cli.commands.tui_commands import TUICommands
        from symfluence.cli.exit_codes import ExitCode

        args = Namespace(config="/some/config.yaml", demo="bow")
        result = TUICommands.launch(args)

        assert result == ExitCode.SUCCESS
        mock_launch.assert_called_once_with(
            config_path="/some/config.yaml",
            demo="bow",
        )

    @patch("symfluence.cli.commands.tui_commands.BaseCommand._console")
    @patch("symfluence.tui.launch_tui")
    def test_launch_with_no_args(self, mock_launch, mock_console):
        """launch() works with no optional args."""
        from symfluence.cli.commands.tui_commands import TUICommands
        from symfluence.cli.exit_codes import ExitCode

        args = Namespace()
        result = TUICommands.launch(args)

        assert result == ExitCode.SUCCESS
        mock_launch.assert_called_once_with(config_path=None, demo=None)

    @patch("symfluence.cli.commands.tui_commands.BaseCommand._console")
    def test_launch_import_error(self, mock_console):
        """launch() returns DEPENDENCY_ERROR when textual not installed."""
        from symfluence.cli.commands.tui_commands import TUICommands
        from symfluence.cli.exit_codes import ExitCode

        with patch.dict("sys.modules", {"symfluence.tui": None}):
            args = Namespace()
            result = TUICommands.launch(args)

        assert result == ExitCode.DEPENDENCY_ERROR
        mock_console.error.assert_called()


# ============================================================================
# Package Imports
# ============================================================================

class TestTUIPackageImports:
    """Tests that TUI package imports work correctly."""

    def test_import_launch_tui(self):
        """launch_tui is importable from symfluence.tui."""
        from symfluence.tui import launch_tui
        assert callable(launch_tui)

    def test_import_all_services(self):
        """All service classes are importable."""
        from symfluence.tui.services import (
            CalibrationDataService,
            DataDirService,
            RunHistoryService,
            SlurmService,
            WorkflowService,
        )
        assert all([
            DataDirService, RunHistoryService, WorkflowService,
            CalibrationDataService, SlurmService,
        ])

    def test_import_all_widgets(self):
        """All widget classes are importable."""
        from symfluence.tui.widgets import (
            ConfigTreeWidget,
            DomainListWidget,
            LogPanel,
            MetricsTable,
            RunSummaryTable,
            SlurmJobTable,
            SparklineWidget,
            StepProgressWidget,
        )
        assert all([
            DomainListWidget, RunSummaryTable, StepProgressWidget,
            MetricsTable, LogPanel, SparklineWidget,
            ConfigTreeWidget, SlurmJobTable,
        ])

    def test_import_all_screens(self):
        """All screen classes are importable."""
        from symfluence.tui.screens import (
            CalibrationScreen,
            CommandPaletteScreen,
            DashboardScreen,
            HelpScreen,
            PathPromptScreen,
            ResultsCompareScreen,
            RunBrowserScreen,
            RunDetailScreen,
            SlurmMonitorScreen,
            WorkflowLauncherScreen,
        )
        assert all([
            DashboardScreen, RunBrowserScreen, RunDetailScreen,
            WorkflowLauncherScreen, CalibrationScreen,
            SlurmMonitorScreen, ResultsCompareScreen,
            HelpScreen, CommandPaletteScreen, PathPromptScreen,
        ])

    def test_import_app(self):
        """SymfluenceTUI is importable from symfluence.tui.app."""
        from symfluence.tui.app import SymfluenceTUI
        assert SymfluenceTUI is not None

    def test_import_constants(self):
        """Constants module is importable."""
        from symfluence.tui.constants import (
            METRIC_THRESHOLDS,
            SPARKLINE_CHARS,
            STATUS_COLORS,
            STATUS_ICONS,
            WORKFLOW_STEPS,
        )
        assert len(WORKFLOW_STEPS) > 0
        assert len(STATUS_ICONS) > 0
        assert len(SPARKLINE_CHARS) > 0
