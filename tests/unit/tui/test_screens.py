"""Tests for TUI screens — mode switching, navigation, rendering."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")

from symfluence.tui.app import SymfluenceTUI
from symfluence.tui.services.data_dir import DataDirService
from symfluence.tui.services.run_history import RunSummary

pytestmark = [pytest.mark.unit, pytest.mark.tui, pytest.mark.quick]


# ============================================================================
# App Construction
# ============================================================================

class TestSymfluenceTUI:
    """Tests for the root TUI application."""

    def test_app_construction(self):
        """App can be constructed without errors."""
        app = SymfluenceTUI()
        assert app.TITLE == "SYMFLUENCE"
        assert len(app.MODES) == 6
        assert len(app.BINDINGS) == 7

    def test_app_with_config_path(self):
        """Config path is stored on construction."""
        app = SymfluenceTUI(config_path="/some/config.yaml")
        assert app.config_path == "/some/config.yaml"

    def test_app_with_demo(self):
        """Demo name is stored on construction."""
        app = SymfluenceTUI(demo="bow")
        assert app.demo == "bow"

    def test_modes_are_screen_subclasses(self):
        """All MODES values are Screen subclasses."""
        from textual.screen import Screen

        for name, cls in SymfluenceTUI.MODES.items():
            assert issubclass(cls, Screen), f"{name} is not a Screen subclass"


# ============================================================================
# Screen Mode Switching
# ============================================================================

class TestScreenSwitching:
    """Tests for switching between screen modes."""

    def test_starts_on_dashboard(self, mock_data_dir):
        """App starts on DashboardScreen."""
        from symfluence.tui.screens.dashboard import DashboardScreen

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                assert isinstance(app.screen, DashboardScreen)

        asyncio.run(_test())

    def test_switch_to_all_modes(self, mock_data_dir):
        """All 6 modes are reachable via action_switch_mode."""
        from symfluence.tui.screens.dashboard import DashboardScreen
        from symfluence.tui.screens.run_browser import RunBrowserScreen
        from symfluence.tui.screens.workflow_launcher import WorkflowLauncherScreen
        from symfluence.tui.screens.calibration import CalibrationScreen
        from symfluence.tui.screens.slurm_monitor import SlurmMonitorScreen
        from symfluence.tui.screens.results_compare import ResultsCompareScreen

        expected = {
            "dashboard": DashboardScreen,
            "run_browser": RunBrowserScreen,
            "workflow": WorkflowLauncherScreen,
            "calibration": CalibrationScreen,
            "slurm": SlurmMonitorScreen,
            "compare": ResultsCompareScreen,
        }

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                for mode, screen_cls in expected.items():
                    await app.action_switch_mode(mode)
                    await pilot.pause()
                    assert isinstance(app.screen, screen_cls), (
                        f"Expected {screen_cls.__name__} for mode '{mode}', "
                        f"got {type(app.screen).__name__}"
                    )

        asyncio.run(_test())


# ============================================================================
# DashboardScreen
# ============================================================================

class TestDashboardScreen:
    """Tests for DashboardScreen content."""

    def test_shows_domain_count(self, mock_data_dir):
        """Dashboard shows correct domain count."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                stat = app.screen.query_one("#stat-domains", Static)
                assert "2" in str(stat.content)

        asyncio.run(_test())

    def test_shows_run_count(self, mock_data_dir):
        """Dashboard shows total run count."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                stat = app.screen.query_one("#stat-runs", Static)
                assert "2" in str(stat.content)

        asyncio.run(_test())

    def test_shows_slurm_na_on_laptop(self, mock_data_dir):
        """SLURM stat shows N/A on non-HPC systems."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                stat = app.screen.query_one("#stat-slurm", Static)
                assert "N/A" in str(stat.content)

        asyncio.run(_test())

    def test_domain_table_populated(self, mock_data_dir):
        """Domain table has rows for each domain."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from symfluence.tui.widgets.domain_list import DomainListWidget
                table = app.screen.query_one("#domain-table", DomainListWidget)
                assert table.row_count == 2

        asyncio.run(_test())


# ============================================================================
# RunBrowserScreen
# ============================================================================

class TestRunBrowserScreen:
    """Tests for RunBrowserScreen content and filtering."""

    def test_loads_runs_from_all_domains(self, mock_data_dir):
        """Run browser loads runs from all domains."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("run_browser")
                await pilot.pause()
                from symfluence.tui.widgets.run_summary_table import RunSummaryTable
                table = app.screen.query_one("#run-table", RunSummaryTable)
                assert table.row_count == 2

        asyncio.run(_test())


# ============================================================================
# RunDetailScreen
# ============================================================================

class TestRunDetailScreen:
    """Tests for RunDetailScreen push/pop and content."""

    def test_push_and_pop(self, mock_data_dir, sample_run_summary):
        """RunDetailScreen can be pushed and popped."""
        from symfluence.tui.screens.run_detail import RunDetailScreen

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("run_browser")
                await pilot.pause()

                await app.push_screen(RunDetailScreen(sample_run_summary))
                await pilot.pause()
                assert isinstance(app.screen, RunDetailScreen)

                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, RunDetailScreen)

        asyncio.run(_test())

    def test_displays_run_info(self, mock_data_dir, sample_run_summary):
        """RunDetailScreen shows run metadata."""
        from symfluence.tui.screens.run_detail import RunDetailScreen

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.push_screen(RunDetailScreen(sample_run_summary))
                await pilot.pause()

                # Check step progress widget exists
                from symfluence.tui.widgets.step_progress import StepProgressWidget
                steps = app.screen.query_one("#step-progress", StepProgressWidget)
                assert len(steps._step_widgets) > 0

                # Check config tree exists
                from symfluence.tui.widgets.config_tree import ConfigTreeWidget
                tree = app.screen.query_one("#config-tree", ConfigTreeWidget)
                assert tree is not None

        asyncio.run(_test())

    def test_displays_errors(self, mock_data_dir, sample_run_summary):
        """RunDetailScreen shows error details."""
        from symfluence.tui.screens.run_detail import RunDetailScreen
        from textual.widgets import RichLog

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.push_screen(RunDetailScreen(sample_run_summary))
                await pilot.pause()

                error_log = app.screen.query_one("#error-log", RichLog)
                assert error_log is not None

        asyncio.run(_test())


# ============================================================================
# WorkflowLauncherScreen
# ============================================================================

class TestWorkflowLauncherScreen:
    """Tests for WorkflowLauncherScreen layout."""

    def test_screen_renders(self, mock_data_dir):
        """Workflow launcher screen renders without errors."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("workflow")
                await pilot.pause()

                from textual.widgets import Input, Button
                config_input = app.screen.query_one("#config-path", Input)
                assert config_input is not None

                run_btn = app.screen.query_one("#btn-run", Button)
                assert run_btn.disabled is True  # Disabled until config loaded

        asyncio.run(_test())

    def test_config_path_prefilled(self, mock_data_dir):
        """Config path is prefilled when passed via CLI."""
        async def _test():
            app = SymfluenceTUI(config_path="/some/config.yaml")
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("workflow")
                await pilot.pause()

                from textual.widgets import Input
                config_input = app.screen.query_one("#config-path", Input)
                assert config_input.value == "/some/config.yaml"

        asyncio.run(_test())

    def test_slurm_options_hidden_on_laptop(self, mock_data_dir):
        """SLURM options are hidden on non-HPC systems."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("workflow")
                await pilot.pause()

                from textual.widgets import Static
                slurm_header = app.screen.query_one("#slurm-header", Static)
                assert slurm_header.display is False

        asyncio.run(_test())


# ============================================================================
# CalibrationScreen
# ============================================================================

class TestCalibrationScreen:
    """Tests for CalibrationScreen layout."""

    def test_screen_renders(self, mock_data_dir):
        """Calibration screen renders without errors."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("calibration")
                await pilot.pause()

                from textual.widgets import Select
                domain_select = app.screen.query_one("#cal-domain", Select)
                assert domain_select is not None

        asyncio.run(_test())


# ============================================================================
# SlurmMonitorScreen
# ============================================================================

class TestSlurmMonitorScreen:
    """Tests for SlurmMonitorScreen layout."""

    def test_screen_renders(self, mock_data_dir):
        """SLURM monitor screen renders without errors."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("slurm")
                await pilot.pause()

                from textual.widgets import Static
                status = app.screen.query_one("#slurm-status", Static)
                # On non-HPC, should show "not available"
                assert "not available" in str(status.content).lower() or \
                       "0 job" in str(status.content).lower()

        asyncio.run(_test())


# ============================================================================
# ResultsCompareScreen
# ============================================================================

class TestResultsCompareScreen:
    """Tests for ResultsCompareScreen layout."""

    def test_screen_renders(self, mock_data_dir):
        """Results compare screen renders without errors."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("compare")
                await pilot.pause()

                from textual.widgets import Select
                domain_select = app.screen.query_one("#cmp-domain", Select)
                assert domain_select is not None

        asyncio.run(_test())
