"""Tests for TUI screens — mode switching, navigation, rendering."""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        assert len(app.BINDINGS) >= 9

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

    def test_command_palette_contains_core_actions(self):
        """Command palette contains key navigation and setup actions."""
        app = SymfluenceTUI()
        command_ids = {item[0] for item in app._command_palette_items()}
        assert "mode:dashboard" in command_ids
        assert "mode:workflow" in command_ids
        assert "app:set_data_dir" in command_ids


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
        from symfluence.tui.screens.calibration import CalibrationScreen
        from symfluence.tui.screens.dashboard import DashboardScreen
        from symfluence.tui.screens.results_compare import ResultsCompareScreen
        from symfluence.tui.screens.run_browser import RunBrowserScreen
        from symfluence.tui.screens.slurm_monitor import SlurmMonitorScreen
        from symfluence.tui.screens.workflow_launcher import WorkflowLauncherScreen

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
                assert "2" in str(stat.renderable)

        asyncio.run(_test())

    def test_shows_run_count(self, mock_data_dir):
        """Dashboard shows total run count."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                stat = app.screen.query_one("#stat-runs", Static)
                assert "2" in str(stat.renderable)

        asyncio.run(_test())

    def test_shows_slurm_na_on_laptop(self, mock_data_dir):
        """SLURM stat shows N/A on non-HPC systems."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                stat = app.screen.query_one("#stat-slurm", Static)
                assert "N/A" in str(stat.renderable)

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

    def test_domain_selection_applies_run_browser_filter(self, mock_data_dir):
        """Selecting a dashboard domain opens Run Browser with that filter set."""
        from textual.widgets import Input

        from symfluence.tui.screens.run_browser import RunBrowserScreen

        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                app.screen.on_data_table_row_selected(
                    SimpleNamespace(row_key="bow_at_banff")
                )
                await pilot.pause()
                assert isinstance(app.screen, RunBrowserScreen)
                domain_filter = app.screen.query_one("#filter-domain", Input)
                assert domain_filter.value == "bow_at_banff"

        asyncio.run(_test())

    def test_onboarding_visible_without_data_dir(self, monkeypatch):
        """Dashboard shows first-run onboarding when no data directory is configured."""
        monkeypatch.delenv("SYMFLUENCE_DATA_DIR", raising=False)
        monkeypatch.delenv("SYMFLUENCE_DATA", raising=False)

        async def _test():
            app = SymfluenceTUI()
            async with app.run_test(size=(120, 40)) as pilot:
                from textual.widgets import Static
                onboarding = app.screen.query_one("#onboarding-panel", Static)
                assert onboarding.display is not False
                assert "First run setup" in str(onboarding.renderable)

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
        from textual.widgets import RichLog

        from symfluence.tui.screens.run_detail import RunDetailScreen

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

                from textual.widgets import Button, Input
                config_input = app.screen.query_one("#config-path", Input)
                assert config_input is not None

                run_btn = app.screen.query_one("#btn-run", Button)
                assert run_btn.disabled is True  # Disabled until config loaded

        asyncio.run(_test())

    def test_parses_structured_workflow_status(self):
        """Structured status payloads are converted to completed CLI step names."""
        from symfluence.tui.screens.workflow_launcher import WorkflowLauncherScreen

        status = {
            "total_steps": 3,
            "completed_steps": 1,
            "pending_steps": 2,
            "step_details": [
                {"cli_name": "setup_project", "complete": True},
                {"cli_name": "run_model", "complete": False},
            ],
        }
        completed = WorkflowLauncherScreen._completed_steps_from_status(status)
        assert completed == ["setup_project"]

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

    def test_selected_steps_mode_runs_selected_steps(self, mock_data_dir):
        """Selected steps mode dispatches only the chosen steps."""
        async def _test():
            app = SymfluenceTUI()
            app.data_dir_service = DataDirService(str(mock_data_dir))
            async with app.run_test(size=(120, 40)) as pilot:
                await app.action_switch_mode("workflow")
                await pilot.pause()

                from textual.widgets import SelectionList

                screen = app.screen
                screen._workflow_svc._sf = object()
                screen._workflow_svc.run_steps = MagicMock()
                screen._workflow_svc.run_workflow = MagicMock()
                screen._known_step_names = ["setup_project", "run_model"]

                selector = screen.query_one("#step-selector", SelectionList)
                selector.clear_options()
                selector.add_option(("Set up project structure", "setup_project"))
                selector.add_option(("Run hydrological model", "run_model"))
                selector.select("run_model")
                screen._current_run_mode = lambda: "mode-steps"
                screen._refresh_mode_controls()

                captured = {}

                def _capture_worker(work, **kwargs):
                    captured["work"] = work
                    captured["kwargs"] = kwargs

                screen.run_worker = _capture_worker
                screen._do_run()

                captured["work"]()
                screen._workflow_svc.run_steps.assert_called_once_with(["run_model"])
                screen._workflow_svc.run_workflow.assert_not_called()
                screen._cleanup_logger()

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
                assert "not available" in str(status.renderable).lower() or \
                       "0 job" in str(status.renderable).lower()

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
