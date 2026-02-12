"""Tests for TUI widget modules."""

import asyncio
from datetime import datetime

import pytest

pytest.importorskip("textual", reason="textual not installed")

pytestmark = [pytest.mark.unit, pytest.mark.tui, pytest.mark.quick]


# ============================================================================
# SparklineWidget (non-async, pure logic)
# ============================================================================

class TestSparklineWidget:
    """Tests for SparklineWidget rendering logic."""

    def test_empty_values(self):
        """Empty list renders empty string."""
        from symfluence.tui.widgets.sparkline import SparklineWidget

        w = SparklineWidget()
        w.set_values([])
        assert w.renderable == ""

    def test_single_value(self):
        """Single value renders a single block character."""
        from symfluence.tui.widgets.sparkline import SparklineWidget

        w = SparklineWidget()
        w.set_values([5.0])
        text = str(w.renderable)
        assert len(text) == 1

    def test_increasing_values(self):
        """Increasing values produce non-decreasing block heights."""
        from symfluence.tui.widgets.sparkline import SparklineWidget, SPARKLINE_CHARS

        w = SparklineWidget()
        values = list(range(10))
        w.set_values(values)
        text = str(w.renderable)
        assert len(text) == 10

        # First char should be lowest block, last should be highest
        assert text[0] == SPARKLINE_CHARS[0]
        assert text[-1] == SPARKLINE_CHARS[-1]

    def test_constant_values(self):
        """Constant values produce uniform characters."""
        from symfluence.tui.widgets.sparkline import SparklineWidget

        w = SparklineWidget()
        w.set_values([3.0] * 20)
        text = str(w.renderable)
        assert len(set(text)) == 1  # All same character

    def test_downsampling(self):
        """Values longer than width are downsampled."""
        from symfluence.tui.widgets.sparkline import SparklineWidget

        w = SparklineWidget()
        w.set_values(list(range(200)), width=50)
        text = str(w.renderable)
        assert len(text) == 50


# ============================================================================
# ConfigTreeWidget (non-async, data loading)
# ============================================================================

class TestConfigTreeWidget:
    """Tests for ConfigTreeWidget tree building."""

    def test_load_flat_config(self):
        """Flat config creates leaf nodes."""
        from symfluence.tui.widgets.config_tree import ConfigTreeWidget

        w = ConfigTreeWidget("Test")
        w.load_config({"key1": "value1", "key2": 42})
        # Root should have children
        assert len(w.root.children) == 2

    def test_load_nested_config(self):
        """Nested config creates expandable tree nodes."""
        from symfluence.tui.widgets.config_tree import ConfigTreeWidget

        w = ConfigTreeWidget("Test")
        w.load_config({
            "section": {"nested_key": "nested_value"},
            "flat_key": "value",
        })
        # One branch node + one leaf
        assert len(w.root.children) == 2

    def test_load_list_config(self):
        """Lists create indexed child nodes."""
        from symfluence.tui.widgets.config_tree import ConfigTreeWidget

        w = ConfigTreeWidget("Test")
        w.load_config({"items": ["a", "b", "c"]})
        # Root has one child (items), which has 3 leaves
        assert len(w.root.children) == 1

    def test_load_empty_config(self):
        """Empty config clears existing nodes."""
        from symfluence.tui.widgets.config_tree import ConfigTreeWidget

        w = ConfigTreeWidget("Test")
        w.load_config({"key": "value"})
        w.load_config({})
        assert len(w.root.children) == 0


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    """Tests for TUI constants consistency."""

    def test_workflow_steps_are_tuples(self):
        """Each step is a (cli_name, description) tuple."""
        from symfluence.tui.constants import WORKFLOW_STEPS

        for step in WORKFLOW_STEPS:
            assert isinstance(step, tuple)
            assert len(step) == 2
            assert isinstance(step[0], str)
            assert isinstance(step[1], str)

    def test_status_icons_cover_all_statuses(self):
        """Every status has an icon."""
        from symfluence.tui.constants import (
            STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
            STATUS_PARTIAL, STATUS_PENDING, STATUS_ICONS,
        )
        for status in [STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
                       STATUS_PARTIAL, STATUS_PENDING]:
            assert status in STATUS_ICONS

    def test_status_colors_cover_all_statuses(self):
        """Every status has a color."""
        from symfluence.tui.constants import (
            STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
            STATUS_PARTIAL, STATUS_PENDING, STATUS_COLORS,
        )
        for status in [STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
                       STATUS_PARTIAL, STATUS_PENDING]:
            assert status in STATUS_COLORS


# ============================================================================
# Async widget tests (require Textual app context)
# ============================================================================

class TestDomainListWidget:
    """Tests for DomainListWidget within Textual app context."""

    def test_load_domains(self, mock_data_dir):
        """DomainListWidget populates from DomainInfo list."""
        from symfluence.tui.services.data_dir import DataDirService
        from symfluence.tui.widgets.domain_list import DomainListWidget

        svc = DataDirService(str(mock_data_dir))
        domains = svc.list_domains()

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield DomainListWidget(id="table")

            app = TestApp()
            async with app.run_test(size=(120, 30)) as pilot:
                table = app.query_one("#table", DomainListWidget)
                table.load_domains(domains)
                assert table.row_count == 2

        asyncio.run(_test())


class TestRunSummaryTable:
    """Tests for RunSummaryTable filtering."""

    def test_load_and_filter(self, mock_data_dir):
        """Table loads runs and applies domain/status filters."""
        from symfluence.tui.services.data_dir import DataDirService
        from symfluence.tui.services.run_history import RunHistoryService
        from symfluence.tui.widgets.run_summary_table import RunSummaryTable

        svc = DataDirService(str(mock_data_dir))
        all_runs = []
        for d in svc.list_domains():
            rhs = RunHistoryService(d.path)
            all_runs.extend(rhs.list_runs())

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield RunSummaryTable(id="table")

            app = TestApp()
            async with app.run_test(size=(120, 30)) as pilot:
                table = app.query_one("#table", RunSummaryTable)
                table.load_runs(all_runs)
                assert table.row_count == 2

                # Filter by domain
                table.filter_runs(domain_filter="bow")
                assert table.row_count == 1

                # Filter by status
                table.load_runs(all_runs)
                table.filter_runs(status_filter="failed")
                assert table.row_count == 1

                # Combined filter
                table.load_runs(all_runs)
                table.filter_runs(domain_filter="bow", status_filter="failed")
                assert table.row_count == 0

        asyncio.run(_test())


class TestStepProgressWidget:
    """Tests for StepProgressWidget status updates."""

    def test_update_from_completed(self):
        """Step widget renders correctly with completed steps."""
        from symfluence.tui.widgets.step_progress import StepProgressWidget

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield StepProgressWidget(id="steps")

            app = TestApp()
            async with app.run_test(size=(120, 50)) as pilot:
                steps = app.query_one("#steps", StepProgressWidget)
                steps.update_from_completed(
                    ["setup_project", "create_pour_point"],
                    running="acquire_attributes",
                )
                # Verify the widget has children (step labels)
                assert len(steps._step_widgets) > 0

        asyncio.run(_test())


class TestMetricsTable:
    """Tests for MetricsTable display."""

    def test_load_single_metrics(self):
        """Single experiment metrics display."""
        from symfluence.tui.widgets.metrics_table import MetricsTable

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield MetricsTable(id="metrics")

            app = TestApp()
            async with app.run_test(size=(80, 30)) as pilot:
                table = app.query_one("#metrics", MetricsTable)
                table.load_single({"KGE": 0.75, "NSE": 0.68, "RMSE": 12.3})
                assert table.row_count == 3

        asyncio.run(_test())

    def test_load_comparison_metrics(self):
        """Side-by-side comparison display."""
        from symfluence.tui.widgets.metrics_table import MetricsTable

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield MetricsTable(id="metrics")

            app = TestApp()
            async with app.run_test(size=(120, 30)) as pilot:
                table = app.query_one("#metrics", MetricsTable)
                table.load_comparison(
                    ["exp_a", "exp_b"],
                    {
                        "exp_a": {"KGE": 0.75, "NSE": 0.68},
                        "exp_b": {"KGE": 0.80, "NSE": 0.55},
                    },
                )
                assert table.row_count == 2  # KGE + NSE

        asyncio.run(_test())


class TestSlurmJobTable:
    """Tests for SlurmJobTable display."""

    def test_load_jobs(self):
        """Table loads SLURM job data."""
        from symfluence.tui.services.slurm_service import SlurmJob
        from symfluence.tui.widgets.slurm_table import SlurmJobTable

        jobs = [
            SlurmJob("123", "test_job", "RUNNING", "compute", "01:00:00", "1"),
            SlurmJob("456", "other", "PENDING", "gpu", "00:00:00", "2"),
        ]

        async def _test():
            from textual.app import App, ComposeResult
            from textual.widgets import Header

            class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield Header()
                    yield SlurmJobTable(id="jobs")

            app = TestApp()
            async with app.run_test(size=(120, 30)) as pilot:
                table = app.query_one("#jobs", SlurmJobTable)
                table.load_jobs(jobs)
                assert table.row_count == 2

        asyncio.run(_test())
