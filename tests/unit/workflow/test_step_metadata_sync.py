"""Sync tests for workflow step metadata across frontends and orchestrator."""

from unittest.mock import MagicMock

import pytest

from symfluence.cli.argument_parser import WORKFLOW_STEPS as CLI_WORKFLOW_STEPS
from symfluence.cli.commands.workflow_commands import WorkflowCommands
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator
from symfluence.workflow_steps import WORKFLOW_STEP_NAMES

try:
    from symfluence.gui.components.workflow_runner import STEP_ORDER as GUI_STEP_ORDER
    from symfluence.tui.constants import WORKFLOW_STEPS as TUI_WORKFLOW_STEPS
    _HAS_GUI_DEPS = True
except ImportError:
    _HAS_GUI_DEPS = False


def _build_orchestrator() -> WorkflowOrchestrator:
    from symfluence.core.config.models import SymfluenceConfig

    managers = {
        "project": MagicMock(),
        "domain": MagicMock(),
        "data": MagicMock(),
        "model": MagicMock(),
        "analysis": MagicMock(),
        "optimization": MagicMock(),
    }
    config = SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="SUMMA",
        time_start="2010-01-01 00:00",
        time_end="2010-12-31 23:00",
        EXPERIMENT_ID="run_1",
        SYMFLUENCE_DATA_DIR="/tmp/data",
    )
    logger = MagicMock()
    return WorkflowOrchestrator(managers, config, logger)


@pytest.mark.skipif(not _HAS_GUI_DEPS, reason="panel/textual not installed")
def test_frontend_step_lists_match_shared_metadata():
    """CLI/TUI/GUI lists must match shared workflow step names."""
    assert list(CLI_WORKFLOW_STEPS) == WORKFLOW_STEP_NAMES
    assert list(WorkflowCommands.WORKFLOW_STEPS.keys()) == WORKFLOW_STEP_NAMES
    assert [name for name, _ in TUI_WORKFLOW_STEPS] == WORKFLOW_STEP_NAMES
    assert list(GUI_STEP_ORDER) == WORKFLOW_STEP_NAMES


def test_orchestrator_step_list_matches_shared_metadata():
    """Orchestrator cli_name sequence must match shared workflow step names."""
    orchestrator = _build_orchestrator()
    names = [step.cli_name for step in orchestrator.define_workflow_steps()]
    assert names == WORKFLOW_STEP_NAMES
