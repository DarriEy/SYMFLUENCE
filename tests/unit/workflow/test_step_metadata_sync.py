"""Sync tests for workflow step metadata across frontends and orchestrator."""

from unittest.mock import MagicMock

from symfluence.cli.argument_parser import WORKFLOW_STEPS as CLI_WORKFLOW_STEPS
from symfluence.cli.commands.workflow_commands import WorkflowCommands
from symfluence.gui.components.workflow_runner import STEP_ORDER as GUI_STEP_ORDER
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator
from symfluence.tui.constants import WORKFLOW_STEPS as TUI_WORKFLOW_STEPS
from symfluence.workflow_steps import WORKFLOW_STEP_NAMES


def _build_orchestrator() -> WorkflowOrchestrator:
    managers = {
        "project": MagicMock(),
        "domain": MagicMock(),
        "data": MagicMock(),
        "model": MagicMock(),
        "analysis": MagicMock(),
        "optimization": MagicMock(),
    }
    config = {
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "run_1",
        "SYMFLUENCE_DATA_DIR": "/tmp/data",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "lumped",
    }
    logger = MagicMock()
    return WorkflowOrchestrator(managers, config, logger)


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
