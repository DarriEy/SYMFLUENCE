"""
Unit tests for WorkflowOrchestrator.
"""

import pytest
from unittest.mock import MagicMock, patch
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator, WorkflowStep


class TestWorkflowOrchestrator:
    @pytest.fixture
    def mock_managers(self):
        return {
            'project': MagicMock(),
            'domain': MagicMock(),
            'data': MagicMock(),
            'model': MagicMock(),
            'analysis': MagicMock(),
            'optimization': MagicMock()
        }

    @pytest.fixture
    def config(self):
        return {
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'run_1',
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped'
        }

    @pytest.fixture
    def logger(self):
        return MagicMock()

    @pytest.fixture
    def orchestrator(self, mock_managers, config, logger):
        return WorkflowOrchestrator(mock_managers, config, logger)

    def test_run_individual_steps_success(self, orchestrator, mock_managers):
        """Test executing specific steps successfully."""
        # Setup mock steps
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            step1 = WorkflowStep("step1", "cli1", MagicMock(), lambda: True, "desc1")
            step2 = WorkflowStep("step2", "cli2", MagicMock(), lambda: True, "desc2")
            mock_define.return_value = [step1, step2]

            results = orchestrator.run_individual_steps(["cli1", "cli2"])

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is True
            step1.func.assert_called_once()
            step2.func.assert_called_once()

    def test_run_individual_steps_unrecognized(self, orchestrator):
        """Test handling of unrecognized step names."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            mock_define.return_value = []

            results = orchestrator.run_individual_steps(["unknown"])

            assert len(results) == 0
            orchestrator.logger.warning.assert_called_with("Step 'unknown' not recognized; skipping")

    def test_run_individual_steps_failure(self, orchestrator):
        """Test step failure behavior."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            def failing_func():
                raise ValueError("Step failed")

            step1 = WorkflowStep("fail", "fail_cli", failing_func, lambda: False, "fail_desc")
            mock_define.return_value = [step1]

            with pytest.raises(ValueError, match="Step failed"):
                orchestrator.run_individual_steps(["fail_cli"])

    def test_run_individual_steps_continue_on_error(self, orchestrator):
        """Test continue_on_error parameter."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            def failing_func():
                raise ValueError("Step failed")

            step1 = WorkflowStep("fail", "fail_cli", failing_func, lambda: False, "fail_desc")
            step2 = WorkflowStep("success", "ok_cli", MagicMock(), lambda: True, "ok_desc")
            mock_define.return_value = [step1, step2]

            results = orchestrator.run_individual_steps(["fail_cli", "ok_cli"], continue_on_error=True)

            assert len(results) == 2
            assert results[0]["success"] is False
            assert results[1]["success"] is True
            step2.func.assert_called_once()
