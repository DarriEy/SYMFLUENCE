"""
Unit tests for WorkflowOrchestrator.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
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
        return SymfluenceConfig.from_minimal(
            domain_name='test_domain',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-12-31 23:00',
            SYMFLUENCE_DATA_DIR='/tmp/data',
        )

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

            with pytest.raises(ValueError, match="Step 'unknown' not recognized"):
                orchestrator.run_individual_steps(["unknown"])

    def test_run_individual_steps_unrecognized_continue_on_error(self, orchestrator):
        """Test unrecognized step recorded as failure when continue_on_error=True."""
        with patch.object(orchestrator, 'define_workflow_steps') as mock_define:
            mock_define.return_value = []

            results = orchestrator.run_individual_steps(["unknown"], continue_on_error=True)

            assert len(results) == 1
            assert results[0]["success"] is False
            assert "not recognized" in results[0]["error"]

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

    def _build_orchestrator_for_observation_checks(
        self,
        tmp_path: Path,
        *,
        evaluation_data: str
    ) -> WorkflowOrchestrator:
        """Create an orchestrator rooted at tmp_path for observation output checks."""
        managers = {
            'project': MagicMock(),
            'domain': MagicMock(),
            'data': MagicMock(),
            'model': MagicMock(),
            'analysis': MagicMock(),
            'optimization': MagicMock(),
        }
        config = SymfluenceConfig.from_minimal(
            domain_name='obs_domain',
            model='SUMMA',
            time_start='2010-01-01 00:00',
            time_end='2010-01-02 00:00',
            SYMFLUENCE_DATA_DIR=tmp_path,
            EVALUATION_DATA=evaluation_data,
        )
        return WorkflowOrchestrator(managers, config, MagicMock())

    @staticmethod
    def _touch_streamflow_output(project_dir: Path, domain_name: str) -> None:
        path = project_dir / "observations" / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("datetime,streamflow\n2010-01-01,1.0\n", encoding="utf-8")

    @staticmethod
    def _touch_snow_output(project_dir: Path, domain_name: str) -> None:
        path = project_dir / "observations" / "snow" / "preprocessed" / f"{domain_name}_snow_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("datetime,snow\n2010-01-01,1.0\n", encoding="utf-8")

    def test_check_observed_data_requires_all_requested_families(self, tmp_path):
        """
        When multiple observation families are requested, all must exist.
        """
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="streamflow,swe",
        )

        # Only streamflow exists -> should still be incomplete.
        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        assert orchestrator._check_observed_data_exists() is False

    def test_check_observed_data_true_when_all_requested_families_exist(self, tmp_path):
        """Requested families are complete only when each required output exists."""
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="streamflow,swe",
        )

        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        self._touch_snow_output(orchestrator.project_dir, orchestrator.domain_name)

        assert orchestrator._check_observed_data_exists() is True

    def test_check_observed_data_default_behavior_streamflow_only(self, tmp_path):
        """
        With no explicit observation request, fallback still checks streamflow.
        """
        orchestrator = self._build_orchestrator_for_observation_checks(
            tmp_path,
            evaluation_data="",
        )
        assert orchestrator._check_observed_data_exists() is False

        self._touch_streamflow_output(orchestrator.project_dir, orchestrator.domain_name)
        assert orchestrator._check_observed_data_exists() is True
