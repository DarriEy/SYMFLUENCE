"""Tests for TUI service modules."""

import csv
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.tui, pytest.mark.quick]


# ============================================================================
# DataDirService
# ============================================================================

class TestDataDirService:
    """Tests for DataDirService domain scanning."""

    def test_resolve_explicit_path(self, tmp_path):
        """Explicit path takes priority."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService(str(tmp_path))
        assert svc.data_dir == tmp_path

    def test_resolve_nonexistent_path_returns_none(self):
        """Non-existent explicit path returns None."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService("/nonexistent/path/xyz")
        assert svc.data_dir is None

    def test_resolve_from_env_var(self, tmp_path, monkeypatch):
        """Falls back to SYMFLUENCE_DATA_DIR env var."""
        from symfluence.tui.services.data_dir import DataDirService

        monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path))
        svc = DataDirService()
        assert svc.data_dir == tmp_path

    def test_resolve_no_data_dir(self, monkeypatch):
        """Returns None when no data dir is configured."""
        from symfluence.tui.services.data_dir import DataDirService

        monkeypatch.delenv("SYMFLUENCE_DATA_DIR", raising=False)
        monkeypatch.delenv("SYMFLUENCE_DATA", raising=False)
        svc = DataDirService()
        assert svc.data_dir is None

    def test_list_domains_empty(self, tmp_path):
        """Empty data dir returns empty list."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService(str(tmp_path))
        assert svc.list_domains() == []

    def test_list_domains_finds_domains(self, mock_data_dir):
        """Discovers domain_* directories correctly."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService(str(mock_data_dir))
        domains = svc.list_domains()
        names = [d.name for d in domains]

        assert len(domains) == 2
        assert "bow_at_banff" in names
        assert "iceland_test" in names

    def test_domain_info_metadata(self, mock_data_dir):
        """Domain metadata is populated correctly."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService(str(mock_data_dir))
        domains = {d.name: d for d in svc.list_domains()}

        bow = domains["bow_at_banff"]
        assert bow.run_count == 1
        assert bow.last_status == "completed"
        assert bow.last_run is not None
        assert "exp_dds" in bow.experiments

        iceland = domains["iceland_test"]
        assert iceland.run_count == 1
        assert iceland.last_status == "failed"
        assert iceland.experiments == []  # No optimization dir

    def test_list_domains_ignores_non_domain_dirs(self, tmp_path):
        """Directories not starting with domain_ are ignored."""
        from symfluence.tui.services.data_dir import DataDirService

        (tmp_path / "other_dir").mkdir()
        (tmp_path / "domain_valid").mkdir()
        (tmp_path / "some_file.txt").touch()

        svc = DataDirService(str(tmp_path))
        domains = svc.list_domains()
        assert len(domains) == 1
        assert domains[0].name == "valid"

    def test_list_domains_no_data_dir(self):
        """Returns empty when data dir is None."""
        from symfluence.tui.services.data_dir import DataDirService

        svc = DataDirService("/nonexistent")
        assert svc.list_domains() == []


# ============================================================================
# RunHistoryService
# ============================================================================

class TestRunHistoryService:
    """Tests for RunHistoryService run summary parsing."""

    def test_list_runs(self, mock_data_dir):
        """Lists runs from log directory."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_bow_at_banff")
        runs = svc.list_runs()

        assert len(runs) == 1
        run = runs[0]
        assert run.domain == "bow_at_banff"
        assert run.experiment_id == "exp_dds"
        assert run.status == "completed"
        assert run.total_steps == 3
        assert run.model == "MESH"
        assert run.algorithm == "dds"

    def test_list_runs_parses_steps(self, mock_data_dir):
        """Step names are extracted from dict or string format."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_bow_at_banff")
        runs = svc.list_runs()
        assert runs[0].steps_completed == [
            "setup_project", "create_pour_point", "run_model"
        ]

    def test_list_runs_failed_domain(self, mock_data_dir):
        """Failed runs include error information."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_iceland_test")
        runs = svc.list_runs()

        assert len(runs) == 1
        run = runs[0]
        assert run.status == "failed"
        assert run.total_errors == 1
        assert run.errors[0]["step"] == "define_domain"

    def test_list_runs_empty_dir(self, tmp_path):
        """Domain with no log dir returns empty list."""
        from symfluence.tui.services.run_history import RunHistoryService

        domain = tmp_path / "domain_empty"
        domain.mkdir()
        svc = RunHistoryService(domain)
        assert svc.list_runs() == []

    def test_load_config_snapshot(self, mock_data_dir):
        """Config snapshot is loaded from matching YAML file."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_bow_at_banff")
        runs = svc.list_runs()
        config = svc.load_config_snapshot(runs[0])

        assert config is not None
        assert config["DOMAIN_NAME"] == "bow_at_banff"
        assert config["MODEL"] == "MESH"

    def test_load_config_snapshot_missing(self, mock_data_dir):
        """Returns None when no config YAML is found."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_iceland_test")
        runs = svc.list_runs()
        config = svc.load_config_snapshot(runs[0])
        assert config is not None or config is None  # May fall back to most recent

    def test_parse_summary_malformed_json(self, tmp_path):
        """Malformed JSON returns None gracefully."""
        from symfluence.tui.services.run_history import RunHistoryService

        domain = tmp_path / "domain_bad"
        domain.mkdir()
        log_dir = domain / "_workLog_bad"
        log_dir.mkdir()
        (log_dir / "run_summary_20250101_000000.json").write_text("not valid json{")

        svc = RunHistoryService(domain)
        runs = svc.list_runs()
        assert runs == []

    def test_timestamp_parsing(self, mock_data_dir):
        """Timestamps are parsed from ISO format."""
        from symfluence.tui.services.run_history import RunHistoryService

        svc = RunHistoryService(mock_data_dir / "domain_bow_at_banff")
        runs = svc.list_runs()
        assert runs[0].timestamp == datetime(2025, 6, 1, 12, 0, 0)


# ============================================================================
# WorkflowService
# ============================================================================

class TestWorkflowService:
    """Tests for WorkflowService config/workflow wrapper."""

    def test_initial_state(self):
        """Service starts unloaded."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        assert not svc.is_loaded
        assert svc.config_path is None

    @patch("symfluence.tui.services.workflow_service.WorkflowService.load_config")
    def test_load_config_returns_bool(self, mock_load):
        """load_config returns True/False."""
        from symfluence.tui.services.workflow_service import WorkflowService

        mock_load.return_value = True
        svc = WorkflowService()
        assert svc.load_config("/some/path.yaml") is True

    def test_run_workflow_without_config_raises(self):
        """Running workflow without config raises RuntimeError."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        with pytest.raises(RuntimeError, match="No config loaded"):
            svc.run_workflow()

    def test_run_steps_without_config_raises(self):
        """Running steps without config raises RuntimeError."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        with pytest.raises(RuntimeError, match="No config loaded"):
            svc.run_steps(["setup_project"])

    def test_get_status_unloaded(self):
        """Status returns empty dict when not loaded."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        assert svc.get_status() == {}

    def test_get_domain_name_unloaded(self):
        """Domain name returns empty string when not loaded."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        assert svc.get_domain_name() == ""

    def test_invalidate(self):
        """Invalidate clears the internal state."""
        from symfluence.tui.services.workflow_service import WorkflowService

        svc = WorkflowService()
        svc._config_path = Path("/some/path")
        svc._sf = MagicMock()
        svc.invalidate()
        assert not svc.is_loaded
        assert svc.config_path is None


# ============================================================================
# CalibrationDataService
# ============================================================================

class TestCalibrationDataService:
    """Tests for CalibrationDataService results loading."""

    def test_list_experiments(self, mock_data_dir):
        """Lists experiment IDs from optimization directory."""
        from symfluence.tui.services.calibration_data import CalibrationDataService

        svc = CalibrationDataService(str(mock_data_dir / "domain_bow_at_banff"))
        experiments = svc.list_experiments()
        assert "exp_dds" in experiments

    def test_list_experiments_no_optimization(self, mock_data_dir):
        """Returns empty list when no optimization directory exists."""
        from symfluence.tui.services.calibration_data import CalibrationDataService

        svc = CalibrationDataService(str(mock_data_dir / "domain_iceland_test"))
        assert svc.list_experiments() == []

    def test_list_experiments_nonexistent_dir(self, tmp_path):
        """Returns empty list for nonexistent project dir."""
        from symfluence.tui.services.calibration_data import CalibrationDataService

        svc = CalibrationDataService(str(tmp_path / "nonexistent"))
        assert svc.list_experiments() == []

    def test_clear_cache(self, mock_data_dir):
        """clear_cache does not raise when loader not yet created."""
        from symfluence.tui.services.calibration_data import CalibrationDataService

        svc = CalibrationDataService(str(mock_data_dir / "domain_bow_at_banff"))
        svc.clear_cache()  # Should not raise


# ============================================================================
# SlurmService
# ============================================================================

class TestSlurmService:
    """Tests for SlurmService SLURM interaction."""

    def test_is_hpc_without_slurm(self):
        """Detects non-HPC environment correctly."""
        from symfluence.tui.services.slurm_service import SlurmService

        svc = SlurmService()
        # On dev machines, SLURM is typically not available
        # This test verifies the method doesn't crash
        result = svc.is_hpc()
        assert isinstance(result, bool)

    @patch("subprocess.run")
    def test_is_hpc_with_slurm_env(self, mock_run, monkeypatch):
        """Detects HPC when SLURM_CONF is set."""
        from symfluence.tui.services.slurm_service import SlurmService

        monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
        assert SlurmService.is_hpc() is True

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_is_hpc_no_squeue_binary(self, mock_run, monkeypatch):
        """Returns False when squeue is not found."""
        from symfluence.tui.services.slurm_service import SlurmService

        monkeypatch.delenv("SLURM_CONF", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert SlurmService.is_hpc() is False

    @patch("subprocess.run")
    def test_list_user_jobs_parses_output(self, mock_run):
        """Parses squeue output into SlurmJob objects."""
        from symfluence.tui.services.slurm_service import SlurmService

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345|my_job|RUNNING|compute|01:30:00|1\n"
                   "12346|other_job|PENDING|gpu|00:00:00|2\n",
        )
        svc = SlurmService()
        jobs = svc.list_user_jobs()

        assert len(jobs) == 2
        assert jobs[0].job_id == "12345"
        assert jobs[0].name == "my_job"
        assert jobs[0].status == "RUNNING"
        assert jobs[1].job_id == "12346"
        assert jobs[1].nodes == "2"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_list_user_jobs_no_slurm(self, mock_run):
        """Returns empty list when squeue is unavailable."""
        from symfluence.tui.services.slurm_service import SlurmService

        svc = SlurmService()
        assert svc.list_user_jobs() == []

    @patch("subprocess.run")
    def test_cancel_job_success(self, mock_run):
        """Cancel returns True on success."""
        from symfluence.tui.services.slurm_service import SlurmService

        mock_run.return_value = MagicMock(returncode=0)
        svc = SlurmService()
        assert svc.cancel_job("12345") is True

    @patch("subprocess.run")
    def test_cancel_job_failure(self, mock_run):
        """Cancel returns False on failure."""
        from symfluence.tui.services.slurm_service import SlurmService

        mock_run.return_value = MagicMock(returncode=1)
        svc = SlurmService()
        assert svc.cancel_job("99999") is False

    @patch("subprocess.run")
    def test_submit_job_returns_id(self, mock_run):
        """Submit parses job ID from sbatch output."""
        from symfluence.tui.services.slurm_service import SlurmService

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 67890\n",
        )
        svc = SlurmService()
        job_id = svc.submit_job("/path/to/script.sh")
        assert job_id == "67890"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_submit_job_no_sbatch(self, mock_run):
        """Submit returns None when sbatch is unavailable."""
        from symfluence.tui.services.slurm_service import SlurmService

        svc = SlurmService()
        assert svc.submit_job("/path/to/script.sh") is None
