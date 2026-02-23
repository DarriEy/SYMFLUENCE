"""TUI test fixtures and utilities."""

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from symfluence.tui.services.run_history import RunSummary

# ============================================================================
# Mock data directory
# ============================================================================

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary SYMFLUENCE_DATA_DIR with mock domain data.

    Structure:
        tmp_path/
            domain_bow_at_banff/
                _workLog_bow_at_banff/
                    run_summary_20250601_120000.json
                    config_bow_at_banff_20250601_120000.yaml
                optimization/
                    exp_dds_parallel_iteration_results.csv
                    exp_dds_best_params.json
            domain_iceland_test/
                _workLog_iceland_test/
                    run_summary_20250715_090000.json
    """
    # --- Domain 1: bow_at_banff ---
    d1 = tmp_path / "domain_bow_at_banff"
    d1.mkdir()

    log1 = d1 / "_workLog_bow_at_banff"
    log1.mkdir()

    summary1 = {
        "timestamp": "2025-06-01T12:00:00",
        "domain": "bow_at_banff",
        "experiment_id": "exp_dds",
        "execution_time_seconds": 1200.0,
        "steps_completed": [
            {"cli": "setup_project", "fn": "setup_project"},
            {"cli": "create_pour_point", "fn": "create_pour_point"},
            {"cli": "run_model", "fn": "run_model"},
        ],
        "total_steps_completed": 3,
        "errors": [],
        "total_errors": 0,
        "warnings": ["Low snow fraction"],
        "total_warnings": 1,
        "debug_mode": False,
        "status": "completed",
        "configuration": {
            "hydrological_model": "MESH",
            "domain_definition_method": "lumped",
            "optimization_algorithm": "dds",
            "force_run_all": False,
        },
    }
    (log1 / "run_summary_20250601_120000.json").write_text(json.dumps(summary1))

    config_yaml = "DOMAIN_NAME: bow_at_banff\nEXPERIMENT_ID: exp_dds\nMODEL: MESH\n"
    (log1 / "config_bow_at_banff_20250601_120000.yaml").write_text(config_yaml)

    opt1 = d1 / "optimization"
    opt1.mkdir()

    with open(opt1 / "exp_dds_parallel_iteration_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "kge", "KSAT", "SDEP"])
        for i in range(50):
            w.writerow([i, 0.2 + i * 0.01, 100 - i * 0.5, 1.5 + i * 0.01])

    best_params = {"KSAT": 75.0, "SDEP": 2.0, "best_score": 0.69}
    (opt1 / "exp_dds_best_params.json").write_text(json.dumps(best_params))

    # --- Domain 2: iceland_test ---
    d2 = tmp_path / "domain_iceland_test"
    d2.mkdir()

    log2 = d2 / "_workLog_iceland_test"
    log2.mkdir()

    summary2 = {
        "timestamp": "2025-07-15T09:00:00",
        "domain": "iceland_test",
        "experiment_id": "exp_fuse",
        "execution_time_seconds": 45.0,
        "steps_completed": [{"cli": "setup_project"}],
        "total_steps_completed": 1,
        "errors": [{"step": "define_domain", "error": "Shapefile not found"}],
        "total_errors": 1,
        "warnings": [],
        "total_warnings": 0,
        "debug_mode": False,
        "status": "failed",
        "configuration": {
            "hydrological_model": "FUSE",
            "domain_definition_method": "lumped",
            "optimization_algorithm": "multi_gauge",
            "force_run_all": False,
        },
    }
    (log2 / "run_summary_20250715_090000.json").write_text(json.dumps(summary2))

    return tmp_path


@pytest.fixture
def sample_run_summary(tmp_path):
    """A single RunSummary dataclass for widget/screen tests."""
    log_dir = tmp_path / "domain_test" / "_workLog_test"
    log_dir.mkdir(parents=True)
    summary_file = log_dir / "run_summary_20250601_120000.json"
    summary_file.write_text("{}")

    return RunSummary(
        file_path=summary_file,
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
        domain="test_domain",
        experiment_id="exp_001",
        status="completed",
        execution_time=342.5,
        steps_completed=["setup_project", "create_pour_point", "run_model"],
        total_steps=3,
        errors=[{"step": "calibrate_model", "error": "Convergence failed"}],
        total_errors=1,
        warnings=["Minor CRS mismatch"],
        total_warnings=1,
        model="SUMMA",
        algorithm="dds",
    )


@pytest.fixture
def mock_tui_app(mock_data_dir):
    """Create a SymfluenceTUI app with injected mock data dir."""
    from symfluence.tui.app import SymfluenceTUI
    from symfluence.tui.services.data_dir import DataDirService

    app = SymfluenceTUI()
    app.data_dir_service = DataDirService(str(mock_data_dir))
    return app
