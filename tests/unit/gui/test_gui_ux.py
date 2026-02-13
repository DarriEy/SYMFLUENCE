from __future__ import annotations

import json

import pytest

pn = pytest.importorskip("panel")

from symfluence.gui.app import SymfluenceApp
from symfluence.gui.utils.telemetry import UXTelemetry


def test_app_collect_stage_artifacts_counts(tmp_path):
    app = SymfluenceApp()
    app.state.project_dir = str(tmp_path)

    (tmp_path / "shapefiles").mkdir()
    (tmp_path / "shapefiles" / "basin.shp").write_text("shape")
    (tmp_path / "forcing").mkdir()
    (tmp_path / "forcing" / "forcing.nc").write_text("forcing")
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "model.txt").write_text("settings")
    (tmp_path / "simulations").mkdir()
    (tmp_path / "simulations" / "run.out").write_text("sim")
    (tmp_path / "reporting").mkdir()
    (tmp_path / "reporting" / "plot.png").write_text("image")

    counts = app._collect_stage_artifacts()

    assert counts["Domain"] == 1
    assert counts["Forcing"] == 1
    assert counts["Model"] == 1
    assert counts["Run"] == 1
    assert counts["Analyze"] == 1


def test_recommended_step_advances_after_completed_steps():
    app = SymfluenceApp()
    app.state.workflow_status = {
        "step_details": [
            {"cli_name": "setup_project", "complete": True},
            {"cli_name": "create_pour_point", "complete": False},
        ]
    }

    assert app._recommended_step_for_stage("Domain") == "create_pour_point"


def test_telemetry_is_opt_in_and_writes_jsonl(tmp_path):
    output_path = tmp_path / "gui_telemetry.jsonl"
    telemetry = UXTelemetry(enabled=False, output_path=str(output_path))

    telemetry.record("ignored_event")
    assert not output_path.exists()

    telemetry.set_enabled(True)
    telemetry.record("manual_event", value=7)

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2

    first = json.loads(rows[0])
    second = json.loads(rows[1])
    assert first["event"] == "telemetry_enabled"
    assert second["event"] == "manual_event"
    assert second["value"] == 7
