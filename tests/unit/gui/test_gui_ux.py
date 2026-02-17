from __future__ import annotations

import json

import pytest

pn = pytest.importorskip("panel")

from symfluence.gui.utils.telemetry import UXTelemetry


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
