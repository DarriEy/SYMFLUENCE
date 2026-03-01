# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Lightweight opt-in GUI telemetry.

Writes JSONL events to a local file for UX improvement analysis.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UXTelemetry:
    """Tiny opt-in telemetry sink for GUI interaction events."""

    def __init__(self, enabled: bool = False, output_path: str | None = None):
        env_path = os.environ.get("SYMFLUENCE_GUI_TELEMETRY_PATH")
        self._path = Path(output_path or env_path or (Path.home() / ".symfluence" / "gui_telemetry.jsonl"))
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable recording."""
        self._enabled = bool(enabled)
        if self._enabled:
            self.record("telemetry_enabled", path=str(self._path))

    def record(self, event: str, **payload: Any) -> None:
        """Write a telemetry event when opt-in is enabled."""
        if not self._enabled:
            return
        row: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:  # noqa: BLE001 — UI resilience
            # Telemetry is non-critical; never break GUI flows.
            return
