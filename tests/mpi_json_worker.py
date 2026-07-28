"""Importable worker used by the real MPI JSON hand-off smoke test."""

from __future__ import annotations

from typing import Any


def double_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "individual_id": task["individual_id"],
        "score": float(task["value"]) * 2.0,
        "params": task.get("params", {}),
    }
