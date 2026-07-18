# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure lifecycle decisions shared by model optimizers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def adjust_end_time_for_forcing(end_time: str, timestep_seconds: int) -> str:
    """Align a simulation end time with the last forcing step of its day."""
    if timestep_seconds < 3600:
        return end_time

    parsed = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    timestep_hours = timestep_seconds / 3600
    last_hour = max(0, int(24 - (24 % timestep_hours) - timestep_hours))
    if parsed.hour > last_hour or (parsed.hour == 23 and last_hour < 23):
        return parsed.replace(hour=last_hour, minute=0).strftime("%Y-%m-%d %H:%M")
    return end_time


def fallback_simulation_dir(temp_root: str, domain_name: str, algorithm: str) -> Path:
    """Return the deterministic fallback directory for an optimization run."""
    return Path(temp_root) / "symfluence" / domain_name / f"run_{algorithm}"
