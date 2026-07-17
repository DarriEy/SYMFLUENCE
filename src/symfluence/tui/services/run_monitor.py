# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Sidebar data for the modelling chat: what is the experiment doing right now?

Fed by cheap file reads (the same ``agent.inspection`` helpers the MCP tools
use, plus the background-job records) — deliberately independent of the agent
itself, so a long-blocking tool call cannot freeze the sidebar. Textual-free
and unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunStatus:
    """One sidebar refresh."""

    config_name: str | None = None
    domain: str | None = None
    calibration: dict | None = None      # inspection.calibration_status payload
    jobs: list[dict] = field(default_factory=list)   # active/most recent jobs
    last_log_line: str | None = None


class RunMonitor:
    """Polls the domain tree and job records for the chat sidebar."""

    def __init__(self, config_path: str | Path | None):
        self.config_path = Path(config_path) if config_path else None

    def poll(self) -> RunStatus:
        """One refresh; every section degrades to None/[] on any problem."""
        from symfluence.agent import jobs as jobs_mod
        from symfluence.agent.inspection import (
            calibration_status,
            read_run_log,
            resolve_domain,
        )

        config_name = self.config_path.name if self.config_path else None
        domain = None
        calibration = None
        last_log_line = None

        if self.config_path is not None:
            try:
                domain, _ = resolve_domain(self.config_path)
            except (ValueError, OSError):
                domain = None
            try:
                calibration = calibration_status(self.config_path)
            except (ValueError, OSError):
                calibration = None
            try:
                lines = read_run_log(self.config_path, tail_lines=1)['lines']
                last_log_line = lines[-1] if lines else None
            except (ValueError, OSError):
                last_log_line = None

        try:
            all_jobs = jobs_mod.list_jobs()
            running = [j for j in all_jobs if j['state'] == jobs_mod.RUNNING]
            jobs = running or all_jobs[:1]
        except OSError:
            jobs = []

        return RunStatus(
            config_name=config_name,
            domain=domain,
            calibration=calibration,
            jobs=jobs,
            last_log_line=last_log_line,
        )
