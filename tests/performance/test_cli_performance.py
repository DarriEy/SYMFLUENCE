# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CLI performance regression tests.

Guards the lazy-import discipline at CLI startup (RTI architecture review
item 20): heavy dependencies (xarray, geopandas, rasterio, torch) must not be
imported eagerly by the CLI path. The regression this catches adds several
seconds to every command, so the ceilings are generous — cold start is ~5 s
locally and CI runners are slower — while still failing on an
order-of-magnitude jump from an eager heavy import.

Each measurement runs in a fresh subprocess so module caching in the test
runner cannot mask a regression.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.cli]

# Ceilings in seconds; see module docstring for the calibration rationale.
IMPORT_TIME_LIMIT_S = 15.0
COMMAND_TIME_LIMIT_S = 20.0


def _timed_run(cmd: list[str]) -> float:
    """Run a command in a fresh subprocess and return its wall-clock seconds."""
    start = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return time.perf_counter() - start


def test_cli_parser_import_time():
    """Importing the CLI argument parser must stay lightweight."""
    elapsed = _timed_run([sys.executable, "-c", "import symfluence.cli.argument_parser"])
    assert elapsed < IMPORT_TIME_LIMIT_S, (
        f"CLI parser import took {elapsed:.2f}s (limit {IMPORT_TIME_LIMIT_S}s) — "
        "a heavy dependency is likely being imported eagerly at CLI startup"
    )


def test_cli_help_time():
    """`symfluence --help` must respond without loading the scientific stack."""
    elapsed = _timed_run([sys.executable, "-m", "symfluence", "--help"])
    assert elapsed < COMMAND_TIME_LIMIT_S, (
        f"'symfluence --help' took {elapsed:.2f}s (limit {COMMAND_TIME_LIMIT_S}s) — "
        "a heavy dependency is likely being imported eagerly at CLI startup"
    )


def test_cli_list_steps_time():
    """`symfluence workflow list-steps` must respond without a full framework load."""
    elapsed = _timed_run([sys.executable, "-m", "symfluence", "workflow", "list-steps"])
    assert elapsed < COMMAND_TIME_LIMIT_S, (
        f"'symfluence workflow list-steps' took {elapsed:.2f}s (limit {COMMAND_TIME_LIMIT_S}s) — "
        "a heavy dependency is likely being imported eagerly on the workflow path"
    )
