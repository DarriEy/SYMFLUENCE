# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""TauDEM `module load` execution is hardened against shell injection (review item 13).

The module-load path must run through a shell, so the command is validated (module
names + an allowlisted TauDEM executable), fully shlex-quoted, and executed via an
explicit ``bash -lc`` argv with ``shell=False`` instead of ``shell=True`` on a raw
interpolated string.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.geospatial.geofabric.processors import taudem_executor
from symfluence.geospatial.geofabric.processors.taudem_executor import TauDEMExecutor


def _executor(num_processes=4):
    return TauDEMExecutor(
        {"NUM_PROCESSES": num_processes}, logging.getLogger("test"), "/opt/taudem"
    )


def test_valid_command_is_quoted_and_prefixed():
    ex = _executor()
    out = ex._safe_module_load_command(
        "module load gdal taudem && pitremove -z dem.tif", "mpirun", has_mpi_prefix=False
    )
    assert out == "module load gdal taudem && mpirun -n 4 pitremove -z dem.tif"


def test_already_prefixed_command_not_double_prefixed():
    ex = _executor()
    out = ex._safe_module_load_command(
        "module load taudem && mpirun -n 4 pitremove -z dem.tif", "mpirun", has_mpi_prefix=True
    )
    assert out == "module load taudem && mpirun -n 4 pitremove -z dem.tif"


def test_no_mpi_launcher_omits_prefix():
    ex = _executor()
    out = ex._safe_module_load_command(
        "module load taudem && pitremove -z dem.tif", None, has_mpi_prefix=False
    )
    assert out == "module load taudem && pitremove -z dem.tif"


def test_rejects_non_allowlisted_executable():
    ex = _executor()
    with pytest.raises(ValueError, match="allowlisted TauDEM"):
        ex._safe_module_load_command(
            "module load taudem && rm -rf /", "mpirun", has_mpi_prefix=False
        )


def test_rejects_injection_in_module_name():
    ex = _executor()
    with pytest.raises(ValueError, match="module name"):
        ex._safe_module_load_command(
            "module load evil;rm && pitremove -z dem.tif", "mpirun", has_mpi_prefix=False
        )


def test_injection_metacharacters_are_quoted_not_executed():
    """A ';' smuggled into an argument is shlex-quoted, so it cannot chain a command."""
    ex = _executor()
    out = ex._safe_module_load_command(
        "module load taudem && pitremove -z 'dem.tif; rm -rf /'", "mpirun", has_mpi_prefix=False
    )
    # The dangerous token is a single quoted argument, not a shell separator.
    assert "'dem.tif; rm -rf /'" in out
    assert out.count("&&") == 1


def test_module_load_runs_via_bash_lc_without_shell():
    """End-to-end: the module-load path uses ['bash','-lc',...] with shell=False."""
    ex = _executor()
    completed = MagicMock(stdout="", stderr="", returncode=0)
    with patch.object(TauDEMExecutor, "_get_mpi_command", return_value="mpirun"), patch.object(
        taudem_executor.subprocess, "run", return_value=completed
    ) as mock_run:
        ex.run_command("module load gdal && pitremove -z dem.tif", retry=False)

    args, kwargs = mock_run.call_args
    full_command = args[0]
    assert full_command[:2] == ["bash", "-lc"]
    assert kwargs["shell"] is False
    assert "pitremove" in full_command[2]
