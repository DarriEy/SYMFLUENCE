# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for MPI launcher discovery (`symfluence.core.mpi_utils`).

Also guards against the regression that motivated this module: the ParFlow,
CLM-ParFlow and WRF-Hydro runners import ``find_mpirun`` at module load, so a
missing ``mpi_utils`` makes those models unimportable (which the old in-tree
import loop silently swallowed at debug level).
"""

from __future__ import annotations

import importlib
import os
import stat

import pytest

from symfluence.core import mpi_utils
from symfluence.core.mpi_utils import find_mpirun

MPI_MODELS = ["parflow", "clmparflow", "wrfhydro"]

# The bundled-launcher tests create real, extension-less, executable files,
# which only carry meaning on POSIX (Windows uses PATHEXT + has no exec bit).
# Cross-platform behaviour is covered by the shutil.which-mocked tests below.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="POSIX exec-bit / extension-less launcher semantics"
)


def _make_executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@posix_only
def test_find_mpirun_prefers_bundled(tmp_path):
    """A launcher next to the executable wins over PATH."""
    _make_executable(tmp_path / "mpirun")
    exe = tmp_path / "parflow.exe"
    exe.write_text("")
    assert find_mpirun(exe) == str(tmp_path / "mpirun")


@posix_only
def test_find_mpirun_bundled_mpiexec(tmp_path):
    """mpiexec is accepted as a bundled launcher too."""
    _make_executable(tmp_path / "mpiexec")
    assert find_mpirun(tmp_path / "model.exe") == str(tmp_path / "mpiexec")


@posix_only
def test_non_executable_bundled_file_is_ignored(tmp_path, monkeypatch):
    """A non-executable file named mpirun is not treated as a bundled launcher."""
    (tmp_path / "mpirun").write_text("not executable")  # no +x bit
    # Keep the real which for the bundled (path=<dir>) lookup so its exec-bit
    # check is exercised; neutralise the PATH (path=None) lookup so the bundled
    # file is the only candidate.
    real_which = mpi_utils.shutil.which
    monkeypatch.setattr(
        mpi_utils.shutil,
        "which",
        lambda name, path=None: real_which(name, path=path) if path is not None else None,
    )
    assert find_mpirun(tmp_path / "model.exe") is None


def test_find_mpirun_falls_back_to_path(tmp_path, monkeypatch):
    """With no bundled launcher, fall back to PATH via shutil.which."""
    # Bundled lookup passes path=<dir> and finds nothing; PATH lookup passes no
    # path and resolves mpirun.
    monkeypatch.setattr(
        mpi_utils.shutil,
        "which",
        lambda name, path=None: "/usr/bin/mpirun" if (name == "mpirun" and path is None) else None,
    )
    assert find_mpirun(tmp_path / "model.exe") == "/usr/bin/mpirun"


def test_find_mpirun_returns_none_when_absent(tmp_path, monkeypatch):
    """No bundled launcher and none on PATH -> None."""
    monkeypatch.setattr(mpi_utils.shutil, "which", lambda name, path=None: None)
    assert find_mpirun(tmp_path / "model.exe") is None


def test_find_mpirun_no_exe_searches_path(monkeypatch):
    """Called without an executable, it searches PATH only."""
    monkeypatch.setattr(
        mpi_utils.shutil,
        "which",
        lambda name, path=None: "/opt/mpiexec" if name == "mpiexec" else None,
    )
    assert find_mpirun() == "/opt/mpiexec"


@pytest.mark.parametrize("model", MPI_MODELS)
def test_mpi_dependent_model_imports(model):
    """ParFlow/CLM-ParFlow/WRF-Hydro import cleanly (regression: mpi_utils missing)."""
    mod = importlib.import_module(f"symfluence.models.{model}")
    assert callable(getattr(mod, "register", None))
