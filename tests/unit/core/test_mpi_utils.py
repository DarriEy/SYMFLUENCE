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


def _make_executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_find_mpirun_prefers_bundled(tmp_path):
    """A launcher next to the executable wins over PATH."""
    _make_executable(tmp_path / "mpirun")
    exe = tmp_path / "parflow.exe"
    exe.write_text("")
    assert find_mpirun(exe) == str(tmp_path / "mpirun")


def test_find_mpirun_bundled_mpiexec(tmp_path):
    """mpiexec is accepted as a bundled launcher too."""
    _make_executable(tmp_path / "mpiexec")
    assert find_mpirun(tmp_path / "model.exe") == str(tmp_path / "mpiexec")


def test_find_mpirun_falls_back_to_path(tmp_path, monkeypatch):
    """With no bundled launcher, fall back to PATH via shutil.which."""
    monkeypatch.setattr(
        mpi_utils.shutil, "which", lambda name: "/usr/bin/mpirun" if name == "mpirun" else None
    )
    # exe dir has no launcher next to it
    assert find_mpirun(tmp_path / "model.exe") == "/usr/bin/mpirun"


def test_find_mpirun_returns_none_when_absent(tmp_path, monkeypatch):
    """No bundled launcher and none on PATH -> None."""
    monkeypatch.setattr(mpi_utils.shutil, "which", lambda name: None)
    assert find_mpirun(tmp_path / "model.exe") is None


def test_find_mpirun_no_exe_searches_path(monkeypatch):
    """Called without an executable, it searches PATH only."""
    monkeypatch.setattr(
        mpi_utils.shutil, "which", lambda name: "/opt/mpiexec" if name == "mpiexec" else None
    )
    assert find_mpirun() == "/opt/mpiexec"


def test_bundled_launcher_must_be_executable(tmp_path, monkeypatch):
    """A non-executable file named mpirun is ignored (falls through to PATH)."""
    (tmp_path / "mpirun").write_text("not executable")  # no +x bit
    monkeypatch.setattr(mpi_utils.shutil, "which", lambda name: None)
    assert find_mpirun(tmp_path / "model.exe") is None


@pytest.mark.parametrize("model", MPI_MODELS)
def test_mpi_dependent_model_imports(model):
    """ParFlow/CLM-ParFlow/WRF-Hydro import cleanly (regression: mpi_utils missing)."""
    mod = importlib.import_module(f"symfluence.models.{model}")
    assert callable(getattr(mod, "register", None))
