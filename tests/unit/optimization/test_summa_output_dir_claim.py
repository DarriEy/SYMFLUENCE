# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""An evaluation must never run into an output directory it cannot clear.

Native Windows model binaries deadlock at DLL_PROCESS_DETACH and never release
their output NetCDF handle (see symfluence.core.process_exec), so the stale
file cannot be deleted. _cleanup_stale_output_files used to log a suppressed
warning and continue, which gave two bad outcomes:

* SUMMA cannot create its output and dies with a bare ``STOP 1`` — no message
  at all, because the wedge also means its stdout buffer is never flushed;
* or the run "succeeds" and metrics are computed from the *previous*
  iteration's leftovers, which is exactly what that cleanup exists to prevent.

The second is the dangerous one: a silently wrong score.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.optimization.workers.summa.model_execution import (
    _cleanup_stale_output_files,
    _usable_output_dir,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_summa_output_dir_claim")


def test_clean_directory_is_used_as_is(tmp_path, logger):
    """No redirect when the directory can be cleared — the normal path."""
    (tmp_path / "old_timestep.nc").write_text("stale")
    assert _usable_output_dir(tmp_path, logger) == tmp_path
    assert not list(tmp_path.glob("*.nc")), "stale output should have been removed"


def test_cleanup_reports_files_it_could_not_remove(tmp_path, logger, monkeypatch):
    held = tmp_path / "held_timestep.nc"
    held.write_text("locked")

    def _refuse(self):
        raise PermissionError("held by a process that exited without releasing it")

    monkeypatch.setattr(Path, "unlink", _refuse)
    assert _cleanup_stale_output_files(tmp_path, logger) == [held]


def test_undeletable_output_forces_a_fresh_directory(tmp_path, logger, monkeypatch):
    """The fix: get a directory of our own rather than run into a poisoned one."""
    (tmp_path / "held_timestep.nc").write_text("locked")

    def _refuse(self):
        raise PermissionError("held")

    monkeypatch.setattr(Path, "unlink", _refuse)

    claimed = _usable_output_dir(tmp_path, logger)

    assert claimed != tmp_path
    assert claimed.is_dir()
    assert claimed.parent == tmp_path.parent
    # The decisive property: no glob in the new directory can reach the stale
    # file, so metrics cannot be computed from the previous iteration.
    assert not list(claimed.glob("*timestep.nc"))


def test_successive_claims_do_not_collide(tmp_path, logger, monkeypatch):
    """A second poisoned round must not hand back the same redirect."""
    (tmp_path / "held_timestep.nc").write_text("locked")
    real_unlink = Path.unlink

    def _refuse_only_in_root(self, *a, **k):
        if self.parent == tmp_path:
            raise PermissionError("held")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _refuse_only_in_root)

    first = _usable_output_dir(tmp_path, logger)
    # first is clean, so claiming again from the poisoned root must still move
    # somewhere usable rather than back onto the stale file
    second = _usable_output_dir(tmp_path, logger)
    assert first != tmp_path and second != tmp_path
    assert second.is_dir()
