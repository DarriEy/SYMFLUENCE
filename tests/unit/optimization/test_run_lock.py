# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the calibration run-directory lock (issue #329).

Two runs of the same model+experiment share per-process settings and
simulation dirs, so the second re-stages over the live one and its cleanup()
deletes them. Different models in one domain are scoped apart and must stay
free to run concurrently. These tests pin both halves of that behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from symfluence.core.calibration.optimizers.run_lock import RunDirectoryLock, lock_name


@pytest.fixture
def logger():
    return logging.getLogger("test_run_lock")


def test_acquire_creates_lock(tmp_path, logger):
    lock = RunDirectoryLock(tmp_path, "exp_a", logger)
    assert lock.acquire() is True
    assert lock.owned is True
    payload = json.loads(lock.path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["experiment_id"] == "exp_a"


def test_same_process_may_reacquire(tmp_path, logger):
    """A second optimizer in one process (e.g. final evaluation) is not a conflict."""
    RunDirectoryLock(tmp_path, "exp_a", logger).acquire()
    second = RunDirectoryLock(tmp_path, "exp_a", logger)
    assert second.acquire() is True


def test_live_foreign_holder_is_refused(tmp_path, logger):
    """The incident case: a second process must fail fast, not proceed."""
    psutil = pytest.importorskip("psutil")
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / lock_name("SUMMA", "exp_b")).write_text(json.dumps({
            "pid": victim.pid,
            "signature": f"{victim.pid}:{psutil.Process(victim.pid).create_time():.3f}",
            "experiment_id": "exp_b",
            "model": "SUMMA",
            "acquired_at": "2026-07-20T12:00:00Z",
        }))
        with pytest.raises(RuntimeError, match="already using"):
            RunDirectoryLock(tmp_path, "exp_b", logger, model="SUMMA").acquire()
    finally:
        victim.terminate()
        victim.wait()


def test_stale_lock_is_reclaimed(tmp_path, logger):
    """A lock left by a killed run must never block the next one."""
    (tmp_path / lock_name("", "exp_c")).write_text(json.dumps({
        "pid": 2 ** 22,  # not a live PID
        "signature": None,
        "experiment_id": "dead_run",
        "acquired_at": "2026-07-20T12:00:00Z",
    }))
    assert RunDirectoryLock(tmp_path, "exp_c", logger).acquire() is True


def test_nonzero_mpi_rank_does_not_arbitrate(tmp_path, logger, monkeypatch):
    """Ranks of one mpirun share the tree legitimately; only rank 0 locks."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    lock = RunDirectoryLock(tmp_path, "exp_d", logger)
    assert lock.acquire() is False
    assert not lock.path.exists()


def test_release_removes_only_own_lock(tmp_path, logger):
    lock = RunDirectoryLock(tmp_path, "exp_e", logger)
    lock.acquire()
    lock.release()
    assert not lock.path.exists()

    # a lock owned by someone else survives our release
    foreign = RunDirectoryLock(tmp_path, "exp_f", logger)
    foreign.acquire()
    foreign.owned = True
    foreign.path.write_text(json.dumps({
        "pid": 2 ** 22, "signature": None,
        "experiment_id": "other", "acquired_at": "2026-07-20T12:00:00Z",
    }))
    foreign.release()
    assert foreign.path.exists()


def test_unwritable_directory_does_not_block_calibration(tmp_path, logger, monkeypatch):
    """The lock is a safety net; it must never be a prerequisite to run."""
    def boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("pathlib.Path.write_text", boom)
    lock = RunDirectoryLock(tmp_path, "exp_g", logger)
    assert lock.acquire() is False  # degraded, but no exception


def test_different_models_may_run_concurrently(tmp_path, logger):
    """Different models share run_<algorithm>/ legitimately — per-process dirs
    are scoped by experiment and model, so they must NOT block each other."""
    psutil = pytest.importorskip("psutil")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / lock_name("SUMMA", "run_1")).write_text(json.dumps({
            "pid": other.pid,
            "signature": f"{other.pid}:{psutil.Process(other.pid).create_time():.3f}",
            "experiment_id": "run_1", "model": "SUMMA", "acquired_at": "x",
        }))
        # a different model in the same run dir is fine
        assert RunDirectoryLock(tmp_path, "run_1", logger, model="FUSE").acquire() is True
        # the same model+experiment is the real collision
        with pytest.raises(RuntimeError, match="already using"):
            RunDirectoryLock(tmp_path, "run_1", logger, model="SUMMA").acquire()
    finally:
        other.terminate()
        other.wait()
