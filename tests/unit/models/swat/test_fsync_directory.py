"""Regression tests for SWAT run-directory fsync on Windows.

The SWAT runner and calibration worker flush the assembled TxtInOut
directory to disk before launching the Fortran executable.  The original
implementation opened the directory with ``os.open(dir, os.O_RDONLY)`` and
called ``os.fsync`` — a POSIX-only technique.  On Windows opening a
directory path this way raises ``PermissionError`` ([Errno 13]), which
crashed every native SWAT run during ``_prepare_run`` / ``run_model``:

    Failed during SWAT model execution: [Errno 13] Permission denied:
    '...\\simulations\\run_1\\SWAT'

These tests pin the fixed behaviour: flushing a *directory* must never
raise, and on Windows the call is a silent no-op.
"""
from __future__ import annotations

import os

import pytest

from symfluence.models.swat.calibration.worker import (
    _fsync_directory as worker_fsync_directory,
)
from symfluence.models.swat.runner import _fsync_directory as runner_fsync_directory


@pytest.mark.parametrize(
    "fsync_directory",
    [runner_fsync_directory, worker_fsync_directory],
    ids=["runner", "worker"],
)
def test_fsync_directory_does_not_raise_on_a_directory(fsync_directory, tmp_path):
    """Flushing a real directory must not raise (the [Errno 13] regression).

    On Windows, ``os.open(dir, O_RDONLY)`` raises PermissionError; this must
    be handled so the SWAT run does not crash during directory setup.
    """
    run_dir = tmp_path / "SWAT"
    run_dir.mkdir()
    # Populate as a real run dir would be, so a naive open()-as-file is doubly wrong.
    (run_dir / "file.cio").write_text("cio\n", encoding="utf-8")

    # Must complete cleanly regardless of platform.
    fsync_directory(run_dir)


@pytest.mark.parametrize(
    "fsync_directory",
    [runner_fsync_directory, worker_fsync_directory],
    ids=["runner", "worker"],
)
def test_fsync_directory_is_noop_on_windows(fsync_directory, tmp_path, monkeypatch):
    """On Windows the helper returns without ever touching os.open.

    Simulates ``os.name == 'nt'`` on any host and asserts os.open is never
    called (the operation that raised [Errno 13] on native Windows).
    """
    monkeypatch.setattr(os, "name", "nt")

    def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("os.open must not be called on Windows")

    monkeypatch.setattr(os, "open", _boom)

    # Path need not even exist; the helper short-circuits before any I/O.
    fsync_directory(tmp_path / "SWAT")


@pytest.mark.parametrize(
    "fsync_directory",
    [runner_fsync_directory, worker_fsync_directory],
    ids=["runner", "worker"],
)
def test_fsync_directory_tolerates_missing_path(fsync_directory, tmp_path):
    """A non-existent path is tolerated (best-effort flush, never fatal)."""
    fsync_directory(tmp_path / "does_not_exist" / "SWAT")
