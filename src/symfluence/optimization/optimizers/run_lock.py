# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Exclusive lock over a calibration's parallel run directory.

Per-process calibration directories are keyed by domain and algorithm only
(``simulations/run_<algorithm>/process_<i>``), so two SYMFLUENCE processes
working the same domain with the same algorithm share one directory tree.
Merely constructing a second optimizer re-stages settings over a live run's
files, and calling its ``cleanup()`` deletes them outright — observed
destroying a live NSGA-II calibration mid-run, after which every evaluation
failed and the workflow still reported success (see issue #329).

This module makes that collision loud and immediate instead of silent and
destructive: the first process to reach the directory writes a lock naming
itself, and a second live process refuses to start.

The lock is deliberately advisory-but-checked rather than an OS file lock:
runs are long, machines crash, and a hard lock left behind by a killed job
would be worse than the collision it prevents. Liveness is therefore checked
against the recorded PID *and* its start time, so a stale lock from a dead or
recycled PID is reclaimed automatically.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOCK_NAME = ".symfluence_run.lock"


def _mpi_rank() -> int:
    """Best-effort MPI rank of this process (0 when not under MPI).

    Only rank 0 arbitrates the lock: every rank of one ``mpirun`` shares the
    run directory by design, so rank-blind locking would have the job's own
    ranks refuse to start.
    """
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK",
                "MV2_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _process_signature(pid: int) -> Optional[str]:
    """Return a stable identity for a live PID, or None if it is not running.

    The start time distinguishes a genuinely live owner from an unrelated
    process that happens to have inherited a recycled PID.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return f"{pid}:{psutil.Process(pid).create_time():.3f}"
    except (psutil.Error, OSError, ValueError):
        # No such process, or it is not inspectable — either way we must not
        # treat it as a live owner. Failing open is deliberate: a lock that
        # cannot be verified should never block a legitimate calibration.
        return None


class RunDirectoryLock:
    """Guards one ``simulations/run_<algorithm>`` tree against concurrent use."""

    def __init__(self, base_dir: Path, experiment_id: str, logger: Any):
        self.path = Path(base_dir) / LOCK_NAME
        self.experiment_id = experiment_id
        self.logger = logger
        self.owned = False

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def _payload(self) -> dict:
        pid = os.getpid()
        return {
            "pid": pid,
            "signature": _process_signature(pid),
            "experiment_id": self.experiment_id,
            "acquired_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def acquire(self) -> bool:
        """Take the lock. Returns True when this instance owns it.

        Returns False (without raising) for non-zero MPI ranks, which share
        the directory with their own rank 0 legitimately.

        Raises:
            RuntimeError: another live process holds the lock.
        """
        if _mpi_rank() != 0:
            return False

        existing = self._read()
        if existing:
            holder_pid = existing.get("pid")
            holder_sig = existing.get("signature")
            if holder_pid == os.getpid():
                self.owned = True  # re-entry within one process
                return True
            live_sig = _process_signature(holder_pid) if holder_pid else None
            still_live = live_sig is not None and (
                holder_sig is None or holder_sig == live_sig
            )
            if still_live:
                raise RuntimeError(
                    f"Another SYMFLUENCE calibration is already using "
                    f"{self.path.parent}: PID {holder_pid} "
                    f"(experiment '{existing.get('experiment_id')}', started "
                    f"{existing.get('acquired_at')}).\n"
                    f"  Per-process calibration directories are shared by domain "
                    f"and algorithm, so running a second one here would re-stage "
                    f"settings over the live run and corrupt both.\n"
                    f"  Wait for that run to finish, or use a different "
                    f"algorithm/domain. If PID {holder_pid} is definitely gone, "
                    f"delete {self.path}."
                )
            self.logger.info(
                "Reclaiming stale run lock in %s (PID %s is no longer running)",
                self.path.parent, holder_pid,
            )

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._payload(), indent=2))
            self.owned = True
        except OSError as exc:
            # An unwritable directory should not stop a calibration; the
            # lock is a safety net, not a prerequisite.
            self.logger.warning("Could not write run lock %s: %s", self.path, exc)
            self.owned = False
        return self.owned

    def release(self) -> None:
        """Drop the lock if this process holds it."""
        if not self.owned:
            return
        existing = self._read()
        if existing and existing.get("pid") not in (None, os.getpid()):
            return  # someone else's lock; never remove it
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self.owned = False
