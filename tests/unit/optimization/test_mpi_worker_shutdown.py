# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The persistent MPI worker pool must not outlive the calibration that made it.

The pool is created lazily inside ``execute_batch`` and deliberately survives
individual batches, so nothing below ``run_optimization`` knows when the run is
over. Before this was wired up, ``_shutdown_mpi_strategy()`` was reachable only
from the *failure* path of ``execute_batch`` and from ``cleanup()``, which has no
in-tree callers — so a calibration that SUCCEEDED never reaped its ranks. They
were re-parented to init and kept consuming a core each; sequential calibrations
piled up pools until the host was oversubscribed (a 14-config scaling sweep left
57 stale ranks on 16 cores, and its timings measured contention, not scaling).

These tests pin the three things that keep that from recurring: the run reaps the
pool on every exit path, the launcher is signalled as a process group so the ranks
actually die, and an interpreter that exits early still has a reaper registered.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.calibration.mixins.parallel.execution_strategies.mpi_persistent import (
    PersistentMPIExecutionStrategy,
)
from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer

# ---------------------------------------------------------------------------
# run_optimization always reaps the pool
# ---------------------------------------------------------------------------

def _mock_optimizer():
    """A stand-in for a constructed optimizer.

    ``run_optimization`` is called unbound so the test exercises the wrapper's
    control flow without building a real optimizer (which needs a full config,
    a domain on disk, and a model binary).
    """
    opt = MagicMock()
    opt._run_optimization_impl.return_value = Path("/tmp/results.json")
    return opt


def test_successful_run_shuts_down_the_worker_pool():
    """The regression: a run that succeeds must still reap its ranks."""
    opt = _mock_optimizer()

    result = BaseModelOptimizer.run_optimization(opt, "dds")

    assert result == Path("/tmp/results.json")
    opt._shutdown_mpi_strategy.assert_called_once_with()


def test_failed_run_shuts_down_the_worker_pool():
    """A crashing algorithm must not strand the pool either."""
    opt = _mock_optimizer()
    opt._run_optimization_impl.side_effect = RuntimeError("algorithm blew up")

    with pytest.raises(RuntimeError, match="algorithm blew up"):
        BaseModelOptimizer.run_optimization(opt, "dds")

    opt._shutdown_mpi_strategy.assert_called_once_with()


def test_shutdown_failure_does_not_mask_the_run_result():
    """Cleanup is best-effort: a failing reap must not swallow a good run."""
    opt = _mock_optimizer()
    opt._shutdown_mpi_strategy.side_effect = OSError("no such process")

    assert BaseModelOptimizer.run_optimization(opt, "dds") == Path("/tmp/results.json")


def test_shutdown_failure_does_not_mask_the_original_exception():
    opt = _mock_optimizer()
    opt._run_optimization_impl.side_effect = RuntimeError("the real failure")
    opt._shutdown_mpi_strategy.side_effect = OSError("no such process")

    with pytest.raises(RuntimeError, match="the real failure"):
        BaseModelOptimizer.run_optimization(opt, "dds")


# ---------------------------------------------------------------------------
# The pool is signalled as a group, so the ranks actually die
# ---------------------------------------------------------------------------

def _strategy(tmp_path) -> PersistentMPIExecutionStrategy:
    return PersistentMPIExecutionStrategy(tmp_path, num_processes=2, logger=MagicMock())


@pytest.mark.skipif(os.name == "nt", reason="POSIX session semantics")
def test_launcher_starts_its_own_process_group(tmp_path):
    """Without a new session, signalling the launcher can leave ranks running."""
    assert _strategy(tmp_path)._process_group_kwargs() == {"start_new_session": True}


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_signal_group_targets_the_whole_group(tmp_path):
    strat = _strategy(tmp_path)
    strat._process = MagicMock(pid=4321)

    with patch.object(os, "getpgid", return_value=4321) as getpgid, \
            patch.object(os, "killpg") as killpg:
        strat._signal_group(signal.SIGTERM)

    getpgid.assert_called_once_with(4321)
    killpg.assert_called_once_with(4321, signal.SIGTERM)
    # The group call succeeded, so the single-pid fallback must not also fire.
    strat._process.terminate.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_signal_group_falls_back_to_the_launcher_pid(tmp_path):
    """A vanished group must degrade to signalling the pid, not raise."""
    strat = _strategy(tmp_path)
    strat._process = MagicMock(pid=4321)

    with patch.object(os, "getpgid", side_effect=ProcessLookupError):
        strat._signal_group(signal.SIGTERM)

    strat._process.terminate.assert_called_once_with()


def test_shutdown_escalates_when_the_poison_file_is_ignored(tmp_path):
    """Ranks that ignore the graceful signal get SIGTERM, then SIGKILL."""
    strat = _strategy(tmp_path)
    comm = tmp_path / "comm"
    comm.mkdir()
    strat._comm_dir = comm

    proc = MagicMock(pid=99)
    proc.poll.return_value = None                     # never exits on its own
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="mpirun", timeout=30)
    strat._process = proc

    # Recorded from inside the first signal, because shutdown() removes the comm
    # directory on its way out — by the time it returns the file is gone.
    poison_seen = []

    with patch.object(strat, "_signal_group",
                      side_effect=lambda *_: poison_seen.append(
                          (comm / "shutdown").exists())) as sig:
        strat.shutdown()

    assert [c.args[0] for c in sig.call_args_list] == [signal.SIGTERM, signal.SIGKILL]
    assert poison_seen[0] is True                     # graceful path tried first
    assert strat._process is None
    assert not comm.exists()                          # comm dir cleaned up


def test_shutdown_is_idempotent(tmp_path):
    strat = _strategy(tmp_path)
    strat.shutdown()
    strat.shutdown()
    assert strat._process is None


# ---------------------------------------------------------------------------
# atexit safety net
# ---------------------------------------------------------------------------

def test_atexit_reaper_is_registered_while_the_pool_lives(tmp_path):
    strat = _strategy(tmp_path)
    with patch("atexit.register") as reg:
        strat._register_atexit_reaper()
    reg.assert_called_once()
    assert strat._atexit_hook is not None


def test_atexit_reaper_is_removed_on_shutdown(tmp_path):
    """A stale hook would signal a pid this run no longer owns."""
    strat = _strategy(tmp_path)
    strat._register_atexit_reaper()
    hook = strat._atexit_hook

    with patch("atexit.unregister") as unreg:
        strat.shutdown()

    unreg.assert_called_once_with(hook)
    assert strat._atexit_hook is None


def test_atexit_reaper_kills_a_pool_that_ignores_sigterm(tmp_path):
    strat = _strategy(tmp_path)
    proc = MagicMock(pid=77)
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="mpirun", timeout=5)
    strat._process = proc
    strat._register_atexit_reaper()

    with patch.object(strat, "_signal_group") as sig:
        strat._atexit_hook()

    assert [c.args[0] for c in sig.call_args_list] == [signal.SIGTERM, signal.SIGKILL]


def test_atexit_reaper_is_a_noop_when_the_pool_already_exited(tmp_path):
    strat = _strategy(tmp_path)
    proc = MagicMock(pid=77)
    proc.poll.return_value = 0        # already gone
    strat._process = proc
    strat._register_atexit_reaper()

    with patch.object(strat, "_signal_group") as sig:
        strat._atexit_hook()

    sig.assert_not_called()
