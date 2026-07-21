# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""A process that exits without signaling must still be reported as finished.

On Windows the process object signals only when its last thread exits, and the
native model binaries here regularly terminate with a valid exit code while
leaving a thread parked in a kernel wait. ``Popen.wait()`` blocks on that
signal and ``Popen.poll()`` gates on it too, so the stdlib cannot observe a
result the OS already has: a SUMMA worker that failed instantly with STOP 1
still billed its full 7200s timeout.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from symfluence.core import process_exec


class _NeverSignals:
    """A Popen whose wait() always times out, mimicking the wedged case."""

    def __init__(self, exit_code=1):
        self.args = ["fake.exe"]
        self.returncode = None
        self._handle = 1234
        self._exit_code = exit_code
        self.stdout = self.stderr = self.stdin = None
        self.waits = 0
        self.killed = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.returncode is not None:
            return self.returncode
        raise subprocess.TimeoutExpired(self.args, timeout)

    def kill(self):
        self.killed = True


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(process_exec, "IS_WINDOWS", True)


def test_unsignaled_exit_is_reported_promptly(monkeypatch, on_windows):
    """The whole point: don't wait on a process that already finished."""
    proc = _NeverSignals(exit_code=1)
    monkeypatch.setattr(process_exec, "_terminated_exit_code", lambda p: p._exit_code)

    started = time.monotonic()
    assert process_exec.wait(proc, timeout=7200) == 1
    assert proc.returncode == 1
    # Must not have burned anything close to the timeout.
    assert time.monotonic() - started < 5


def test_still_running_process_still_times_out(monkeypatch, on_windows):
    """The escape hatch must not mask a genuinely running process."""
    proc = _NeverSignals()
    monkeypatch.setattr(process_exec, "_terminated_exit_code", lambda p: None)

    with pytest.raises(subprocess.TimeoutExpired):
        process_exec.wait(proc, timeout=0.5)


def test_still_active_code_is_not_mistaken_for_exit(monkeypatch, on_windows):
    """A running process reads as STILL_ACTIVE; that is not a result."""
    proc = _NeverSignals()

    # _terminated_exit_code returns None for "running" — assert the real
    # helper's contract is what wait() relies on, not the raw 259.
    monkeypatch.setattr(process_exec, "_terminated_exit_code",
                        lambda p: None)
    with pytest.raises(subprocess.TimeoutExpired):
        process_exec.wait(proc, timeout=0.3)

    # ...but a process that genuinely exited *with* 259 is a real result.
    monkeypatch.setattr(process_exec, "_terminated_exit_code",
                        lambda p: process_exec.STILL_ACTIVE)
    assert process_exec.wait(proc, timeout=5) == process_exec.STILL_ACTIVE


def test_posix_delegates_to_popen_wait(monkeypatch):
    monkeypatch.setattr(process_exec, "IS_WINDOWS", False)
    proc = _NeverSignals()
    with pytest.raises(subprocess.TimeoutExpired):
        process_exec.wait(proc, timeout=0.1)
    assert proc.waits == 1  # single delegated call, no polling loop


# --- end-to-end against real processes -------------------------------------

def test_run_matches_subprocess_run_for_normal_processes(tmp_path):
    """The shim must be boring for processes that behave."""
    log = tmp_path / "out.log"
    with open(log, "w") as fh:
        result = process_exec.run(
            [sys.executable, "-c", "print('hello'); raise SystemExit(3)"],
            stdout=fh, stderr=subprocess.STDOUT,
        )
    assert result.returncode == 3
    assert "hello" in log.read_text()


def test_run_captures_piped_output():
    result = process_exec.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('out'); sys.stderr.write('err')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert result.returncode == 0
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_run_supports_capture_output():
    """capture_output is a run()-only kwarg; Popen rejects it outright."""
    result = process_exec.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('o'); sys.stderr.write('e')"],
        capture_output=True, text=True,
    )
    assert (result.stdout, result.stderr) == ("o", "e")


def test_capture_output_conflict_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="capture_output"):
        with open(tmp_path / "x.log", "w") as fh:
            process_exec.run([sys.executable, "-c", "pass"],
                             capture_output=True, stdout=fh)


def test_run_check_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        process_exec.run([sys.executable, "-c", "raise SystemExit(2)"], check=True)


def test_run_timeout_kills_and_raises():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        process_exec.run([sys.executable, "-c", "import time; time.sleep(30)"],
                         timeout=1)
    assert time.monotonic() - started < 20


def test_timeout_path_leaves_a_returncode(monkeypatch, on_windows):
    """An unkillable process must not be left for a later unbounded wait.

    Popen.__exit__/__del__ call wait() with no timeout; if the timeout path
    left returncode as None it would hang there instead, which is the bug
    this module exists to remove.
    """
    proc = _NeverSignals()
    monkeypatch.setattr(process_exec.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(process_exec, "_terminated_exit_code", lambda p: None)

    with pytest.raises(subprocess.TimeoutExpired):
        process_exec.run(["fake.exe"], timeout=0.3)

    assert proc.killed
    assert proc.returncode is not None
