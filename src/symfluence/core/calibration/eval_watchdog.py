# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wall-clock watchdog for calibration evaluations.

A single model evaluation is seconds-to-minutes of work, but on Windows we have
observed individual evaluations wedge *indefinitely* on things no Python thread
or signal can interrupt: a JAX/XLA compile deadlock, and a ``gc.collect()``
finalizer deadlock (an xarray/netCDF handle closing into the netCDF-S3 shutdown
hang). One such eval freezes the whole calibration run for days, so an
orchestrating group runner never advances.

This watchdog converts that indefinite freeze into a fast, recoverable process
death: the optimizer ``arm()``s it, each evaluation ``beat()``s it, and if no
beat arrives for longer than the timeout a background thread dumps every
thread's stack (so the log shows *where* it wedged) and calls ``os._exit`` --
NOT ``sys.exit``, because the hang may itself be in a finalizer/atexit path that
a clean shutdown would re-enter and deadlock on. The orchestrator then re-spawns
the config on a fresh process.

Configure with ``SYMFLUENCE_EVAL_WATCHDOG_SECONDS`` (default 1800; ``0`` or
negative disables). Size it above the slowest *legitimate* single evaluation.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from typing import Optional

_DEFAULT_TIMEOUT_S = 1800.0          # 30 min
_EXIT_CODE = 75                      # distinct, so a harness can tag watchdog kills
_ENV_VAR = "SYMFLUENCE_EVAL_WATCHDOG_SECONDS"


class EvalWatchdog:
    """No-progress detector that self-terminates a wedged calibration process."""

    def __init__(self, timeout_s: float, logger=None):
        self._timeout = float(timeout_s)
        self._logger = logger
        self._last_beat: Optional[float] = None
        self._armed = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._timeout > 0

    def arm(self) -> None:
        """Begin (or resume) monitoring; starts the daemon thread on first use."""
        if not self.enabled:
            return
        with self._lock:
            self._armed = True
            self._last_beat = time.monotonic()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._monitor, name="eval-watchdog", daemon=True
                )
                self._thread.start()
        if self._logger:
            self._logger.info(
                f"Eval watchdog armed: a single evaluation exceeding "
                f"{self._timeout:.0f}s terminates the run for auto-recovery "
                f"(set {_ENV_VAR}=0 to disable)."
            )

    def beat(self) -> None:
        """Record evaluation progress. Cheap; safe to call on every evaluation."""
        if not self.enabled:
            return
        with self._lock:
            if self._armed:
                self._last_beat = time.monotonic()

    def disarm(self) -> None:
        """Stop monitoring (e.g. once optimization returns)."""
        with self._lock:
            self._armed = False

    def _monitor(self) -> None:
        poll = max(5.0, min(30.0, self._timeout / 6.0))
        while True:
            time.sleep(poll)
            with self._lock:
                armed, last = self._armed, self._last_beat
            if not armed or last is None:
                continue
            stuck = time.monotonic() - last
            if stuck > self._timeout:
                self._fire(stuck)

    def _fire(self, stuck: float) -> None:
        msg = (
            f"EVAL WATCHDOG: no evaluation progress for {stuck:.0f}s "
            f"(limit {self._timeout:.0f}s). The run is wedged on a single "
            f"evaluation (e.g. a JAX/XLA compile or gc/finalizer deadlock). "
            f"Dumping thread stacks and terminating so the orchestrator can "
            f"recover."
        )
        try:
            if self._logger:
                self._logger.error(msg)
        except Exception:  # noqa: BLE001 — never mask the kill
            pass
        try:
            sys.stderr.write("\n" + msg + "\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(all_threads=True)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        # os._exit, not sys.exit: skip atexit/finalizers -- the hang may BE in a
        # finalizer/gc, which a clean shutdown would re-enter and deadlock on.
        os._exit(_EXIT_CODE)


_singleton: Optional[EvalWatchdog] = None
_singleton_lock = threading.Lock()


def get_watchdog(logger=None) -> EvalWatchdog:
    """Return the process-wide watchdog, constructing it from the env on first use."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            raw = os.environ.get(_ENV_VAR)
            try:
                timeout = float(raw) if raw not in (None, "") else _DEFAULT_TIMEOUT_S
            except (TypeError, ValueError):
                timeout = _DEFAULT_TIMEOUT_S
            _singleton = EvalWatchdog(timeout, logger)
        elif logger is not None and _singleton._logger is None:
            _singleton._logger = logger
        return _singleton
