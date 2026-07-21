# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Every FUSE invocation must be time-bounded and must not inherit stdin.

FUSE's calibration path has always passed ``timeout=FUSE_TIMEOUT``, but the
runner and subcatchment paths did not. A wedged binary therefore blocked the
Python workflow forever rather than failing: observed on Windows as fuse.exe
sitting at 0% CPU for 37h after writing nothing but its output NetCDF header,
holding a reproduction run slot the entire time while the campaign appeared
merely "slow". These tests pin the bound so the failure stays loud and finite.
"""
from __future__ import annotations

import subprocess

import pytest

from symfluence.models.fuse import runner as fuse_runner
from symfluence.models.fuse import subcatchment_processor as fuse_subcat

MODULES = [fuse_runner, fuse_subcat]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.split(".")[-1])
def test_fuse_is_invoked_with_a_timeout_and_no_inherited_stdin(module, monkeypatch):
    """Pin the call contract at every fuse.exe site in the module."""
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    src = module.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()

    # Each subprocess.run in these modules launches a model binary; none may
    # be unbounded, and none may inherit the parent's stdin.
    for block in text.split("subprocess.run(")[1:]:
        head = block[: block.index("\n            )") + 1] if "\n            )" in block else block[:600]
        assert "timeout=" in head, f"unbounded subprocess.run in {src}:\n{head}"
        assert "stdin=subprocess.DEVNULL" in head, f"stdin inherited in {src}:\n{head}"


def test_mixin_applies_a_backstop_when_no_timeout_is_given(monkeypatch, tmp_path):
    """The canonical runner entry point must never wait forever.

    ``timeout=None`` previously meant "wait forever", and most call sites took
    that default — which is how a wedged fuse.exe held a run slot for 37h.
    """
    import logging

    from symfluence.models.mixins import subprocess_execution as se

    seen = {}

    class _Completed:
        returncode = 0
        stdout = stderr = ""

    def _fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _Completed()

    monkeypatch.setattr(se.subprocess, "run", _fake_run)
    monkeypatch.delenv("SYMFLUENCE_SUBPROCESS_TIMEOUT", raising=False)

    class _Runner(se.SubprocessExecutionMixin):
        logger = logging.getLogger("test_backstop")

    _Runner().execute_subprocess(["x"], tmp_path / "l.log", check=False)

    assert seen["timeout"] == se.DEFAULT_SUBPROCESS_TIMEOUT
    assert seen["stdin"] is subprocess.DEVNULL

    # An explicit caller timeout still wins...
    seen.clear()
    _Runner().execute_subprocess(["x"], tmp_path / "l.log", check=False, timeout=42)
    assert seen["timeout"] == 42

    # ...and the escape hatch restores unbounded behaviour.
    seen.clear()
    monkeypatch.setenv("SYMFLUENCE_SUBPROCESS_TIMEOUT", "0")
    _Runner().execute_subprocess(["x"], tmp_path / "l.log", check=False)
    assert seen["timeout"] is None


def test_timeout_is_reported_as_failure_not_raised(monkeypatch, tmp_path):
    """A timeout must degrade to a logged failure, not escape as an exception."""
    import logging

    class _Runner(fuse_runner.FUSERunner):
        def __init__(self):  # bypass the heavy real __init__
            self.logger = logging.getLogger("test_fuse_timeout")
            self.output_path = tmp_path
            self.setup_dir = tmp_path
            self.fuse_exe = tmp_path / "fuse.exe"
            self.domain_name = "d"

        def _get_config_value(self, _getter, default=None, dict_key=None):
            return default

    def _timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 3600))

    monkeypatch.setattr(fuse_runner.subprocess, "run", _timeout)

    runner = _Runner()
    # Should return False rather than propagating TimeoutExpired or hanging.
    assert runner._execute_fuse_distributed() is False
