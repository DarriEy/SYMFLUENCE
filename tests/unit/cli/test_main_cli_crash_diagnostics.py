"""
Tests for the CLI's native-crash diagnostics.

Without the fault handler a crash inside a compiled dependency (PyTorch, GDAL,
HDF5, a model binary) kills the interpreter silently: the log stops mid-step and
the shell reports only exit 139. Enabling faulthandler makes the interpreter
print the offending stack to stderr, where the workflow logs capture it.
"""
from __future__ import annotations

import faulthandler
from unittest.mock import patch

from symfluence.main_cli import enable_crash_diagnostics


class TestEnableCrashDiagnostics:
    """Tests for enable_crash_diagnostics()."""

    def test_enables_the_fault_handler(self, monkeypatch):
        """The handler is installed so native faults print a Python stack."""
        monkeypatch.delenv("SYMFLUENCE_NO_FAULTHANDLER", raising=False)

        was_enabled = faulthandler.is_enabled()
        try:
            faulthandler.disable()
            assert enable_crash_diagnostics() is True
            assert faulthandler.is_enabled()
        finally:
            if was_enabled:
                faulthandler.enable()

    def test_opt_out_via_environment(self, monkeypatch):
        """SYMFLUENCE_NO_FAULTHANDLER leaves the interpreter untouched."""
        monkeypatch.setenv("SYMFLUENCE_NO_FAULTHANDLER", "1")

        with patch("faulthandler.enable") as enable:
            assert enable_crash_diagnostics() is False
        enable.assert_not_called()

    def test_unusable_stderr_does_not_break_startup(self, monkeypatch):
        """A redirected/closed stderr must not stop the CLI from starting."""
        monkeypatch.delenv("SYMFLUENCE_NO_FAULTHANDLER", raising=False)

        with patch("faulthandler.is_enabled", return_value=False), \
                patch("faulthandler.enable", side_effect=ValueError("sys.stderr is invalid")):
            assert enable_crash_diagnostics() is False
