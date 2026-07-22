# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests for the embedded-R runtime setup used by the GR models.

These pin the two behaviours that made GR4J unusable on native Windows:

  1. R's own ``bin`` directory must reach ``PATH`` before rpy2 starts the
     embedded interpreter. rpy2 only registers it with
     ``os.add_dll_directory()``, which R's internal ``dyn.load()`` does not
     consult — so ``stats.dll`` fails to find ``Rlapack.dll`` and every
     compiled R package, airGR included, becomes unloadable.
  2. An unloadable airGR must raise an actionable error. The previous
     behaviour — fall back to ``install.packages()`` against CRAN — wedged the
     interpreter into a SIGSEGV on offline machines, 53 minutes into a run.
"""
from __future__ import annotations

import os
import sys

import pytest

from symfluence.core.exceptions import ModelExecutionError
from symfluence.models.gr import r_environment

pytestmark = [pytest.mark.unit, pytest.mark.quick]


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test gets a fresh detection; the real functions memoise."""
    monkeypatch.setattr(r_environment, "_configured", False)
    monkeypatch.setattr(r_environment, "_r_bin_dir", None)
    monkeypatch.setattr(r_environment, "_airgr_verified", False)


@pytest.fixture
def fake_r_home(tmp_path, monkeypatch):
    """A minimal Windows R installation tree."""
    r_home = tmp_path / "R-4.6.1"
    bin_dir = r_home / "bin" / "x64"
    bin_dir.mkdir(parents=True)
    (bin_dir / "R.dll").write_bytes(b"MZ")
    (bin_dir / "Rlapack.dll").write_bytes(b"MZ")
    monkeypatch.setenv("R_HOME", str(r_home))
    return r_home, bin_dir


class _FakeRobjects:
    """Stand-in for ``rpy2.robjects`` that returns canned R results."""

    def __init__(self, loadable: bool, diagnosis: str = "diagnosis text"):
        self._loadable = loadable
        self._diagnosis = diagnosis
        self.calls: list[str] = []

    def r(self, code: str):
        self.calls.append(code)
        if "loadNamespace" in code and "paste" in code:
            return [self._diagnosis]
        return [self._loadable]


class TestConfigureRDllSearch:
    def test_noop_off_windows(self, monkeypatch, fake_r_home):
        monkeypatch.setattr(sys, "platform", "linux")
        before = os.environ["PATH"]
        assert r_environment.configure_r_dll_search() is None
        assert os.environ["PATH"] == before

    def test_prepends_r_bin_to_path(self, monkeypatch, fake_r_home):
        r_home, bin_dir = fake_r_home
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)
        monkeypatch.setenv("PATH", "C:\\somewhere")

        assert r_environment.configure_r_dll_search() == bin_dir
        assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)

    def test_does_not_duplicate_existing_path_entry(self, monkeypatch, fake_r_home):
        r_home, bin_dir = fake_r_home
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)
        monkeypatch.setenv("PATH", os.pathsep.join(["C:\\somewhere", str(bin_dir)]))

        r_environment.configure_r_dll_search()
        assert os.environ["PATH"].split(os.pathsep).count(str(bin_dir)) == 1

    def test_sets_r_home_with_forward_slashes(self, monkeypatch, fake_r_home):
        """A backslash R_HOME is rejected by the embedded interpreter."""
        r_home, _ = fake_r_home
        monkeypatch.delenv("R_HOME")
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)
        monkeypatch.setattr(r_environment, "_candidate_r_homes", lambda: [r_home])

        r_environment.configure_r_dll_search()
        assert os.environ["R_HOME"] == r_home.as_posix()
        assert "\\" not in os.environ["R_HOME"]

    def test_ignores_r_home_without_a_usable_bin_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)
        monkeypatch.setattr(r_environment, "_candidate_r_homes", lambda: [tmp_path / "not-r"])

        assert r_environment.configure_r_dll_search() is None

    def test_result_is_memoised(self, monkeypatch, fake_r_home):
        _, bin_dir = fake_r_home
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)

        assert r_environment.configure_r_dll_search() == bin_dir
        monkeypatch.setattr(r_environment, "_candidate_r_homes", _unreachable)
        assert r_environment.configure_r_dll_search() == bin_dir


def _unreachable():
    raise AssertionError("detection should not run twice")


class TestEnsureAirgrAvailable:
    def test_passes_when_airgr_loads(self):
        robjects = _FakeRobjects(loadable=True)
        r_environment.ensure_airgr_available(robjects)
        assert len(robjects.calls) == 1  # no diagnosis query on the happy path

    def test_memoises_the_success(self):
        robjects = _FakeRobjects(loadable=True)
        r_environment.ensure_airgr_available(robjects)
        r_environment.ensure_airgr_available(robjects)
        assert len(robjects.calls) == 1

    def test_raises_with_embedded_r_diagnosis(self):
        robjects = _FakeRobjects(
            loadable=False,
            diagnosis="R.home(): C:/R | .libPaths(): C:/lib | error: unable to load stats.dll",
        )
        with pytest.raises(ModelExecutionError) as excinfo:
            r_environment.ensure_airgr_available(robjects)

        message = str(excinfo.value)
        assert "unable to load stats.dll" in message
        assert ".libPaths()" in message
        # The message must point at the real cause on Windows, not just
        # "package missing" — that misdiagnosis is what triggered the
        # doomed install.packages() call.
        assert "Rlapack.dll" in message

    def test_never_attempts_a_cran_install(self):
        """Installing mid-run is neither reproducible nor available offline."""
        robjects = _FakeRobjects(loadable=False)
        with pytest.raises(ModelExecutionError):
            r_environment.ensure_airgr_available(robjects)
        assert not any("install.packages" in call for call in robjects.calls)

    def test_raises_when_rpy2_is_absent(self):
        with pytest.raises(ModelExecutionError, match="rpy2"):
            r_environment.ensure_airgr_available(None)
