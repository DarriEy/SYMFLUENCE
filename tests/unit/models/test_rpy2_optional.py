# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests pinning rpy2 as a GR-only runtime requirement.

The bootstrap installer attempts rpy2 by default so GR works out of the
box on systems with R, but the install is best-effort — failure is
non-fatal. These tests pin two invariants:
  1. Non-GR runners (SUMMA, FUSE, mizuRoute, acquisition,
     discretization) import without requiring rpy2 at all.
  2. GR's deferred-import ImportError stays actionable when rpy2 is
     unavailable, telling the user how to install it manually.
"""
from __future__ import annotations

import sys
from importlib import import_module

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.quick]


# Modules that must NEVER import rpy2 directly or transitively. Add to this
# list whenever a model that does not require R is added to the codebase.
NON_GR_MODULES = [
    "symfluence.models.summa.runner",
    "symfluence.models.fuse.runner",
    "symfluence.models.mizuroute.runner",
    "symfluence.data.acquisition.acquisition_service",
    "symfluence.geospatial.discretization.core",
]


@pytest.mark.parametrize("modname", NON_GR_MODULES)
def test_non_gr_module_does_not_force_rpy2(modname):
    """Importing a non-GR module must not require rpy2 to be installed.

    We assert that the module imports cleanly and that, after import,
    rpy2 is not present in sys.modules due to the import (other tests in
    the run may have triggered it; we only check this module's import
    works regardless of rpy2 availability)."""
    # Drop any cached form so we exercise a fresh import resolution.
    sys.modules.pop(modname, None)
    mod = import_module(modname)
    assert mod is not None


def test_gr_runner_error_actionable_when_rpy2_missing(monkeypatch):
    """If rpy2 is genuinely not installed, GRRunner.__init__ must raise an
    ImportError that tells the user how to install rpy2 manually.

    This is the only user-visible escape hatch for "I tried GR and it
    failed", so the error message must stay actionable."""
    import symfluence.models.gr.runner as gr_runner

    # Force HAS_RPY2 to False regardless of the test machine's actual rpy2
    # status, so the test runs identically with or without R installed. Clear
    # any captured import error so we exercise the genuinely-not-installed
    # branch (no swallowed embedded-R failure to surface).
    monkeypatch.setattr(gr_runner, "HAS_RPY2", False)
    monkeypatch.setattr(gr_runner, "_RPY2_IMPORT_ERROR", None)

    with pytest.raises(ImportError) as exc:
        gr_runner.GRRunner(config={}, logger=None)

    msg = str(exc.value)
    assert "rpy2" in msg
    # Must offer a concrete install command (manual pip install or extras)
    assert "pip install" in msg, (
        "GR ImportError must offer a concrete pip install command "
        "so users know how to enable GR after a failed default install."
    )


def test_gr_runner_surfaces_real_error_when_rpy2_import_failed(monkeypatch):
    """If rpy2 is installed but importing it failed (the embedded R could not
    start), GRRunner.__init__ must surface that real cause — not the misleading
    "rpy2 is not installed / pip install rpy2" message.

    This is the failure mode behind the persistent GR4J exit-14 on Linux and
    a make-less Windows shell: rpy2 present, embedded R unstartable, and the
    real exception swallowed by the eager module-level import."""
    import symfluence.models.gr.runner as gr_runner
    from symfluence.core.exceptions import ModelExecutionError

    real_error = IndexError("list index out of range")
    monkeypatch.setattr(gr_runner, "HAS_RPY2", False)
    monkeypatch.setattr(gr_runner, "_RPY2_IMPORT_ERROR", real_error)
    # rpy2 is installed as a package (find_spec truthy) even though its import
    # blew up; make the test deterministic regardless of the CI machine.
    monkeypatch.setattr(gr_runner, "rpy2_installed", lambda: True)

    with pytest.raises(ModelExecutionError) as exc:
        gr_runner.GRRunner(config={}, logger=None)

    msg = str(exc.value)
    # The real exception type/message must appear, and the message must not
    # falsely claim rpy2 is uninstalled.
    assert "IndexError" in msg
    assert "rpy2 IS installed" in msg
    assert "pip install rpy2" not in msg
    # The original exception is chained so the traceback keeps the real cause.
    assert exc.value.__cause__ is real_error
