# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""A model that cannot import must still be buildable.

Build instructions are declared from inside a package's ``register()``, so a
package whose ``__init__.py`` raises declares nothing and vanishes from
``symfluence binary install`` — exactly when the user needs it most, because
building the binary is often what fixes the broken import (a missing compiled
dependency, say).

``_recover_build_instructions_from_failed_plugins`` closes that gap by stubbing
the single failing leaf package and importing its ``build_instructions`` module
directly. These tests prove it works against a package that genuinely fails to
import, and that it stays narrow: no stub survives, and a healthy install does
no recovery at all.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from symfluence.cli import external_tools_config as etc
from symfluence.core.registries import R

pytestmark = [pytest.mark.unit]


@pytest.fixture
def broken_plugin(tmp_path, monkeypatch):
    """A package whose __init__ raises but whose build_instructions is fine."""
    pkg = tmp_path / "brokenmodel"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise ImportError('simulated missing compiled dependency')\n",
        encoding="utf-8",
    )
    # Uses a relative import, as the in-tree build_instructions modules do —
    # this only resolves if the stub carries a correct __path__.
    (pkg / "_commands.py").write_text(
        "BUILD = ['make brokenmodel']\n", encoding="utf-8"
    )
    (pkg / "build_instructions.py").write_text(
        textwrap.dedent(
            """
            from symfluence.core.registries import R

            from ._commands import BUILD

            R.build_instructions.add('brokenmodel', {
                'description': 'Recovered tool',
                'build_commands': BUILD,
                'order': 99,
            })
            """
        ).strip() + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    monkeypatch.setattr(
        etc, "_register_infrastructure_tools", lambda: None, raising=False
    )
    monkeypatch.setattr(
        "symfluence.core._bootstrap.failed_plugin_entry_points",
        lambda: (("brokenmodel", "brokenmodel:register"),),
    )

    yield "brokenmodel"

    for name in [m for m in sys.modules if m.startswith("brokenmodel")]:
        sys.modules.pop(name, None)
    try:
        R.build_instructions.remove("brokenmodel")
    except (KeyError, RuntimeError):
        pass


def test_package_import_genuinely_fails(broken_plugin):
    """Guard the fixture itself: the package must really be broken."""
    with pytest.raises(ImportError):
        __import__(broken_plugin)


def test_recovery_registers_build_instructions(broken_plugin):
    assert R.build_instructions.get(broken_plugin) is None

    etc._recover_build_instructions_from_failed_plugins()

    recovered = R.build_instructions.get(broken_plugin)
    assert recovered is not None, (
        "a package that cannot import lost its build instructions, so its "
        "binary cannot be built — the failure this recovery exists to prevent"
    )
    assert recovered["build_commands"] == ["make brokenmodel"]


def test_recovery_leaves_no_stub_behind(broken_plugin):
    etc._recover_build_instructions_from_failed_plugins()
    assert broken_plugin not in sys.modules, (
        "the stub used to bypass the broken __init__ leaked into sys.modules; "
        "a later real import would silently get an empty module"
    )


def test_recovery_is_idempotent(broken_plugin):
    etc._recover_build_instructions_from_failed_plugins()
    etc._recover_build_instructions_from_failed_plugins()
    assert R.build_instructions.get(broken_plugin) is not None


def test_no_recovery_attempted_when_nothing_failed(monkeypatch):
    """A healthy install must not run the stubbing path at all."""
    monkeypatch.setattr(
        "symfluence.core._bootstrap.failed_plugin_entry_points", lambda: ()
    )
    before = set(sys.modules)
    etc._recover_build_instructions_from_failed_plugins()
    assert set(sys.modules) == before
