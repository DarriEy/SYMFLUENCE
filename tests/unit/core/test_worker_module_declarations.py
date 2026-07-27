# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration workers resolve by declaration, not by filesystem layout.

Workers are the one calibration component with no discovery pass of its own:
they register as a side effect of importing the owning package's
``calibration/worker.py``. The coupled optimizer used to reach that module by
building ``symfluence.models.<name>.calibration.worker`` with an f-string —
a runtime upward edge from ``core`` into the models layer that the AST layering
guard could not see, and that no external plugin could ever satisfy.

Packages now declare the module instead (``R.workers.add_module``) and core
drains the declarations. These tests are what make removing that fallback safe:
if an in-tree package ever stops being declared, the coupled optimizer would
silently lose it, so the drift fails the build here instead.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from symfluence.core.registries import R

pytestmark = [pytest.mark.unit]

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
_MODELS_DIR = _SOURCE_ROOT / "symfluence" / "models"


def _packages_shipping_a_worker() -> set[str]:
    return {
        path.relative_to(_MODELS_DIR).parts[0]
        for path in _MODELS_DIR.glob("*/calibration/worker.py")
    }


def test_every_in_tree_worker_module_is_declared():
    """No in-tree package may ship a worker the registry cannot reach."""
    import symfluence.models  # noqa: F401 — import declares the modules

    declared = {
        module for module in R.workers.declared_modules()
        if module.endswith(".calibration.worker")
    }
    declared_packages = {module.split(".")[-3] for module in declared}

    missing = _packages_shipping_a_worker() - declared_packages
    assert not missing, (
        f"packages ship calibration/worker.py but never declare it: {sorted(missing)}. "
        "The coupled optimizer resolves participants through R.workers.load_modules(), "
        "so an undeclared worker is unreachable."
    )


def test_declared_worker_modules_all_import():
    """A declaration that cannot be imported is worse than no declaration.

    ``load_modules()`` swallows ImportError by design (a consumer must tolerate
    an optional dependency being absent), which would turn a typo in a declared
    path into a worker that silently never registers.
    """
    import importlib

    import symfluence.models  # noqa: F401

    for module in R.workers.declared_modules():
        if not module.endswith(".calibration.worker"):
            continue
        importlib.import_module(module)


def test_draining_declarations_is_idempotent():
    before = sorted(R.workers.keys())
    R.workers.load_modules()
    R.workers.load_modules()
    assert sorted(R.workers.keys()) == before


# ---------------------------------------------------------------------------
# COUPLED registration must not depend on the models distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "registry_name,expected",
    [
        ("optimizers", "symfluence.core.calibration.coupled.optimizer"),
        ("workers", "symfluence.core.calibration.coupled.worker"),
        ("parameter_managers", "symfluence.core.calibration.coupled.parameter_manager"),
    ],
)
def test_coupled_resolves_from_core(registry_name, expected):
    component = getattr(R, registry_name).get("COUPLED")
    assert component is not None, f"COUPLED missing from R.{registry_name}"
    assert component.__module__ == expected


_COUPLED_ABSENT_SMOKE = r'''
import importlib.abc
import importlib.metadata
import sys

sys.path.insert(0, sys.argv[1])
importlib.metadata.entry_points = lambda **kwargs: []


class _ModelsBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'symfluence.models' or fullname.startswith('symfluence.models.'):
            raise ImportError(f'blocked absent model package: {fullname}')
        return None


sys.meta_path.insert(0, _ModelsBlocker())

import symfluence  # noqa: F401
from symfluence.core.registries import R

# Reading any of the three registries fires the deferred seeder.
for name in ('optimizers', 'workers', 'parameter_managers'):
    component = getattr(R, name).get('COUPLED')
    assert component is not None, f'COUPLED absent from R.{name} without the models package'
    assert component.__module__.startswith('symfluence.core.calibration.coupled')

print('COUPLED-WITHOUT-MODELS-OK')
'''


def test_coupled_registers_without_the_models_package():
    """An install carrying only external model plugins still gets COUPLED.

    Registration used to fire only when ``optimization._autodiscover``
    pkgutil-scanned ``symfluence.models.*`` and happened to import the
    back-compat shim — so the framework held the code while
    ``R.optimizers.get('COUPLED')`` returned None and ``optimization_manager``
    routed to it anyway.
    """
    result = subprocess.run(
        [sys.executable, "-c", _COUPLED_ABSENT_SMOKE, str(_SOURCE_ROOT)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"COUPLED did not register without the models package.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "COUPLED-WITHOUT-MODELS-OK" in result.stdout
