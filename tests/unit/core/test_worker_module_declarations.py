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


#: In-tree packages shipping ``calibration/worker.py`` that deliberately
#: register no worker under their own key. Enumerated so the import test below
#: can assert every OTHER declared module actually reaches the registry —
#: without this, "the module imported" would be all it proved.
_WORKERS_WITHOUT_A_MODEL_KEY = {
    # Self-training models: calibration is gradient descent inside the run
    # step, so their worker module carries helpers but registers no worker.
    "gnn", "lstm",
    # Registers the COUPLED_GW pipeline, not a 'MODFLOW' worker.
    "modflow",
}


def _packages_shipping_a_worker() -> set[str]:
    return {
        path.relative_to(_MODELS_DIR).parts[0]
        for path in _MODELS_DIR.glob("*/calibration/worker.py")
    }


def _declared_in_tree_worker_packages() -> set[str]:
    prefix = "symfluence.models."
    return {
        module[len(prefix):].split(".")[0]
        for module in R.workers.declared_modules()
        if module.startswith(prefix) and module.endswith(".calibration.worker")
    }


def test_every_in_tree_worker_module_is_declared():
    """No in-tree package may ship a worker the registry cannot reach.

    Exact equality, both directions: a shipped-but-undeclared worker is
    unreachable through ``R.workers.load_modules()``, and a declared-but-absent
    module is a path that ``load_modules`` will silently skip on ImportError.
    """
    import symfluence.models  # noqa: F401 — import declares the modules

    shipped = _packages_shipping_a_worker()
    assert shipped, (
        "no in-tree package ships calibration/worker.py — the source layout "
        "moved and every assertion in this file is now vacuous"
    )
    assert _declared_in_tree_worker_packages() == shipped, (
        "declared in-tree worker modules disagree with the packages that ship "
        f"one: declared-only "
        f"{sorted(_declared_in_tree_worker_packages() - shipped)}, "
        f"shipped-only {sorted(shipped - _declared_in_tree_worker_packages())}. "
        "The coupled optimizer resolves participants through "
        "R.workers.load_modules(), so an undeclared worker is unreachable."
    )


def test_declared_worker_modules_all_import_and_register():
    """A declaration that cannot be imported is worse than no declaration.

    ``load_modules()`` swallows ImportError by design (a consumer must tolerate
    an optional dependency being absent), which would turn a typo in a declared
    path into a worker that silently never registers. Importing each module is
    only half the check — the previous version of this test made no assertion
    at all, so it also passed when ``declared_modules()`` was empty and when a
    module imported without registering anything.
    """
    import importlib

    import symfluence.models  # noqa: F401

    declared = _declared_in_tree_worker_packages()
    assert declared, "no in-tree worker modules are declared"

    unregistered = []
    for package in sorted(declared):
        importlib.import_module(f"symfluence.models.{package}.calibration.worker")
        if package in _WORKERS_WITHOUT_A_MODEL_KEY:
            continue
        if R.workers.get(package.upper()) is None:
            unregistered.append(package)

    assert not unregistered, (
        f"these worker modules import but register no worker under their own "
        f"model key: {unregistered}. Either the @R.workers.add decorator was "
        "lost, or the package belongs in _WORKERS_WITHOUT_A_MODEL_KEY."
    )
    # The exemptions must stay exemptions, or the list silently rots.
    still_exempt = sorted(
        package for package in _WORKERS_WITHOUT_A_MODEL_KEY
        if package in declared and R.workers.get(package.upper()) is not None
    )
    assert not still_exempt, (
        f"{still_exempt} now register a worker under their own key — drop them "
        "from _WORKERS_WITHOUT_A_MODEL_KEY"
    )


def test_load_modules_actually_imports_a_declared_module(tmp_path, monkeypatch):
    """``load_modules()`` is the seam; a no-op implementation must fail here.

    Asserting only that the live registry's keys are unchanged across two
    ``load_modules()`` calls (which is what this test used to do) passes just as
    happily when the method body is ``pass``. A registry nobody else touches,
    with a module nobody else imports, is what makes the import observable.
    """
    from symfluence.core.registry import Registry

    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "symfluence_probe_declared_worker.py").write_text(
        "LOADED = True\n", encoding="utf-8"
    )
    monkeypatch.delitem(
        sys.modules, "symfluence_probe_declared_worker", raising=False
    )

    registry: Registry = Registry("probe_workers")
    registry.add_module("symfluence_probe_declared_worker")
    assert registry.declared_modules() == ("symfluence_probe_declared_worker",)
    assert "symfluence_probe_declared_worker" not in sys.modules

    registry.load_modules()
    assert "symfluence_probe_declared_worker" in sys.modules, (
        "load_modules() did not import the declared module"
    )
    assert sys.modules["symfluence_probe_declared_worker"].LOADED is True

    # Draining again is a no-op rather than a re-import or a re-registration.
    registry.load_modules()
    assert registry.declared_modules() == ("symfluence_probe_declared_worker",)


def test_draining_declarations_is_idempotent():
    """Draining the LIVE registry adds nothing beyond what is already there."""
    R.workers.load_modules()
    before = sorted(R.workers.keys())
    assert before, "R.workers is empty after draining its declarations"
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
