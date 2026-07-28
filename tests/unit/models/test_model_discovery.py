# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Discovery of the per-model side-modules that register optional capabilities.

Forcing adapters and ``symfluence init`` presets register through a decorator
that only fires when ``<model>/forcing_adapter.py`` / ``<model>/init_preset.py``
is imported.  The framework used to find those modules by globbing the models
source tree from ``core``/``cli`` — an upward dependency on a distribution that
is being extracted, and one that structurally could not see external plugin
packages.

Now the models distribution *declares* the modules it owns into the registries
(``Registry.add_module``) when it is imported, and the framework only drains the
declarations (``Registry.load_modules``).  ``model_packages_with`` survives as
the distribution's own introspection helper feeding that declaration (RTI review
item 18: it replaced the hardcoded SupportedModels.WITH_* lists, which drifted).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import symfluence.models
from symfluence.core.registries import R
from symfluence.models import model_packages_with


def compiled_string_constants(source: str, filename: str) -> set[str]:
    """Every string constant in *source*, recursively through code objects.

    A source-text grep for a dotted path is defeated by writing it as
    ``"symfluence." + "models"``. Compiling first defeats that back: CPython's
    peephole optimiser folds adjacent literal concatenation into a single
    constant, so the reassembled path shows up here.
    """
    def walk(code: types.CodeType) -> set[str]:
        found: set[str] = set()
        for constant in code.co_consts:
            if isinstance(constant, str):
                found.add(constant)
            elif isinstance(constant, types.CodeType):
                found |= walk(constant)
        return found

    return walk(compile(source, filename, "exec"))


def test_forcing_adapter_discovery_includes_watflood():
    """watflood has a forcing_adapter.py but was missing from the old list (drift fix)."""
    pkgs = model_packages_with("forcing_adapter")
    assert {"summa", "fuse", "watflood"} <= set(pkgs)
    assert pkgs == tuple(sorted(pkgs))


def test_preset_discovery_matches_disk():
    """Preset discovery returns only models that actually ship init_preset.py."""
    pkgs = model_packages_with("init_preset")
    assert {"summa", "fuse"} <= set(pkgs)
    models_dir = Path(symfluence.models.__file__).resolve().parent
    for name in pkgs:
        assert (models_dir / name / "init_preset.py").is_file()


def test_unknown_submodule_returns_empty():
    assert model_packages_with("definitely_not_a_submodule") == ()


# ----------------------------------------------------------------------
# Declaration: importing the models distribution declares its capability
# modules into the registries the framework reads.
# ----------------------------------------------------------------------


_MODELS_DIR = Path(symfluence.models.__file__).resolve().parent


def _packages_on_disk_with(submodule: str) -> set[str]:
    """The packages shipping ``<submodule>.py``, globbed by this test itself.

    Deliberately NOT ``model_packages_with`` — that is the call whose output
    produced the declarations under test, so comparing the two only proves the
    declaration loop agrees with itself. Make ``model_packages_with`` return
    ``()`` and both sides go empty: the assertions below used to pass while the
    framework saw no forcing adapters and no presets at all.
    """
    return {
        path.relative_to(_MODELS_DIR).parts[0]
        for path in _MODELS_DIR.glob(f"*/{submodule}.py")
    }


def _declared_packages(registry, submodule: str) -> set[str]:
    prefix = "symfluence.models."
    suffix = f".{submodule}"
    return {
        module[len(prefix):-len(suffix)]
        for module in registry.declared_modules()
        if module.startswith(prefix) and module.endswith(suffix)
    }


@pytest.mark.parametrize("submodule,registry_name", [
    ("forcing_adapter", "forcing_adapters"),
    ("init_preset", "presets"),
])
def test_capability_modules_are_declared_into_the_registry(submodule, registry_name):
    """Every in-tree side-module is declared, so core/CLI never glob for it.

    Exact equality against an independent glob, and a non-empty precondition:
    an empty declaration set is the failure this guards, not a pass.
    """
    on_disk = _packages_on_disk_with(submodule)
    assert on_disk, (
        f"no in-tree package ships {submodule}.py — the source layout moved "
        "and this test is now vacuous"
    )
    declared = _declared_packages(getattr(R, registry_name), submodule)
    assert declared == on_disk, (
        f"declared {submodule} modules disagree with the packages that ship "
        f"one: declared-only {sorted(declared - on_disk)}, shipped-only "
        f"{sorted(on_disk - declared)}"
    )


def test_model_packages_with_agrees_with_the_disk():
    """The introspection helper feeding the declarations is itself correct."""
    for submodule in ("forcing_adapter", "init_preset", "calibration/worker"):
        expected = {
            path.relative_to(_MODELS_DIR).parts[0]
            for path in _MODELS_DIR.glob(f"*/{submodule}.py")
        }
        assert expected, f"nothing ships {submodule}.py"
        assert set(model_packages_with(submodule)) == expected, submodule


def test_declarations_are_idempotent():
    """Re-running the declaration (a model declaring for itself) is a no-op."""
    before = R.presets.declared_modules()
    symfluence.models._declare_capability_modules()
    assert R.presets.declared_modules() == before


def test_framework_discovery_does_not_import_the_models_package():
    """The two facades must not name the models package any more.

    This is the seam the extraction depends on: both consumers read the
    registry, so the framework keeps working when the models distribution is
    not installed at all (tests/conformance/test_models_absent.py).

    Two scans, because a plain ``"symfluence.models" not in source`` grep is
    defeated by ``"symfluence." + "models"``:

    * the source text, so prose in those modules keeps saying "the models
      package" rather than spelling the dotted name;
    * every string constant in the COMPILED module, recursively through nested
      code objects. CPython folds adjacent literal concatenation at compile
      time, so a split literal reassembles into one constant here and is
      caught.
    """
    import inspect

    from symfluence.cli import preset_registry
    from symfluence.core.modeling.adapters import adapter_registry

    for module in (adapter_registry, preset_registry):
        source = inspect.getsource(module)
        assert "symfluence.models" not in source, (
            f"{module.__name__} still reaches into the models package"
        )
        offenders = sorted(
            constant for constant in compiled_string_constants(source, module.__file__)
            if "symfluence.models" in constant
        )
        assert not offenders, (
            f"{module.__name__} builds a models-package path from string "
            f"fragments: {offenders}"
        )


def test_declared_adapter_modules_register_their_adapters():
    """Draining the declarations yields the adapters the glob loop used to."""
    from symfluence.core.modeling.adapters.adapter_registry import ForcingAdapterRegistry

    registered = set(ForcingAdapterRegistry.get_registered_models())
    assert {"SUMMA", "FUSE", "GR", "HYPE", "MESH", "NGEN", "RHESSYS"} <= registered
    # watflood ships a forcing_adapter.py that registers nothing; its absence is
    # pre-existing behaviour, pinned so that "fixing" it is a conscious change.
    assert "WATFLOOD" not in registered


def test_declared_preset_modules_register_their_presets():
    """Draining the declarations yields the presets the glob loop used to."""
    from symfluence.cli.preset_registry import PresetRegistry

    assert {"summa-basic", "bow-river", "fuse-basic", "fuse-provo"} <= set(
        PresetRegistry.list_presets()
    )
