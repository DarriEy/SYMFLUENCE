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

from pathlib import Path

import symfluence.models
from symfluence.core.registries import R
from symfluence.models import model_packages_with


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


def test_forcing_adapter_modules_are_declared_into_the_registry():
    """Every in-tree forcing_adapter.py is declared, so core never globs for it."""
    declared = set(R.forcing_adapters.declared_modules())
    for name in model_packages_with("forcing_adapter"):
        assert f"symfluence.models.{name}.forcing_adapter" in declared


def test_preset_modules_are_declared_into_the_registry():
    """Every in-tree init_preset.py is declared, so the CLI never globs for it."""
    declared = set(R.presets.declared_modules())
    for name in model_packages_with("init_preset"):
        assert f"symfluence.models.{name}.init_preset" in declared


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

    A plain source scan, so prose in those modules should say "the models
    package" rather than spelling the dotted name.
    """
    import inspect

    from symfluence.cli import preset_registry
    from symfluence.core.modeling.adapters import adapter_registry

    for module in (adapter_registry, preset_registry):
        assert "symfluence.models" not in inspect.getsource(module), (
            f"{module.__name__} still reaches into the models package"
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
