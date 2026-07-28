# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""External model plugins keep working against the promoted contract tier.

The JAX model packages (jhbv, jsacsma, ...) are independent PyPI packages that
subclass the calibration and adapter bases. After the phase-0/step-0
promotions those bases live in core (``core.calibration``, ``core.modeling``)
with shims at the historical paths — these tests pin that an installed
external plugin still registers and that its components resolve to classes
subclassing the canonical (core) bases.

Two of the guards here (the contract-surface tests at the bottom) do NOT need a
plugin installed and always run. The rest need the ``jax`` extra; when it is
absent this module says so out loud rather than skipping quietly, because a
silently inert external-plugin guard is the worst outcome for a file whose
whole job is noticing that plugins broke.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import warnings

import pytest

import symfluence  # noqa: F401 — triggers plugin registration
from symfluence.core.calibration.workers.inmemory_worker import InMemoryModelWorker
from symfluence.core.registries import R

#: Model key -> the external distribution that registers it. All ship together
#: in the ``jax`` extra (see pyproject), so one missing module means the extra
#: is not installed rather than one plugin being individually broken.
_EXTERNAL_PLUGIN_PACKAGES = {
    "HBV": "jhbv",
    "HECHMS": "jhechms",
    "SACSMA": "jsacsma",
    "TOPMODEL": "jtopmodel",
    "XINANJIANG": "jxaj",
}

_MISSING_PLUGIN_PACKAGES = sorted(
    module for module in set(_EXTERNAL_PLUGIN_PACKAGES.values())
    if importlib.util.find_spec(module) is None
)

_INERT_REASON = (
    "the jax extra is not installed "
    f"(missing: {', '.join(_MISSING_PLUGIN_PACKAGES) or 'none'}); the "
    "external-plugin compatibility guard in "
    "tests/unit/core/test_external_plugin_compat.py is INERT in this run. "
    "A contract break against an external model package will NOT be caught "
    "here. Install with: pip install 'symfluence[jax]'"
)

if _MISSING_PLUGIN_PACKAGES:
    # Surfaced in pytest's warnings summary so an inert guard is visible at the
    # end of the run, not buried in a skip count.
    warnings.warn(_INERT_REASON, UserWarning, stacklevel=1)

requires_jax_extra = pytest.mark.skipif(
    bool(_MISSING_PLUGIN_PACKAGES), reason=_INERT_REASON
)


# ---------------------------------------------------------------------------
# Guards that always run — no plugin installation required
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_installed_plugin_was_dropped_during_discovery():
    """Every installed out-of-tree plugin completed registration.

    This is the one guard in this file that cannot go inert: it derives the
    plugin set from the entry points actually installed rather than from a
    hardcoded list, so it has teeth for whatever plugins the environment has —
    including plugins this file has never heard of — and asserts nothing when
    there are genuinely none.

    ``_discover_plugins`` turns a plugin whose ``register()`` raised into a log
    line and an absent model. ``failed_plugin_entry_points()`` is the record of
    that, and a non-empty external entry here is a MAJOR break disguised as a
    warning.
    """
    from importlib.metadata import entry_points

    from symfluence.core._bootstrap import (
        PLUGIN_ENTRY_POINT_GROUP,
        failed_plugin_entry_points,
    )

    installed_external = {
        ep.name for ep in entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        if not ep.value.startswith("symfluence.")
    }
    dropped = sorted(
        (name, value) for name, value in failed_plugin_entry_points()
        if name in installed_external
    )
    assert not dropped, (
        f"installed external plugins failed to register: {dropped}. Check for "
        "an ImportError against a name that was removed or renamed on the core "
        "contract surface."
    )


@pytest.mark.unit
def test_spatial_capabilities_public_name_survives_registry_backing():
    """``MODEL_SPATIAL_CAPABILITIES`` stays importable with mapping semantics.

    It is imported directly by external model packages. The registration seam
    (``register_model_spatial_capability``) is built BEHIND it: the name must
    stay a live, dict-like view of the same data the seam serves, never a
    snapshot that goes stale once a package registers.
    """
    from symfluence.core.modeling import spatial_modes
    from symfluence.core.modeling.spatial_modes import (
        MODEL_SPATIAL_CAPABILITIES,
        ModelSpatialCapability,
        SpatialMode,
        get_model_capabilities,
        registered_spatial_capability_models,
    )

    # The exported name must BE the registry, not a copy of it. A snapshot
    # would still satisfy ``isinstance(..., dict)``, which is why that is not
    # what is asserted: the keys have to agree with the seam's own read side.
    assert set(MODEL_SPATIAL_CAPABILITIES) == set(
        registered_spatial_capability_models()
    ), "the legacy mapping and the seam's read side disagree — one is a snapshot"
    assert MODEL_SPATIAL_CAPABILITIES, "capability mapping is empty"
    # Mapping operations third-party code uses against it, over live data.
    assert "SUMMA" in MODEL_SPATIAL_CAPABILITIES
    assert MODEL_SPATIAL_CAPABILITIES.get("SUMMA") is (
        MODEL_SPATIAL_CAPABILITIES["SUMMA"]
    )
    assert len(MODEL_SPATIAL_CAPABILITIES) == len(dict(MODEL_SPATIAL_CAPABILITIES))

    # The record's four documented fields, by name and by resolved value —
    # ``hasattr`` on a dataclass instance cannot fail for a declared field, so
    # the field list itself is what is pinned.
    assert [f.name for f in dataclasses.fields(ModelSpatialCapability)] == [
        "supported_modes", "default_mode", "requires_routing", "warning_message",
    ]
    capability = MODEL_SPATIAL_CAPABILITIES["SUMMA"]
    assert capability.supported_modes and all(
        isinstance(mode, SpatialMode) for mode in capability.supported_modes
    )
    assert isinstance(capability.default_mode, SpatialMode)
    assert isinstance(capability.requires_routing, dict)

    # Live in both directions: a seam registration is visible through the
    # legacy name, and a legacy write is visible through the seam.
    probe = ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
    )
    try:
        spatial_modes.register_model_spatial_capability("_PROBE_SEAM", probe)
        assert MODEL_SPATIAL_CAPABILITIES["_PROBE_SEAM"] is probe

        MODEL_SPATIAL_CAPABILITIES["_PROBE_LEGACY"] = probe
        assert get_model_capabilities("_probe_legacy") is probe
    finally:
        MODEL_SPATIAL_CAPABILITIES.pop("_PROBE_SEAM", None)
        MODEL_SPATIAL_CAPABILITIES.pop("_PROBE_LEGACY", None)


@pytest.mark.unit
def test_historical_base_import_paths_alias_core_classes():
    """The shim paths external packages import must be the SAME class objects.

    Deleting ``symfluence/models/base/__init__.py`` or
    ``symfluence/optimization/optimizers/base_model_optimizer.py`` breaks every
    installed plugin built against the pre-promotion layout. Both sides are
    imported here — the LEGACY path an external package writes, and the
    CANONICAL path core now defines the class at — so the shim's existence is
    what the assertion depends on. (An earlier revision imported the canonical
    path twice and asserted ``X is X``, which held with both shims deleted.)
    """
    from symfluence.core.calibration.optimizers.base_model_optimizer import (
        BaseModelOptimizer as core_bmo,
    )
    from symfluence.core.modeling.base import BaseModelRunner as core_bmr
    from symfluence.models.base import BaseModelRunner as legacy_bmr
    from symfluence.optimization.optimizers.base_model_optimizer import (
        BaseModelOptimizer as legacy_bmo,
    )

    assert legacy_bmr is core_bmr, (
        "symfluence.models.base.BaseModelRunner is no longer the core class; "
        "an external plugin subclassing it would not be recognised"
    )
    assert legacy_bmo is core_bmo, (
        "symfluence.optimization.optimizers.base_model_optimizer."
        "BaseModelOptimizer is no longer the core class"
    )


# ---------------------------------------------------------------------------
# Guards that need the jax extra installed
# ---------------------------------------------------------------------------

@pytest.mark.unit
@requires_jax_extra
def test_jax_plugin_registers_components():
    for namespace in (R.runners, R.workers, R.parameter_managers, R.config_schemas):
        assert namespace.get("HBV") is not None, f"HBV missing from {namespace}"


@pytest.mark.unit
@requires_jax_extra
def test_jax_worker_subclasses_canonical_core_base():
    worker_cls = R.workers.get("HBV")
    assert issubclass(worker_cls, InMemoryModelWorker), (
        f"{worker_cls} does not subclass the canonical core InMemoryModelWorker — "
        "an old-path shim is no longer aliasing the same class object"
    )


@pytest.mark.unit
@requires_jax_extra
@pytest.mark.parametrize("model", sorted(_EXTERNAL_PLUGIN_PACKAGES))
def test_installed_plugin_registers_its_runner(model):
    """Every installed JAX plugin completes registration.

    Regression guard: removing a public name from the ``models`` contract
    surface makes the importing plugin's ``register()`` raise ImportError,
    which the plugin loader turns into an "incompatible with this SYMFLUENCE
    version" warning and an absent model — a MAJOR break disguised as a log
    line. This test turns it into a failing assertion.
    """
    assert R.runners.get(model) is not None, (
        f"{model} is not registered — its external plugin package "
        f"({_EXTERNAL_PLUGIN_PACKAGES[model]}) failed to register. Check for an "
        "ImportError against a name that was removed or renamed on the core "
        "contract surface."
    )
