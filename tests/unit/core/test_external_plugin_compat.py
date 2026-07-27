# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""External model plugins keep working against the promoted contract tier.

The JAX model packages (jhbv, jsacsma, ...) are independent PyPI packages that
subclass the calibration and adapter bases. After the phase-0/step-0
promotions those bases live in core (``core.calibration``, ``core.modeling``)
with shims at the historical paths — these tests pin that an installed
external plugin still registers and that its components resolve to classes
subclassing the canonical (core) bases.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jhbv", reason="requires the jax extra (pip install 'symfluence[jax]')")

import symfluence  # noqa: E402,F401 — triggers plugin registration
from symfluence.core.calibration.workers.inmemory_worker import (  # noqa: E402
    InMemoryModelWorker,
)
from symfluence.core.registries import R  # noqa: E402


@pytest.mark.unit
def test_jax_plugin_registers_components():
    for namespace in (R.runners, R.workers, R.parameter_managers, R.config_schemas):
        assert namespace.get("HBV") is not None, f"HBV missing from {namespace}"


@pytest.mark.unit
def test_jax_worker_subclasses_canonical_core_base():
    worker_cls = R.workers.get("HBV")
    assert issubclass(worker_cls, InMemoryModelWorker), (
        f"{worker_cls} does not subclass the canonical core InMemoryModelWorker — "
        "an old-path shim is no longer aliasing the same class object"
    )


#: Models served by the installed external JAX plugin packages. If a plugin
#: fails to register (e.g. because a name it imports vanished from the contract
#: surface), symfluence logs an incompatibility warning and silently drops the
#: model — so absence here is the observable symptom of a contract break.
_EXTERNAL_PLUGIN_MODELS = ["HBV", "HECHMS", "SACSMA", "TOPMODEL", "XINANJIANG"]


@pytest.mark.unit
@pytest.mark.parametrize("model", _EXTERNAL_PLUGIN_MODELS)
def test_installed_plugin_registers_its_runner(model):
    """Every installed JAX plugin completes registration.

    Regression guard: removing a public name from the ``models`` contract
    surface makes the importing plugin's ``register()`` raise ImportError,
    which the plugin loader turns into an "incompatible with this SYMFLUENCE
    version" warning and an absent model — a MAJOR break disguised as a log
    line. This test turns it into a failing assertion.
    """
    assert R.runners.get(model) is not None, (
        f"{model} is not registered — its external plugin package failed to "
        "register. Check for an ImportError against a name that was removed or "
        "renamed on the core contract surface."
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
    )

    assert isinstance(MODEL_SPATIAL_CAPABILITIES, dict)
    assert MODEL_SPATIAL_CAPABILITIES, "capability mapping is empty"

    # Values expose the four documented fields.
    capability = MODEL_SPATIAL_CAPABILITIES["SUMMA"]
    for attribute in ("supported_modes", "default_mode",
                      "requires_routing", "warning_message"):
        assert hasattr(capability, attribute), attribute

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
    """The shim paths external packages import must be the SAME class objects."""
    from symfluence.core.calibration.optimizers.base_model_optimizer import (
        BaseModelOptimizer as core_bmo,
    )
    from symfluence.core.calibration.optimizers.base_model_optimizer import (
        BaseModelOptimizer as legacy_bmo,
    )
    from symfluence.core.modeling.base import BaseModelRunner as core_bmr
    from symfluence.core.modeling.base import BaseModelRunner as legacy_bmr

    assert legacy_bmr is core_bmr
    assert legacy_bmo is core_bmo
