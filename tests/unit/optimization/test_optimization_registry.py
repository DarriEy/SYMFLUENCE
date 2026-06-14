# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for optimization-registry wiring (review item 17).

The generic Registry and the R facade are covered elsewhere; this covers the
optimization-specific behaviour: auto-discovery of model optimizers / parameter
managers, the calibration-target composite-key helper, and the objectives facade.
"""

from __future__ import annotations

import pytest

from symfluence.core.registries import R


@pytest.fixture
def temp_register():
    """Register entries into the shared R registries and remove them afterwards."""
    registered = []

    def _add(registry, key, value):
        registry.add(key, value)
        registered.append((registry, key))
        return value

    yield _add
    for registry, key in registered:
        try:
            registry.remove(key)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


# ---- auto-discovery ------------------------------------------------------


def test_model_optimizers_are_autodiscovered_on_import():
    import symfluence.optimization.model_optimizers  # noqa: F401 - triggers discovery

    assert {"SUMMA", "FUSE", "NGEN", "HYPE", "MESH"} <= set(R.optimizers.keys())


def test_parameter_managers_are_autodiscovered_on_import():
    import symfluence.optimization.parameter_managers  # noqa: F401 - triggers discovery

    assert {"SUMMA", "FUSE", "HYPE", "MESH"} <= set(R.parameter_managers.keys())


# ---- registration round-trip --------------------------------------------


def test_optimizer_registration_roundtrip_is_case_insensitive(temp_register):
    class FakeOptimizer:
        pass

    temp_register(R.optimizers, "FAKEMODEL", FakeOptimizer)
    assert R.optimizers.get("fakemodel") is FakeOptimizer  # normalized to upper
    assert "FAKEMODEL" in R.optimizers
    assert R.optimizers["FAKEMODEL"] is FakeOptimizer


def test_unknown_optimizer_get_returns_none():
    assert R.optimizers.get("NOT_A_REAL_MODEL_XYZ") is None


# ---- calibration-target composite key -----------------------------------


def test_calibration_target_composite_key(temp_register):
    class FakeTarget:
        pass

    temp_register(R.calibration_targets, "FAKEMODEL_STREAMFLOW", FakeTarget)
    assert R.get_calibration_target("fakemodel", "streamflow") is FakeTarget  # case-insensitive
    assert R.get_calibration_target("FAKEMODEL", "snow") is None  # unregistered target type


# ---- objectives facade ---------------------------------------------------


def test_objective_registry_get_and_list(temp_register):
    from symfluence.optimization.objectives.registry import ObjectiveRegistry

    class FakeObjective:
        def __init__(self, config, logger):
            self.config = config
            self.logger = logger

    temp_register(R.objectives, "FAKEOBJ", FakeObjective)
    instance = ObjectiveRegistry.get_objective("FAKEOBJ", {}, None)
    assert isinstance(instance, FakeObjective)
    assert ObjectiveRegistry.get_objective("DEFINITELY_UNKNOWN", {}, None) is None
    assert "FAKEOBJ" in ObjectiveRegistry.list_objectives()
