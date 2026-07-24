# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The parameter-bounds extension seam (service-decomposition prep).

External model packages must be able to contribute calibration bounds without
editing core: ``register_model_bounds()`` + ``get_model_bounds()``.
"""
from __future__ import annotations

import pytest

from symfluence.core.calibration.parameters import (
    ParameterInfo,
    get_model_bounds,
    register_model_bounds,
    registered_bound_models,
)
from symfluence.core.calibration.parameters import parameter_bounds_registry as pbr


@pytest.fixture(autouse=True)
def _clean_seam():
    """Keep seam state test-local."""
    saved = (dict(pbr._EXTENSION_PARAMS), dict(pbr._MODEL_PARAM_NAMES),
             dict(pbr._MODEL_NAME_PREFIXES))
    yield
    pbr._EXTENSION_PARAMS.clear()
    pbr._EXTENSION_PARAMS.update(saved[0])
    pbr._MODEL_PARAM_NAMES.clear()
    pbr._MODEL_PARAM_NAMES.update(saved[1])
    pbr._MODEL_NAME_PREFIXES.clear()
    pbr._MODEL_NAME_PREFIXES.update(saved[2])
    pbr._registry = None  # rebuild singleton without test params


@pytest.mark.unit
def test_external_package_registers_new_params():
    register_model_bounds(
        "extmodel",
        params={
            "ext_k": ParameterInfo(0.1, 10.0, "1/day", "Recession coefficient", "baseflow"),
            "ext_smax": ParameterInfo(10.0, 500.0, "mm", "Max storage", "soil"),
        },
    )
    bounds = get_model_bounds("EXTMODEL")
    assert bounds == {
        "ext_k": {"min": 0.1, "max": 10.0, "transform": "linear"},
        "ext_smax": {"min": 10.0, "max": 500.0, "transform": "linear"},
    }
    assert "EXTMODEL" in registered_bound_models()


@pytest.mark.unit
def test_registration_composes_catalogue_names():
    """A model may compose its bound set from existing catalogue entries."""
    register_model_bounds("composed", names=["MBASE", "MFMAX"])
    bounds = get_model_bounds("composed")
    assert set(bounds) == {"MBASE", "MFMAX"}
    assert bounds["MBASE"] == pbr.get_registry().get_bounds("MBASE")


@pytest.mark.unit
def test_shared_catalogue_names_keep_central_definition():
    """Collisions with built-in catalogue names keep the central bounds."""
    central = pbr.get_registry().get_bounds("MBASE")
    register_model_bounds(
        "clash",
        params={"MBASE": ParameterInfo(-99.0, 99.0, "°C", "clashing", "snow")},
    )
    assert get_model_bounds("clash")["MBASE"] == central


@pytest.mark.unit
def test_strip_prefix_matches_namespaced_convention():
    register_model_bounds(
        "prefixed",
        params={"pfx_a": ParameterInfo(0.0, 1.0), "pfx_b": ParameterInfo(1.0, 2.0)},
        strip_prefix="pfx_",
    )
    assert set(get_model_bounds("prefixed")) == {"a", "b"}


@pytest.mark.unit
def test_builtin_models_served_and_unknown_raises():
    assert get_model_bounds("FUSE") == pbr.get_fuse_bounds()
    assert get_model_bounds("gsflow") == pbr.get_gsflow_bounds()
    with pytest.raises(KeyError, match="register_model_bounds"):
        get_model_bounds("no_such_model")


@pytest.mark.unit
def test_registration_after_singleton_creation_is_visible():
    pbr.get_registry()  # force singleton
    register_model_bounds(
        "late", params={"late_p": ParameterInfo(0.0, 1.0, "-", "late", "other")}
    )
    assert get_model_bounds("late") == {"late_p": {"min": 0.0, "max": 1.0, "transform": "linear"}}


@pytest.mark.unit
def test_catalogue_has_no_silent_bound_conflicts():
    """A parameter name defined in two category dicts with different bounds is
    a silent override (dict-merge order wins) — exactly how FUSE once
    calibrated against Snow-17 melt bounds. Same-valued duplicates are fine."""
    cats = [a for a in vars(pbr.ParameterBoundsRegistry) if a.endswith("_PARAMS")]
    seen: dict = {}
    conflicts = []
    for cat in cats:
        for name, info in getattr(pbr.ParameterBoundsRegistry, cat).items():
            key = (info.min, info.max, info.transform)
            if name in seen and seen[name][1] != key:
                conflicts.append(f"{name}: {seen[name][0]}{seen[name][1]} vs {cat}{key}")
            else:
                seen.setdefault(name, (cat, key))
    assert conflicts == [], "silently-overriding bound definitions:\n" + "\n".join(conflicts)
