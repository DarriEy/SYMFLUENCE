# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The parameter-bounds extension seam (service-decomposition prep).

External model packages must be able to contribute calibration bounds without
editing core: ``register_model_bounds()`` + ``get_model_bounds()``.

Since item 2 of the decomposition campaign the in-tree models use the same
seam, so this file also pins the three-tier split itself: composition and
solo definitions are owned by the model package, shared definitions stay
central, and neither side may silently override the other.
"""
from __future__ import annotations

import pytest

from symfluence.core.calibration.parameters import (
    ParameterInfo,
    bounds_registration_conflicts,
    get_model_bounds,
    register_model_bounds,
    registered_bound_models,
)
from symfluence.core.calibration.parameters import parameter_bounds_registry as pbr


@pytest.fixture(autouse=True)
def _clean_seam():
    """Keep seam state test-local."""
    saved = (dict(pbr._EXTENSION_PARAMS), dict(pbr._MODEL_PARAM_NAMES),
             dict(pbr._MODEL_NAME_PREFIXES), list(pbr._REGISTRATION_CONFLICTS))
    yield
    pbr._EXTENSION_PARAMS.clear()
    pbr._EXTENSION_PARAMS.update(saved[0])
    pbr._MODEL_PARAM_NAMES.clear()
    pbr._MODEL_PARAM_NAMES.update(saved[1])
    pbr._MODEL_NAME_PREFIXES.clear()
    pbr._MODEL_NAME_PREFIXES.update(saved[2])
    # Deliberate clashes below must not leak into the post-discovery
    # conflict report the whole-install guard reads.
    pbr._REGISTRATION_CONFLICTS[:] = saved[3]
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
def test_losing_redefinition_is_reported_not_silent():
    """The override that central-wins performs must leave a trace.

    Splitting the catalogue across packages opens a second route to the #368
    defect: two packages (or a package and core) defining one name with
    different bounds, the loser silently ignored. ``central wins`` is the right
    policy; being unable to notice it happened is not.
    """
    before = len(bounds_registration_conflicts())
    register_model_bounds(
        "loud_clash",
        params={"MBASE": ParameterInfo(-99.0, 99.0, "°C", "clashing", "snow")},
    )
    reported = bounds_registration_conflicts()[before:]
    assert len(reported) == 1
    assert "MBASE" in reported[0]
    assert "LOUD_CLASH" in reported[0]


@pytest.mark.unit
def test_identical_redefinition_is_not_a_conflict():
    """Only a *disagreement* is worth reporting — a same-valued repeat is not."""
    central = pbr.ParameterBoundsRegistry.SACSMA_PARAMS["MBASE"]
    before = len(bounds_registration_conflicts())
    register_model_bounds(
        "quiet_repeat",
        params={"MBASE": ParameterInfo(central.min, central.max, "°C", "same", "snow")},
    )
    assert bounds_registration_conflicts()[before:] == []


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


# ---------------------------------------------------------------------------
# The three-tier split (service-decomposition item 2)
# ---------------------------------------------------------------------------

#: In-tree models that own their bound set. Each registers from its package
#: ``register()``, which plugin discovery invokes on ``import symfluence``.
_PACKAGE_OWNED = {
    "FUSE": "symfluence.models.fuse",
    "GR": "symfluence.models.gr",
    "GSFLOW": "symfluence.models.gsflow",
    "HYPE": "symfluence.models.hype",
    "IGNACIO": "symfluence.models.ignacio",
    "MESH": "symfluence.models.mesh",
    "MIZUROUTE": "symfluence.models.mizuroute",
    "NGEN": "symfluence.models.ngen",
    "NGEN_CFE": "symfluence.models.ngen",
    "NGEN_NOAH": "symfluence.models.ngen",
    "NGEN_PET": "symfluence.models.ngen",
    "NGEN_SACSMA": "symfluence.models.ngen",
    "NGEN_SNOW17": "symfluence.models.ngen",
    "NGEN_TOPMODEL": "symfluence.models.ngen",
    "NOAHMP": "symfluence.models.noahmp",
    "RHESSYS": "symfluence.models.rhessys",
    "VIC": "symfluence.models.vic",
    "WATFLOOD": "symfluence.models.watflood",
}

#: Bound sets core still serves itself. DEPTH is SUMMA's soil-depth facet (no
#: model package of its own yet); the rest are served by external JAX plugin
#: packages that predate the seam and still import ``get_<model>_bounds()``.
_STILL_CENTRAL = {"DEPTH", "HBV", "HECHMS", "SACSMA", "SNOW17", "TOPMODEL",
                  "XINANJIANG"}


@pytest.mark.unit
def test_every_servable_model_is_owned_exactly_once():
    """No third category, and no drift between the two lists above."""
    assert set(registered_bound_models()) == set(_PACKAGE_OWNED) | _STILL_CENTRAL
    assert not (set(_PACKAGE_OWNED) & _STILL_CENTRAL)


@pytest.mark.unit
@pytest.mark.parametrize("model", sorted(_PACKAGE_OWNED))
def test_package_owned_models_register_their_composition(model):
    """Tier A: the model package, not core, says what the model calibrates."""
    assert model in pbr._MODEL_PARAM_NAMES, (
        f"{model} did not register a composition — plugin discovery either did "
        "not run or its register() no longer calls register_bounds()"
    )
    assert get_model_bounds(model)


@pytest.mark.unit
@pytest.mark.parametrize("model", sorted(_PACKAGE_OWNED))
def test_package_owned_models_are_not_duplicated_in_core(model):
    """Core keeps no second copy of a migrated bound set.

    A fallback copy is exactly what drifts: ``optimization/core/
    parameter_bounds_registry.py`` is a stale duplicate of this catalogue that
    still serves FUSE the Snow-17 melt bounds #368 removed. The migrated
    helpers must resolve the registration or fail loudly, never serve data of
    their own.
    """
    pbr._MODEL_PARAM_NAMES.pop(model)
    with pytest.raises(KeyError, match=_PACKAGE_OWNED[model]):
        pbr._BUILTIN_MODEL_BOUNDS[model]()


@pytest.mark.unit
def test_shared_parameters_are_defined_only_in_core():
    """Tier C: a model package must never carry its own copy of a shared name.

    ``register_model_bounds`` would keep the central definition anyway, so a
    duplicate is not a wrong *value* — it is a second place to edit, which is
    how the two definitions of ``MBASE`` drifted apart in the first place.
    """
    central = set()
    for attr in pbr.ParameterBoundsRegistry.CATEGORY_ATTRS:
        central |= set(getattr(pbr.ParameterBoundsRegistry, attr))
    duplicated = central & set(pbr._EXTENSION_PARAMS)
    assert not duplicated, (
        f"model packages redefine central parameters {sorted(duplicated)} — "
        "compose them by name in the bound set instead"
    )


@pytest.mark.unit
def test_split_catalogue_covers_every_composed_name():
    """Every name any model composes resolves to a definition.

    Guards the failure mode the split introduces: a composition listing a name
    whose definition stayed on the other side of the boundary. Such a name is
    dropped without error by ``get_bounds_for_params``, silently shrinking the
    calibrated parameter set.
    """
    known = set(pbr.get_registry().all_param_names)
    missing = {}
    for model, names in pbr._MODEL_PARAM_NAMES.items():
        absent = sorted(set(names) - known)
        if absent:
            missing[model] = absent
    assert not missing, f"compositions reference undefined parameters: {missing}"


@pytest.mark.unit
def test_installed_packages_agree_with_the_central_catalogue():
    """Whole-install guard: discovery produced no losing redefinition."""
    assert bounds_registration_conflicts() == []


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
