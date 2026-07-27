# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Spatial capabilities are owned by model packages, never by ``core``.

``core.modeling.spatial_modes`` used to hold a 16-entry table of per-model
``ModelSpatialCapability`` values. Those values moved into the ``register()``
of the package that owns each model, contributed through the public
``register_model_spatial_capability()`` seam at plugin-discovery time — the
same path an out-of-tree plugin uses.

These are the drift guards for that arrangement:

* core declares the record *type* and the read side, and no instance of it;
* every in-tree model that has a capability declares it from its own package;
* a model that declares nothing (or whose package is not installed) is
  permitted any spatial mode, so the seam can never be more restrictive than
  having no seam at all;
* nothing in core reads a capability at import time — reads must happen after
  plugin discovery, i.e. from inside a function body.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import symfluence  # noqa: F401 — triggers plugin discovery -> model register()
from symfluence.core.modeling import spatial_modes
from symfluence.core.modeling.spatial_modes import (
    SpatialMode,
    get_model_capabilities,
    registered_spatial_capability_models,
    validate_spatial_mode,
)

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "symfluence"
CORE = SRC / "core"
MODELS = SRC / "models"

#: The models whose capability declarations used to live in core. They moved to
#: their packages; a package that stops declaring one silently loses spatial
#: validation (unknown models are permitted any mode), which no test would
#: otherwise notice. Frozen here so that regression is loud.
MIGRATED_FROM_CORE = {
    "CRHM", "FUSE", "GNN", "GR", "GSFLOW", "HYPE", "LSTM", "MESH", "MHM",
    "NGEN", "PCRGLOBWB", "RHESSYS", "SUMMA", "SWAT", "VIC", "WATFLOOD",
}

#: Names that read the capability registry. A module-level call to any of them
#: would run before plugin discovery has populated it.
_CAPABILITY_READERS = {
    "validate_spatial_mode",
    "get_model_capabilities",
    "spatial_capabilities",
    "registered_spatial_capability_models",
    "MODEL_SPATIAL_CAPABILITIES",
}


def _model_init_files() -> list[Path]:
    return sorted(MODELS.glob("*/__init__.py"))


def _declared_in_package_sources() -> dict[str, str]:
    """Model key -> owning package, read from the model ``register()`` sources."""
    declared: dict[str, str] = {}
    for path in _model_init_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "register_model_spatial_capability" or not node.args:
                continue
            key = node.args[0]
            assert isinstance(key, ast.Constant) and isinstance(key.value, str), (
                f"{path} registers a spatial capability under a non-literal key; "
                "this guard (and a human reader) cannot tell which model it owns"
            )
            declared[key.value.upper()] = path.parent.name
    return declared


@pytest.mark.unit
def test_core_holds_no_per_model_spatial_values():
    """The compatibility seed is gone — core declares no capability values."""
    assert not hasattr(spatial_modes, "_BUILTIN_SPATIAL_CAPABILITIES"), (
        "core is seeding per-model spatial capabilities again; the values "
        "belong in the register() of the package that owns each model"
    )
    assert not hasattr(spatial_modes, "_seed_builtin_capabilities")

    source = Path(spatial_modes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    constructions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelSpatialCapability"
    ]
    assert not constructions, (
        "core constructs a ModelSpatialCapability: it owns the record type and "
        "the read side, never a model's values"
    )


@pytest.mark.unit
def test_no_core_module_declares_a_capability():
    """No module anywhere under ``core`` calls the registration seam."""
    seam_module = Path(spatial_modes.__file__).resolve()
    offenders = []
    for path in CORE.rglob("*.py"):
        if path.resolve() == seam_module:
            # Defines the seam, and forwards legacy
            # ``MODEL_SPATIAL_CAPABILITIES['X'] = cap`` writes into it.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_model_spatial_capability"
            ):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        f"core registers spatial capabilities in {offenders} — that is per-model "
        "knowledge, and it must come from the owning package's register()"
    )


@pytest.mark.unit
@pytest.mark.parametrize("model", sorted(MIGRATED_FROM_CORE))
def test_migrated_model_declares_from_its_own_package(model):
    """Each model that moved out of core declares its capability, in its package."""
    declared = _declared_in_package_sources()
    assert model in declared, (
        f"{model} declares no spatial capability from any model package. It had "
        "one in core before the migration; losing it silently downgrades the "
        "model to 'unknown', which is validated against nothing."
    )
    assert declared[model] == model.lower(), (
        f"{model}'s capability is declared by package '{declared[model]}' — a "
        "model's capability must be owned by the model's own package"
    )
    assert get_model_capabilities(model) is not None, (
        f"{model} declares a capability in source but none is registered at "
        "runtime; its register() is not reaching the seam"
    )


@pytest.mark.unit
def test_package_declarations_all_reach_the_registry():
    """Every source declaration lands in the live registry after discovery."""
    declared = _declared_in_package_sources()
    registered = set(registered_spatial_capability_models())
    missing = sorted(set(declared) - registered)
    assert not missing, (
        f"declared in a model package but absent from the registry: {missing}. "
        "Plugin discovery did not run the declaring register(), or the call "
        "sits behind a statement that raised."
    )


@pytest.mark.unit
def test_no_in_tree_model_is_declared_from_outside_its_package():
    """A registered in-tree model's declaration comes from its own package."""
    declared = _declared_in_package_sources()
    in_tree_packages = {path.parent.name for path in _model_init_files()}
    foreign = sorted(
        key for key in registered_spatial_capability_models()
        if key.lower() in in_tree_packages and key not in declared
    )
    assert not foreign, (
        f"{foreign} have in-tree model packages but their capabilities are "
        "registered from somewhere else (core seeding them again?)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("mode", list(SpatialMode))
def test_undeclared_model_is_permitted_any_mode(mode):
    """Declaring nothing is never more restrictive than declaring something."""
    assert get_model_capabilities("NOT_A_REAL_MODEL") is None
    assert validate_spatial_mode("NOT_A_REAL_MODEL", mode) == (True, None)


@pytest.mark.unit
def test_absent_package_degrades_exactly_like_an_undeclared_model():
    """An uninstalled model package leaves its model unknown, not invalid.

    SWAT declares lumped-only support, so distributed is rejected while its
    package is installed. With the package absent nothing registers — and the
    model must then be treated exactly as any never-declared model is.
    """
    capabilities = spatial_modes.MODEL_SPATIAL_CAPABILITIES
    declared = capabilities["SWAT"]
    assert validate_spatial_mode("SWAT", SpatialMode.DISTRIBUTED)[0] is False

    capabilities.pop("SWAT")
    try:
        assert get_model_capabilities("SWAT") is None
        assert validate_spatial_mode("SWAT", SpatialMode.DISTRIBUTED) == (True, None)
    finally:
        spatial_modes.register_model_spatial_capability("SWAT", declared)
    assert get_model_capabilities("SWAT") is declared


@pytest.mark.unit
def test_core_never_reads_a_capability_at_import_time():
    """Capability reads happen at call time, after plugin discovery has run.

    A module-level read would execute while ``symfluence`` is still importing —
    before ``_bootstrap._discover_plugins`` has called any model ``register()``
    — and would therefore see an empty registry.
    """
    offenders: list[str] = []
    for path in CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deferred: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for child in ast.walk(node):
                    deferred.add(id(child))
        for node in ast.walk(tree):
            if id(node) in deferred:
                continue
            name = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
            if name in _CAPABILITY_READERS:
                offenders.append(
                    f"{path.relative_to(REPO)}:{node.lineno} reads {name}"
                )
    assert not offenders, (
        "capability registry read at import time (before plugin discovery "
        f"populates it): {offenders}"
    )
