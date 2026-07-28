# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Boundary guards for ``symfluence.resources`` (bundled package data).

``resources`` used to sit in neither list of ``scripts/check_core_layering.py``,
so every edge in and out of it was unguarded. Two problems hid there:

1. a real cycle — ``resources.manager.get_base_settings_dir`` imported
   ``symfluence.core.registries`` to resolve a model's settings anchor, while
   ``core/modeling/base/base_preprocessor.py`` imported that resolver back;
2. seven model modules read their own settings data straight out of
   ``symfluence.resources``, an edge the extracted models distribution must not
   have.

The resolution now lives in ``core.modeling.base_settings``. These tests pin
both directions so the shape cannot silently regress: ``resources`` imports no
``symfluence.core``, and ``models`` reaches settings only through core. The
``models -> resources`` direction is *also* enforced by the layering guard's
BOUNDARY_RULES; asserting it here keeps the invariant readable next to its
sibling and catches a weakening of the rule itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "symfluence"


def _imported_modules(py_file: Path):
    """Yield ``(lineno, module, deferred)`` for every absolute import.

    ``deferred`` is True for imports inside a function body. The distinction
    matters: a MODULE-LEVEL import is what creates an import-time cycle, while
    a call-time one in a deprecated shim cannot — it resolves only when someone
    touches the old name. Treating both alike is what pushed the shim into
    hiding its edge behind ``importlib.import_module`` on a string, which
    satisfied the check without changing the dependency.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    deferred_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    deferred_nodes.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module, id(node) in deferred_nodes
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, id(node) in deferred_nodes


def _python_files(package: str):
    for py_file in sorted((SRC / package).rglob("*.py")):
        if "__pycache__" not in py_file.parts:
            yield py_file


def _offenders(package: str, forbidden_prefix: str, *, include_deferred: bool = True) -> list[str]:
    found = []
    for py_file in _python_files(package):
        for lineno, module, deferred in _imported_modules(py_file):
            if deferred and not include_deferred:
                continue
            if module == forbidden_prefix or module.startswith(forbidden_prefix + "."):
                rel = py_file.relative_to(SRC).as_posix()
                kind = "deferred" if deferred else "MODULE-LEVEL"
                found.append(f"{rel}:{lineno}: {kind} {module}")
    return found


def test_resources_does_not_import_core_at_module_level():
    """No cycle: importing bundled data must not pull in the framework core."""
    assert _offenders("resources", "symfluence.core", include_deferred=False) == [], (
        "symfluence.resources imports symfluence.core at module level — that "
        "re-creates the cycle closed by promoting base-settings resolution to "
        "core.modeling.base_settings. resources holds data and accessors for "
        "its OWN files only."
    )


def test_the_only_resources_to_core_edge_is_the_deprecated_shim():
    """One call-time edge is permitted, and only where it is documented.

    The deprecated ``get_base_settings_dir`` / ``copy_base_settings_to_project``
    names must keep working for external packages, and resolving them means
    reaching core. Doing that inside ``__getattr__`` is harmless — nothing is
    imported until someone touches the old name. Pinning WHERE it may happen is
    what stops a second, less considered edge appearing under the same excuse.
    """
    deferred = _offenders("resources", "symfluence.core")
    assert [edge for edge in deferred if not edge.startswith("resources/__init__.py:")] == [], (
        f"unexpected resources -> core edge outside the shim: {deferred}"
    )


def test_models_does_not_import_resources():
    """Models reach settings through the core contract, not bundled data."""
    assert _offenders("models", "symfluence.resources") == [], (
        "symfluence.models imports symfluence.resources — use "
        "symfluence.core.modeling.base_settings.get_base_settings_dir so the "
        "models distribution depends on symfluence.core alone."
    )


def test_layering_guard_forbids_models_to_resources():
    """The mechanical guard carries the same rule (belt and braces)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_core_layering", REPO / "scripts" / "check_core_layering.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    forbidden_for_models = {
        prefix
        for package, prefixes, _allowed in guard.BOUNDARY_RULES
        if package == "models"
        for prefix in prefixes
    }
    assert "symfluence.resources" in forbidden_for_models
    assert "resources" not in guard.UPPER_LAYERS, (
        "core legitimately reads bundled data — that is a downward edge and "
        "must stay legal."
    )


@pytest.mark.parametrize(
    "name", ["get_base_settings_dir", "copy_base_settings_to_project"]
)
def test_promoted_names_stay_importable_from_resources(name):
    """Deprecated shim keeps external callers working, same object identity."""
    import symfluence.resources as resources
    from symfluence.core.modeling import base_settings

    assert getattr(resources, name) is getattr(base_settings, name)


def test_resources_rejects_unknown_attributes():
    import symfluence.resources as resources

    with pytest.raises(AttributeError):
        resources.__getattr__("definitely_not_a_resource_accessor")


def test_core_modeling_facade_exports_base_settings_resolution():
    from symfluence.core.modeling import copy_base_settings_to_project, get_base_settings_dir
    from symfluence.core.modeling.base_settings import (
        copy_base_settings_to_project as canonical_copy,
    )
    from symfluence.core.modeling.base_settings import (
        get_base_settings_dir as canonical_get,
    )

    assert get_base_settings_dir is canonical_get
    assert copy_base_settings_to_project is canonical_copy


def test_registry_anchor_wins_over_bundled_fallback():
    """Resolution order is registry-first, bundled second — unchanged."""
    from symfluence.core.modeling.base_settings import get_base_settings_dir
    from symfluence.core.registries import R

    anchor = R.base_settings.get("SUMMA")
    assert anchor == "symfluence.models.summa"
    resolved = get_base_settings_dir("SUMMA")
    assert resolved.name == "base_settings"
    assert resolved.parent.name == "summa"


def test_bundled_fallback_serves_unregistered_names():
    """TEST has no registry anchor, so it comes from resources/base_settings."""
    from symfluence.core.modeling.base_settings import get_base_settings_dir
    from symfluence.resources import get_bundled_base_settings_dir

    assert get_base_settings_dir("TEST") == get_bundled_base_settings_dir("TEST")


def test_unknown_model_still_raises_file_not_found():
    from symfluence.core.modeling.base_settings import get_base_settings_dir

    with pytest.raises(FileNotFoundError, match="__no_such_model__"):
        get_base_settings_dir("__no_such_model__")
