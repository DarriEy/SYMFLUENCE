# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Phase C guard 1 — in-tree models load through the entry-point plugin path.

In-tree models register *only* via `symfluence.plugins` entry points declared in
pyproject.toml (one `symfluence_<name> = "symfluence.models.<name>:register"` per
model), the same contract external plugins (jHBV, jXAJ, …) use. There is no
hard-coded model list — the source of truth is the set of `models/<name>` packages
that define a top-level `register()`.

These tests fail the build when that set drifts from the declared entry points, so a
model can never be silently dropped from (or wrongly added to) the load path:
* disk `register()` set  <->  pyproject entry points (install-independent),
* every declared entry point resolves to a callable,
* installed entry-point metadata matches (CI, fresh install).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

import symfluence.models

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - project targets 3.11+
    tomllib = None

ENTRY_POINT_GROUP = "symfluence.plugins"
MODEL_VALUE_PREFIX = "symfluence.models."
_REGISTER_RE = re.compile(r"^def register\b", re.MULTILINE)


def _disk_models() -> set[str]:
    """Model packages on disk that define a top-level register() callable.

    Source scan (no import), so optional-dependency models don't break collection.
    Helper packages (base, adapters, config, …) have no register() and are excluded
    naturally.
    """
    models_dir = Path(symfluence.models.__file__).resolve().parent
    found: set[str] = set()
    for init in models_dir.glob("*/__init__.py"):
        if _REGISTER_RE.search(init.read_text(encoding="utf-8")):
            found.add(init.parent.name)
    return found


def _find_pyproject() -> Path | None:
    """Walk up from the installed package to locate the source pyproject.toml."""
    for parent in Path(symfluence.__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _declared_in_tree_entry_points() -> dict[str, str]:
    """Return {entry_point_name: value} for in-tree models declared in pyproject."""
    pyproject = _find_pyproject()
    if pyproject is None or tomllib is None:
        pytest.skip("source pyproject.toml not locatable from installed package")
    data = tomllib.loads(pyproject.read_text())
    group = data.get("project", {}).get("entry-points", {}).get(ENTRY_POINT_GROUP, {})
    return {name: val for name, val in group.items() if val.startswith(MODEL_VALUE_PREFIX)}


def _model_from_value(value: str) -> str:
    """'symfluence.models.summa:register' -> 'summa'."""
    return value.split(":", 1)[0][len(MODEL_VALUE_PREFIX):]


DISK_MODELS = _disk_models()


# ----------------------------------------------------------------------
# Drift guard: disk register() set must match declared entry points
# ----------------------------------------------------------------------


def test_entry_points_match_disk_models():
    """Every models/<x> with a register() has an entry point, and vice versa."""
    declared = {_model_from_value(v) for v in _declared_in_tree_entry_points().values()}
    assert declared == DISK_MODELS, (
        "drift between models/<x> packages defining register() and the "
        '[project.entry-points."symfluence.plugins"] declarations.\n'
        f"on disk but no entry point: {sorted(DISK_MODELS - declared)}\n"
        f"entry point but no register(): {sorted(declared - DISK_MODELS)}"
    )


def test_entry_point_names_are_namespaced():
    """In-tree entry-point names are prefixed to avoid clashing with external plugins."""
    declared = _declared_in_tree_entry_points()
    bad = [name for name in declared if not name.startswith("symfluence_")]
    assert not bad, f"in-tree entry points must be namespaced 'symfluence_*': {bad}"


def test_entry_point_targets_point_to_register():
    """Each in-tree entry point targets the module's register() callable."""
    declared = _declared_in_tree_entry_points()
    bad = [v for v in declared.values() if not v.endswith(":register")]
    assert not bad, f"in-tree entry points must target ':register': {bad}"


def test_disk_models_nonempty():
    """Sanity: the disk scan actually finds models (guards against a broken scan)."""
    assert len(DISK_MODELS) >= 20, f"only {len(DISK_MODELS)} model register()s found on disk"


# ----------------------------------------------------------------------
# Contract: each importable model exposes a callable register()
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(DISK_MODELS))
def test_model_exposes_register(model):
    """Each in-tree model's __init__ exposes a zero-arg callable register().

    Optional heavy deps may block a model's import; that is an environment
    limitation, not a contract violation, so we skip rather than fail.
    """
    try:
        mod = importlib.import_module(f"symfluence.models.{model}")
    except Exception as exc:  # noqa: BLE001 - optional model dependency
        pytest.skip(f"{model} not importable in this environment: {exc}")
    assert callable(getattr(mod, "register", None)), (
        f"symfluence.models.{model}.register must be callable"
    )
    from symfluence.core.contracts import declared_plugin_contracts

    assert declared_plugin_contracts(mod.register) == symfluence.models.__symfluence_contracts__


# ----------------------------------------------------------------------
# Registration actually populates the registry, and is idempotent
# ----------------------------------------------------------------------


def test_representative_models_register_into_R():
    """Representative in-tree entry-point callables populate the registry.

    SUMMA, FUSE, GR and VIC all register a result_extractor and have no exotic
    runtime dependencies. Invoke their entry-point targets explicitly so this
    source-tree contract does not depend on editable-install metadata freshness;
    installed-metadata parity is covered separately below.
    """
    from symfluence.core.registries import R

    for name, package in (
        ("SUMMA", "summa"),
        ("FUSE", "fuse"),
        ("GR", "gr"),
        ("VIC", "vic"),
    ):
        register = importlib.import_module(f"symfluence.models.{package}").register
        register()
        assert R.result_extractors.get(name) is not None, f"{name} not registered"


def test_register_is_idempotent():
    """Calling a model's register() twice leaves a single, stable registry entry."""
    from symfluence.core.registries import R

    summa = importlib.import_module("symfluence.models.summa")
    summa.register()
    first = R.result_extractors.get("SUMMA")
    summa.register()
    second = R.result_extractors.get("SUMMA")
    assert first is second is not None


# ----------------------------------------------------------------------
# Installed entry-point metadata matches disk (CI fresh install; skips locally)
# ----------------------------------------------------------------------


def test_installed_entry_points_match_disk():
    """Installed metadata exposes exactly the in-tree models found on disk.

    Skips when the package has not been (re)installed since the entry points were
    declared — the only safe local state for an editable install of a worktree.
    """
    from importlib.metadata import entry_points

    eps = [
        ep
        for ep in entry_points(group=ENTRY_POINT_GROUP)
        if ep.value.startswith(MODEL_VALUE_PREFIX)
    ]
    if not eps:
        pytest.skip("in-tree entry points not in installed metadata; reinstall to test")

    # Parity is on declared names — no import needed, so optional-dependency
    # models (e.g. MPI/GPU runners absent on this host) don't break it.
    declared = {_model_from_value(ep.value) for ep in eps}
    assert declared == DISK_MODELS, (
        "installed entry-point metadata is out of sync with the on-disk register() set; "
        "reinstall (`pip install -e .`).\n"
        f"metadata only: {sorted(declared - DISK_MODELS)}\n"
        f"disk only: {sorted(DISK_MODELS - declared)}"
    )

    # Loadability: each EP that *can* import resolves to a callable. Skip models
    # whose optional dependencies are missing in this environment.
    for ep in eps:
        try:
            fn = ep.load()  # imports module + resolves :register
        except ImportError:
            continue
        assert callable(fn), f"{ep.name} -> {ep.value} did not resolve to a callable"
