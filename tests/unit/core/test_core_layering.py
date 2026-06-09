# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit mirror of the core-layering guard (RTI review item 19).

``symfluence.core`` must not depend on any upper-layer package. The standalone
guard ``scripts/check_core_layering.py`` enforces this in CI/pre-commit; this
test runs the same check inside the unit suite so a regression fails fast.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_PATH = REPO_ROOT / "scripts" / "check_core_layering.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_check_core_layering", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_no_disallowed_core_to_upper_imports():
    """core/ has no module-level (or un-allow-listed deferred) upper-layer imports."""
    guard = _load_guard()
    violations = guard.find_violations()
    assert violations == [], "core layering violations:\n" + "\n".join(
        v.describe() for v in violations
    )


@pytest.mark.unit
def test_no_module_level_upper_imports_in_core():
    """Belt-and-braces: zero module-level imports of upper layers (independent walk)."""
    core_root = REPO_ROOT / "src" / "symfluence" / "core"
    upper = {
        "models", "evaluation", "fews", "geospatial", "project",
        "data", "optimization", "cli", "agent", "gui", "tui",
    }
    offenders = []
    for py_file in core_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in tree.body:  # module-level statements only
            mods = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                parts = m.split(".")
                if m.startswith("symfluence.") and len(parts) >= 2 and parts[1] in upper:
                    offenders.append(f"{py_file.relative_to(core_root)}:{node.lineno}: {m}")
    assert offenders == [], "module-level core->upper imports:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_allowed_deferred_entries_are_real():
    """Each ALLOWED_DEFERRED entry must correspond to a real import (no stale allowances)."""
    guard = _load_guard()
    core_root = REPO_ROOT / "src" / "symfluence" / "core"
    for path_suffix, mod_prefix, reason in guard.ALLOWED_DEFERRED:
        target = core_root / path_suffix
        assert target.exists(), f"allow-listed file missing: {path_suffix}"
        text = target.read_text(encoding="utf-8")
        assert mod_prefix in text, (
            f"stale allowance: {path_suffix} no longer imports {mod_prefix} "
            f"(reason was: {reason})"
        )
