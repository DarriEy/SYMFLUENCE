# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""In-tree code must import canonical core paths, never back-compat shims.

The service-decomposition promotions left shim modules at every historical
import path so external packages keep working (removal scheduled for 2.0,
see CHANGELOG "Deprecated"). In-tree code was migrated to the canonical
``symfluence.core.*`` paths — this guard keeps it that way, so the shims'
lifecycle is a purely external-facing decision.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "symfluence"

# Deprecated shim module prefixes (canonical homes are under symfluence.core).
SHIM_PREFIXES = (
    "symfluence.models.base",
    "symfluence.models.mixins",
    "symfluence.models.execution",
    "symfluence.models.state",
    "symfluence.models.templates",
    "symfluence.models.utilities",
    "symfluence.models.adapters",
    "symfluence.models.spatial_modes",
    "symfluence.models.config_resolution",
    "symfluence.models.model_manager",
    "symfluence.models.coupled",
    "symfluence.optimization.optimizers",
    "symfluence.optimization.mixins",
    "symfluence.optimization.workers.base_worker",
    "symfluence.optimization.workers.inmemory_worker",
    "symfluence.optimization.workers.summa",
    "symfluence.optimization.core.base_parameter_manager",
    "symfluence.optimization.core.parameter_bounds_registry",
    "symfluence.optimization.parameter_managers",
    "symfluence.evaluation.metrics",
    "symfluence.evaluation.metric_transformer",
    "symfluence.evaluation.utilities.streamflow_metrics",
    "symfluence.cli.services.build_snippets",
    "symfluence.cli.services.build_snippet_catalog",
    "symfluence.geospatial.geometry_utils",
)

# The shim files themselves (they self-reference their canonical target and
# legitimately live at the deprecated paths).
_SHIM_FILE_MARKER = "Back-compat shim"
_SHIM_DIR_HINTS = (
    "models/base", "models/mixins", "models/execution", "models/state",
    "models/templates", "models/utilities", "models/adapters",
    "optimization/optimizers", "optimization/mixins", "optimization/workers",
    "optimization/core", "optimization/parameter_managers",
    "cli/services", "evaluation", "geospatial",
)


def _is_shim_file(path: Path, text: str) -> bool:
    rel = str(path.relative_to(SRC)).replace("\\", "/")
    return _SHIM_FILE_MARKER in text and any(h in rel for h in _SHIM_DIR_HINTS)


@pytest.mark.unit
def test_no_in_tree_imports_of_deprecated_shim_paths():
    offenders = []
    for py in SRC.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if _is_shim_file(py, text):
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if any(name == p or name.startswith(p + ".") for p in SHIM_PREFIXES):
                    offenders.append(
                        f"{py.relative_to(SRC)}:{node.lineno}: {name}"
                    )
    assert offenders == [], (
        "in-tree imports of deprecated shim paths (use the canonical "
        "symfluence.core.* homes):\n" + "\n".join(offenders)
    )
