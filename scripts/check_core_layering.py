#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Layering guard for the ``core`` package (RTI review item 19).

The architecture intends ``symfluence.core`` to depend on nothing above it:
every other top-level package (``models``, ``evaluation``, ``fews``,
``geospatial``, ``project``, ``data``, ``optimization``, ``cli``, ``agent``,
``gui``, ``tui`` ...) may depend on ``core``, but ``core`` must not depend on
them. This script walks the AST of every module under ``src/symfluence/core``
and flags imports of those upper layers.

Two severities:

* **Module-level imports** (executed at import time) are forbidden outright.
  These create hard, import-time coupling and are what inverts the layering.
* **Deferred imports** (inside a function/method body) are the accepted
  inversion-of-control pattern -- a small number of registry-seeding and
  data-driven-validation seams genuinely need to reach up at call time. Each
  one must be listed in ``ALLOWED_DEFERRED`` with a reason. A new deferred
  edge that is not on the list fails the check, forcing a conscious decision.

Run directly (``python scripts/check_core_layering.py``) or via the mirrored
unit test ``tests/unit/core/test_core_layering.py``. Exit code is non-zero on
any violation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, NamedTuple, Set, Tuple

# Top-level symfluence subpackages that ``core`` must not depend on.
UPPER_LAYERS: Set[str] = {
    "models",
    "evaluation",
    "fews",
    "geospatial",
    "project",
    "data",
    "optimization",
    "cli",
    "agent",
    "gui",
    "tui",
}

# Accepted deferred (call-time) edges: (core-relative path, imported module prefix, reason).
# These are inversion-of-control seams -- core reaches up only at call time to
# seed a registry or validate a value against an upper-layer catalogue. Keep this
# list short and justified; adding to it should be a deliberate decision.
ALLOWED_DEFERRED: List[Tuple[str, str, str]] = [
    # (_bootstrap.py seeding of R.metrics is no longer an upward edge: the
    # metric registry moved into core.metrics with the calibration promotion.)
    (
        "config/factories.py",
        "symfluence.cli.init_presets",
        "from_preset factory resolves a named CLI init preset at call time; the "
        "preset catalogue transitively needs the models layer, and from_preset "
        "is a public SymfluenceConfig classmethod, so this stays a call-time seam.",
    ),
    (
        "calibration/optimizers/base_model_optimizer.py",
        "symfluence.evaluation.registry",
        "Final evaluation resolves the EvaluationRegistry at call time to score "
        "the calibrated run; evaluation is a capability package core must not "
        "import at module level.",
    ),
    (
        "contracts.py",
        "symfluence.data.backends.contract",
        "contract_version('acquisition') surfaces the acquisition family's "
        "PROTOCOL_VERSION (owned by data/, where external services import it) "
        "for a uniform read-only view; resolved at call time only.",
    ),
    (
        "calibration/optimizers/component_factory.py",
        "symfluence.optimization.calibration_targets",
        "The component factory resolves calibration targets through the "
        "optimization facade at call time; targets wrap evaluation evaluators, "
        "which core must not import at module level.",
    ),
]


class Violation(NamedTuple):
    path: str
    lineno: int
    module: str
    deferred: bool

    def describe(self) -> str:
        kind = "deferred (not allow-listed)" if self.deferred else "MODULE-LEVEL"
        return f"{self.path}:{self.lineno}: {kind} import of upper layer '{self.module}'"


def _upper_layer_of(module: str | None) -> str | None:
    """Return the upper-layer package name if *module* targets one, else None."""
    if not module or not module.startswith("symfluence."):
        return None
    parts = module.split(".")
    if len(parts) >= 2 and parts[1] in UPPER_LAYERS:
        return module
    return None


def _is_allowed_deferred(rel_path: str, module: str) -> bool:
    for path_suffix, mod_prefix, _reason in ALLOWED_DEFERRED:
        if rel_path.replace("\\", "/").endswith(path_suffix) and module.startswith(mod_prefix):
            return True
    return False


def _scan_file(py_file: Path, core_root: Path) -> List[Violation]:
    rel_path = str(py_file.relative_to(core_root))
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    # Map each import node to whether it is nested inside a function body or an
    # ``if TYPE_CHECKING:`` block (the latter is erased at runtime, so it is a
    # type-only edge — treated like a deferred import, requiring an allowance).
    deferred_nodes: Set[int] = set()
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(func):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    deferred_nodes.add(id(child))
        elif isinstance(func, ast.If):
            test = func.test
            is_type_checking = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
            ) or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_type_checking:
                for child in ast.walk(func):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        deferred_nodes.add(id(child))

    violations: List[Violation] = []
    for node in ast.walk(tree):
        modules: List[str] = []
        if isinstance(node, ast.ImportFrom):
            # level > 0 -> relative import inside symfluence (not an absolute upper-layer dep)
            if node.level == 0 and node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        else:
            continue

        for module in modules:
            if _upper_layer_of(module) is None:
                continue
            deferred = id(node) in deferred_nodes
            if deferred and _is_allowed_deferred(rel_path, module):
                continue
            violations.append(Violation(rel_path, node.lineno, module, deferred))
    return violations


def find_violations(core_root: Path | None = None) -> List[Violation]:
    if core_root is None:
        core_root = Path(__file__).resolve().parent.parent / "src" / "symfluence" / "core"
    violations: List[Violation] = []
    for py_file in sorted(core_root.rglob("*.py")):
        violations.extend(_scan_file(py_file, core_root))
    return violations


# ---------------------------------------------------------------------------
# Package-boundary rules (monorepo prep for service decomposition)
# ---------------------------------------------------------------------------
# Each rule: (scanned package, forbidden import prefixes, allowed deferred
# edges). Module-level imports of a forbidden prefix are always violations;
# deferred (function-body) imports must be allow-listed with a reason.
#
# Note: the deprecated lazy shims under optimization/ (parameter_managers,
# calibration_targets model files, workers/summa, mixins/summa_optimizer_mixin)
# resolve via importlib.import_module with string paths, which the AST scan
# does not see; the rules below guard against *reintroducing* static imports.

BoundaryRule = Tuple[str, Tuple[str, ...], List[Tuple[str, str, str]]]

BOUNDARY_RULES: List[BoundaryRule] = [
    (
        # The models layer must be removable: nothing may import it at module
        # level, so the framework imports cleanly with the model suite absent.
        "optimization",
        ("symfluence.models",),
        [
            (
                "_autodiscover.py",
                "symfluence.models",
                "Auto-discovery iterates installed model packages at call time; "
                "tolerates absence via try/except ImportError.",
            ),
            (
                "workers/utilities/__init__.py",
                "symfluence.models.utilities.routing_decider",
                "Deprecated lazy re-export (PEP 562) of RoutingDecider from its "
                "canonical home in the models package.",
            ),
            (
                "mixins/summa_optimizer_mixin.py",
                "symfluence.models.summa.calibration",
                "Deprecated lazy re-export (PEP 562) of SUMMAOptimizerMixin; "
                "the TYPE_CHECKING import is type-only and erased at runtime.",
            ),
        ],
    ),
    (
        # Models must not depend on interface layers, nor on the geospatial
        # capability package: the geometry utilities model preprocessors need
        # are part of the core contract surface (core.geometry_utils).
        # Build-environment helpers live in core.build; everything else the
        # adapters need is in core.
        "models",
        (
            "symfluence.cli",
            "symfluence.gui",
            "symfluence.tui",
            "symfluence.agent",
            "symfluence.fews",
            "symfluence.geospatial",
        ),
        [],
    ),
    (
        "evaluation",
        ("symfluence.models",),
        [
            (
                "analysis_manager.py",
                "symfluence.models",
                "Imports model packages at call time solely to trigger analyzer "
                "registration (IoC seam).",
            ),
        ],
    ),
    (
        "project",
        ("symfluence.models",),
        [
            (
                "manager_factory.py",
                "symfluence.models.model_manager",
                "Factory resolves ModelManager at call time; orchestration "
                "entry point for the model suite.",
            ),
        ],
    ),
    (
        # Process adapters resolve model components via R.runners /
        # R.result_extractors — no direct model imports remain.
        "coupling",
        ("symfluence.models",),
        [],
    ),
    (
        "fews",
        ("symfluence.models",),
        [
            (
                "pre_adapter.py",
                "symfluence.models.state",
                "Resolves the model state manager at call time.",
            ),
            (
                "post_adapter.py",
                "symfluence.models.state",
                "Resolves the model state manager at call time.",
            ),
        ],
    ),
    (
        "data_assimilation",
        ("symfluence.models",),
        [
            (
                "enkf/ensemble_manager.py",
                "symfluence.models.state",
                "EnKF resolves state-capable model interfaces at call time.",
            ),
        ],
    ),
    (
        "cli",
        ("symfluence.models",),
        [
            (
                "init_presets.py",
                "symfluence.models",
                "Preset catalogue lists installed model packages at call time.",
            ),
            (
                "preset_registry.py",
                "symfluence.models",
                "Preset registry lists installed model packages at call time.",
            ),
        ],
    ),
    (
        "gui",
        ("symfluence.models",),
        [
            (
                "server.py",
                "symfluence.models",
                "Server triggers model plugin registration at call time.",
            ),
        ],
    ),
    (
        "data",
        ("symfluence.models",),
        [],
    ),
    (
        "geospatial",
        ("symfluence.models",),
        [],
    ),
    (
        "reporting",
        ("symfluence.models",),
        [],
    ),
]


def find_boundary_violations(src_root: Path | None = None) -> List[Violation]:
    """Scan package-boundary rules; reuse the core scanner per package."""
    if src_root is None:
        src_root = Path(__file__).resolve().parent.parent / "src" / "symfluence"
    violations: List[Violation] = []
    global UPPER_LAYERS, ALLOWED_DEFERRED
    saved = (UPPER_LAYERS, ALLOWED_DEFERRED)
    try:
        for package, forbidden, allowed in BOUNDARY_RULES:
            pkg_root = src_root / package
            if not pkg_root.exists():
                continue
            # Reuse _scan_file by temporarily retargeting the module filter.
            UPPER_LAYERS = {f.split(".")[1] for f in forbidden}
            ALLOWED_DEFERRED = allowed
            for py_file in sorted(pkg_root.rglob("*.py")):
                for v in _scan_file(py_file, pkg_root):
                    if any(v.module.startswith(f) for f in forbidden):
                        violations.append(
                            Violation(f"{package}/{v.path}", v.lineno, v.module, v.deferred)
                        )
    finally:
        UPPER_LAYERS, ALLOWED_DEFERRED = saved
    return violations


def main() -> int:
    core_violations = find_violations()
    boundary_violations = find_boundary_violations()
    if not core_violations and not boundary_violations:
        print(
            "core layering OK: no disallowed core -> upper-layer imports; "
            "package boundaries OK (models removable, models -> no interface layers)"
        )
        return 0

    if core_violations:
        print("core layering VIOLATIONS found (RTI review item 19):\n", file=sys.stderr)
        for v in core_violations:
            print(f"  {v.describe()}", file=sys.stderr)
    if boundary_violations:
        print("package-boundary VIOLATIONS found:\n", file=sys.stderr)
        for v in boundary_violations:
            print(f"  {v.describe()}", file=sys.stderr)
    print(
        "\nFix: move shared code down into core, invert the dependency through "
        "the registry, or (for a genuine call-time IoC seam) add it to "
        "ALLOWED_DEFERRED / BOUNDARY_RULES in scripts/check_core_layering.py "
        "with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
