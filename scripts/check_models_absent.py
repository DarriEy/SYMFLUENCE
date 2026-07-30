#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Prove the framework runs against an installation with no models package.

This is the *physical* counterpart to
``tests/conformance/test_models_absent.py``. That test simulates absence inside
the developer's own tree with a ``sys.meta_path`` blocker and a stubbed
``entry_points``, which is fast and runs on every platform — but the models code
is still on disk and still importable, so it cannot catch a real packaging
dependency: a console script that imports a model at start-up, an entry point
pointing into ``symfluence.models``, or ``importlib.resources`` reaching for a
model's bundled data.

Run this against an installed wheel whose ``symfluence/models`` directory has
actually been deleted. Nothing is patched here; if the framework needs the models
layer, the import simply fails.

Usage:
    python scripts/check_models_absent.py

Intended to be invoked by the interpreter of the stripped environment, not the
one that built it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FRAMEWORK_IMPORTS = (
    ("symfluence.core.contracts", "contract_version"),
    ("symfluence.core.registries", "R"),
    ("symfluence.project.manager_factory", "LazyManagerDict"),
    ("symfluence.project.workflow_orchestrator", "WorkflowOrchestrator"),
)


def _fail(message: str) -> None:
    print(f"models-absent contract FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _check_models_really_gone() -> None:
    """Refuse to pass vacuously if the models package is still installed.

    Without this the job would go green on an environment where nothing was
    stripped, which is precisely the false assurance the simulated test already
    risks.
    """
    spec = importlib.util.find_spec("symfluence.models")
    if spec is not None:
        location = getattr(spec, "origin", None) or getattr(spec, "submodule_search_locations", None)
        _fail(
            "symfluence.models is still importable at "
            f"{location} — this check must run against a stripped install, "
            "otherwise it proves nothing"
        )


def _check_framework_imports() -> None:
    import importlib

    for module_name, attribute in FRAMEWORK_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — any failure is a contract breach
            _fail(f"importing {module_name} raised {type(exc).__name__}: {exc}")
        if not hasattr(module, attribute):
            _fail(f"{module_name} has no attribute {attribute!r}")


def _check_no_model_modules_were_imported() -> None:
    leaked = sorted(
        name
        for name in sys.modules
        if name == "symfluence.models" or name.startswith("symfluence.models.")
    )
    if leaked:
        _fail(f"framework imported model modules: {leaked}")


def _check_registries_are_empty() -> None:
    from symfluence.core.registries import R

    models = R.registered_models()
    if models:
        _fail(
            f"expected no registered models in a models-absent install, got {models}"
        )


def _check_contract_version_still_resolves() -> None:
    from symfluence.core.contracts import contract_version

    version = contract_version("models")
    if not version:
        _fail("contract_version('models') must resolve even with models absent")


def main() -> int:
    # Order matters: prove absence first, so a non-stripped environment fails
    # loudly instead of sailing through the rest.
    _check_models_really_gone()
    _check_framework_imports()
    _check_contract_version_still_resolves()
    _check_registries_are_empty()
    _check_no_model_modules_were_imported()

    print("MODELS-ABSENT-CONTRACT-OK")
    print(f"  interpreter: {sys.executable}")
    print(f"  cwd:         {Path.cwd()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
