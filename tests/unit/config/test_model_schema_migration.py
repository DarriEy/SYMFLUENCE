# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-model config schemas live with their model packages, not in core.

Service-decomposition guard: core/config must stay free of model-specific
schema classes. Each model package ships its own ``config_schema`` module and
registers it into ``R.config_schemas`` (manifest / ``register()``), the same
path an external plugin uses. Legacy import paths under
``symfluence.core.config.models`` resolve through the registry.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_CONFIG = Path(__file__).resolve().parents[3] / "src" / "symfluence" / "core" / "config"

# Classes that legitimately live in core/config (framework-level, not per-model).
ALLOWED_CLASSES = {"ModelConfig"}


@pytest.mark.unit
def test_core_config_defines_no_model_schema_classes():
    """No ``<Model>Config`` class may be (re)introduced under core/config/models."""
    from symfluence.core.config.models.model_configs import _LEGACY_SCHEMA_EXPORTS

    model_class_names = set(_LEGACY_SCHEMA_EXPORTS)
    offenders = []
    for py_file in (CORE_CONFIG / "models").glob("model_configs*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in model_class_names or (
                    node.name.endswith("Config") and node.name not in ALLOWED_CLASSES
                ):
                    offenders.append(f"{py_file.name}:{node.lineno}: class {node.name}")
    assert offenders == [], (
        "model-specific schema classes found in core/config — they belong in "
        "their model package (models/<model>/config_schema.py):\n" + "\n".join(offenders)
    )


@pytest.mark.unit
def test_every_legacy_export_is_registered_and_canonical():
    """Each legacy name resolves via the registry to the model package's class."""
    import importlib

    import symfluence  # noqa: F401 — triggers plugin registration
    from symfluence.core.config.models import model_configs
    from symfluence.core.registries import R

    for name, key in model_configs._LEGACY_SCHEMA_EXPORTS.items():
        schema = R.config_schemas.get(key)
        assert schema is not None, f"no schema registered for {key}"
        assert getattr(model_configs, name) is schema
        # canonical home is the model package
        assert schema.__module__.startswith("symfluence.models."), (
            f"{name} resolves to {schema.__module__}, expected a models package"
        )
        # and the model package module actually defines it
        mod = importlib.import_module(schema.__module__)
        assert getattr(mod, name) is schema


@pytest.mark.unit
def test_selected_model_gets_typed_config_from_registry():
    """End-to-end: validation populates model_specific via R.config_schemas."""
    from symfluence.core.config.models import SymfluenceConfig

    cfg = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR="/tmp/x",
        SYMFLUENCE_CODE_DIR="/tmp/y",
        DOMAIN_NAME="d",
        EXPERIMENT_ID="e",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="lumped",
        HYDROLOGICAL_MODEL="FUSE",
        FORCING_DATASET="ERA5",
    )
    fuse_cfg = cfg.model.fuse
    assert type(fuse_cfg).__name__ == "FUSEConfig"
    assert type(fuse_cfg).__module__ == "symfluence.models.fuse.config_schema"
    # unselected models still read as None (historical Optional semantics)
    assert cfg.model.summa is None
