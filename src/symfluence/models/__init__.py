# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# src/symfluence.models/__init__.py
"""Hydrological model utilities.

This module provides:
- ModelRegistry: Central registry for model runners/preprocessors/postprocessors
- Execution Framework: Unified subprocess/SLURM execution (execution submodule)
- Config Schemas: Type-safe configuration contracts (config submodule)
- Templates: Base classes for new model implementations (templates submodule)
"""

# Import execution framework components
from __future__ import annotations

try:
    from .execution import (
        ExecutionResult,
        ModelExecutor,
        RoutingConfig,
        SlurmJobConfig,
        SpatialMode,
        SpatialOrchestrator,
    )
except ImportError:
    pass  # Optional - may not be needed by all users

# Import config schema components
try:
    from .config import (
        ModelConfigSchema,
        get_model_schema,
        validate_model_config,
    )
except ImportError:
    pass  # Optional

# Import template components
try:
    from .templates import (
        ModelRunResult,
        UnifiedModelRunner,
    )
except ImportError:
    pass  # Optional

# Import all models to register them
import logging
import warnings

logger = logging.getLogger(__name__)

# Distribution-level contract declaration. Every entry-point callable under
# ``symfluence.models.*`` inherits these targets during central discovery.
# Keeping this once at the package root avoids 30+ decorators drifting apart.
__symfluence_contracts__ = {
    "models": "0.2.0",
    "calibration": "0.2.0",
    "metrics": "0.1.0",
    "geospatial-utils": "0.1.0",
}

# Suppress experimental module warnings and missing optional dependency warnings
warnings.filterwarnings('ignore', message='.*is an EXPERIMENTAL module.*')
warnings.filterwarnings('ignore', message='.*import failed.*')

# In-tree models register through the `symfluence.plugins` entry points declared
# in pyproject.toml — one `symfluence_<name> = "symfluence.models.<name>:register"`
# per model. At `import symfluence`, the bootstrap (`core/_bootstrap._discover_plugins`)
# loads every entry point and calls its register(), the *same* non-privileged path
# external plugins use. There is deliberately no hard-coded model list and no import
# loop here: adding a model means creating the package with a top-level register() and
# declaring its entry point. `tests/unit/models/test_entry_point_dogfood.py` fails the
# build if the on-disk register() set and the pyproject entry points drift apart.
#
# NOTE: entry points are read from installed dist metadata, so a newly added or pulled
# model only becomes visible after `pip install -e .` regenerates that metadata.
# `_discover_plugins` logs a loud error if zero in-tree models are discovered (the
# stale-install signal).


def model_packages_with(submodule: str) -> tuple[str, ...]:
    """Return the in-tree model package names that contain *submodule*``.py``.

    e.g. ``model_packages_with('forcing_adapter')`` -> ``('fuse', 'gr', ...)``.

    Some optional model facilities (forcing adapters, init presets) register
    themselves as a side effect of importing a per-model submodule, so the set
    of models that have them cannot be read from the registry those imports
    populate — it is discovered from the source tree instead. This replaces the
    hardcoded ``SupportedModels.WITH_*`` lists, which had drifted from reality.
    """
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent
    return tuple(sorted(p.parent.name for p in pkg_dir.glob(f"*/{submodule}.py")))


__all__ = [
    "model_packages_with",
    # Execution Framework
    "ModelExecutor",
    "ExecutionResult",
    "SlurmJobConfig",
    "SpatialOrchestrator",
    "SpatialMode",
    "RoutingConfig",
    # Config Schemas
    "ModelConfigSchema",
    "get_model_schema",
    "validate_model_config",
    # Templates
    "UnifiedModelRunner",
    "ModelRunResult",
]
