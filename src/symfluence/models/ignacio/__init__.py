# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO Fire Model Integration for SYMFLUENCE

This module provides SYMFLUENCE integration for the IGNACIO fire spread model,
which implements the Canadian Forest Fire Behavior Prediction (FBP) System
with Richards' elliptical wave propagation.

IGNACIO is an external Python package that must be installed separately:
    symfluence binary install ignacio

This module provides:
- IGNACIOConfig: Configuration model for fire simulation parameters
- IGNACIORunner: Model runner registered with ModelRegistry
- IGNACIOPreProcessor: Terrain and fuel data preparation
- IGNACIOPostProcessor: Result extraction and comparison with WMFire

The actual fire simulation logic is in the ignacio package:
- ignacio.simulation: Fire spread simulation
- ignacio.fbp: FBP fuel types and rate of spread
- ignacio.fwi: Fire Weather Index calculations
- ignacio.spread: Richards' elliptical propagation

References:
    IGNACIO: https://github.com/KatherineHopeReece/Fire-Engine-Framework
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Import SYMFLUENCE integration components (config layer only — the
# execution/calibration classes are imported lazily below).
try:
    from .config import IGNACIOConfig
except ImportError as e:
    logger.debug(f"Could not import IGNACIOConfig: {e}")

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'IGNACIORunner': ('.runner', 'IGNACIORunner'),
    'IGNACIOPreProcessor': ('.preprocessor', 'IGNACIOPreProcessor'),
    'IGNACIOPostProcessor': ('.postprocessor', 'IGNACIOPostProcessor'),
    'IGNACIOResultExtractor': ('.extractor', 'IGNACIOResultExtractor'),
    'IGNACIOModelOptimizer': ('.calibration', 'IGNACIOModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for IGNACIO module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['IGNACIOConfig'])


__all__ = [
    "IGNACIOConfig",
    "IGNACIORunner",
    "IGNACIOPreProcessor",
    "IGNACIOPostProcessor",
    "IGNACIOResultExtractor",
]

def register() -> None:
    """Register IGNACIO components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    try:
        from symfluence.core.registries import Registries as R
        from symfluence.core.registry import model_manifest

        model_manifest(
            "IGNACIO",
            build_instructions_module="symfluence.models.ignacio.build_instructions",
        )
        base = 'symfluence.models.ignacio'
        R.preprocessors.add_lazy("IGNACIO", f"{base}.preprocessor.IGNACIOPreProcessor")
        R.runners.add_lazy("IGNACIO", f"{base}.runner.IGNACIORunner")
        R.postprocessors.add_lazy("IGNACIO", f"{base}.postprocessor.IGNACIOPostProcessor")
        R.result_extractors.add_lazy("IGNACIO", f"{base}.extractor.IGNACIOResultExtractor")
        R.optimizers.add_lazy("IGNACIO", f"{base}.calibration.optimizer.IGNACIOModelOptimizer")
        R.workers.add_lazy("IGNACIO", f"{base}.calibration.worker.IGNACIOWorker")
        R.parameter_managers.add_lazy("IGNACIO", f"{base}.calibration.parameter_manager.IGNACIOParameterManager")
        R.config_schemas.add_lazy("IGNACIO", f"{base}.config_schema.IGNACIOConfig")
    except Exception:  # noqa: BLE001 — optional dependency
        pass


if TYPE_CHECKING:
    from .calibration import IGNACIOModelOptimizer
    from .extractor import IGNACIOResultExtractor
    from .postprocessor import IGNACIOPostProcessor
    from .preprocessor import IGNACIOPreProcessor
    from .runner import IGNACIORunner
