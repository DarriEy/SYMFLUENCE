# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB 2.0 Global Distributed Hydrological Model."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'PCRGLOBWBPreProcessor': ('.preprocessor', 'PCRGLOBWBPreProcessor'),
    'PCRGLOBWBRunner': ('.runner', 'PCRGLOBWBRunner'),
    'PCRGLOBWBResultExtractor': ('.extractor', 'PCRGLOBWBResultExtractor'),
    'PCRGLOBWBPostProcessor': ('.postprocessor', 'PCRGLOBWBPostProcessor'),
    'PCRGLOBWBModelOptimizer': ('.calibration', 'PCRGLOBWBModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for PCR-GLOBWB module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['PCRGLOBWBConfigAdapter'])


__all__ = [
    "PCRGLOBWBPreProcessor",
    "PCRGLOBWBRunner",
    "PCRGLOBWBResultExtractor",
    "PCRGLOBWBPostProcessor",
    "PCRGLOBWBConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import PCRGLOBWBConfigAdapter


def register() -> None:
    """Register PCRGLOBWB components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "PCRGLOBWB",
        config_adapter=PCRGLOBWBConfigAdapter,
        build_instructions_module="symfluence.models.pcrglobwb.build_instructions",
    )
    base = 'symfluence.models.pcrglobwb'
    R.preprocessors.add_lazy("PCRGLOBWB", f"{base}.preprocessor.PCRGLOBWBPreProcessor")
    R.runners.add_lazy("PCRGLOBWB", f"{base}.runner.PCRGLOBWBRunner")
    R.postprocessors.add_lazy("PCRGLOBWB", f"{base}.postprocessor.PCRGLOBWBPostProcessor")
    R.result_extractors.add_lazy("PCRGLOBWB", f"{base}.extractor.PCRGLOBWBResultExtractor")
    R.optimizers.add_lazy("PCRGLOBWB", f"{base}.calibration.optimizer.PCRGLOBWBModelOptimizer")
    R.workers.add_lazy("PCRGLOBWB", f"{base}.calibration.worker.PCRGLOBWBWorker")
    R.parameter_managers.add_lazy("PCRGLOBWB", f"{base}.calibration.parameter_manager.PCRGLOBWBParameterManager")

    # Spatial capabilities are owned by this package (service-decomposition
    # item 2): declared at plugin-discovery time so core carries no per-model
    # spatial knowledge and a capability change never needs a core release.
    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "PCRGLOBWB",
        ModelSpatialCapability(
            supported_modes={SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            requires_routing={
                # Internal accuTravelTime routing.
                SpatialMode.DISTRIBUTED: False,
            },
            warning_message=(
                "PCR-GLOBWB is inherently grid-based with internal "
                "accuTravelTime routing. Lumped mode uses a 3x3 grid."
            ),
        ),
    )


if TYPE_CHECKING:
    from .calibration import PCRGLOBWBModelOptimizer
    from .extractor import PCRGLOBWBResultExtractor
    from .postprocessor import PCRGLOBWBPostProcessor
    from .preprocessor import PCRGLOBWBPreProcessor
    from .runner import PCRGLOBWBRunner
