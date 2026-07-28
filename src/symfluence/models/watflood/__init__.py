# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""WATFLOOD (Kouwen) Distributed Flood Forecasting Model.

WATFLOOD is a physically-based, distributed hydrological model using
Grouped Response Units (GRUs) on a regular grid with internal channel
routing. It requires only precipitation and temperature forcing
(simplified energy balance).

Input Files:
    .shd: Watershed definition (GRU grid)
    .par: Parameters (per-land-class blocks)
    .evt: Event control
    .met/.rag: Meteorological forcing

Output Files:
    .tb0: Time-bin streamflow/state output
    .csv: Summary output

References:
    Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
    & Flood Forecasting System. University of Waterloo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'WATFLOODPreProcessor': ('.preprocessor', 'WATFLOODPreProcessor'),
    'WATFLOODRunner': ('.runner', 'WATFLOODRunner'),
    'WATFLOODResultExtractor': ('.extractor', 'WATFLOODResultExtractor'),
    'WATFLOODPostProcessor': ('.postprocessor', 'WATFLOODPostProcessor'),
    'WATFLOODModelOptimizer': ('.calibration', 'WATFLOODModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for WATFLOOD module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['WATFLOODConfigAdapter'])


__all__ = [
    "WATFLOODPreProcessor",
    "WATFLOODRunner",
    "WATFLOODResultExtractor",
    "WATFLOODPostProcessor",
    "WATFLOODConfigAdapter",
]

# Register all WATFLOOD components via unified registry
from symfluence.core.registry import model_manifest

from .config import WATFLOODConfigAdapter


def register() -> None:
    """Register WATFLOOD components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "WATFLOOD",
        config_adapter=WATFLOODConfigAdapter,
        build_instructions_module="symfluence.models.watflood.build_instructions",
    )
    base = 'symfluence.models.watflood'
    R.preprocessors.add_lazy("WATFLOOD", f"{base}.preprocessor.WATFLOODPreProcessor")
    R.runners.add_lazy("WATFLOOD", f"{base}.runner.WATFLOODRunner")
    R.postprocessors.add_lazy("WATFLOOD", f"{base}.postprocessor.WATFLOODPostProcessor")
    R.result_extractors.add_lazy("WATFLOOD", f"{base}.extractor.WATFLOODResultExtractor")
    R.optimizers.add_lazy("WATFLOOD", f"{base}.calibration.optimizer.WATFLOODModelOptimizer")
    R.workers.add_lazy("WATFLOOD", f"{base}.calibration.worker.WATFLOODWorker")
    R.parameter_managers.add_lazy("WATFLOOD", f"{base}.calibration.parameter_manager.WATFLOODParameterManager")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "WATFLOOD",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            requires_routing={
                SpatialMode.DISTRIBUTED: False,  # Internal channel routing
                SpatialMode.LUMPED: False,
            },
            warning_message=(
                "WATFLOOD uses GRU-grid distributed structure with internal "
                "channel routing. Lumped mode uses a single-GRU configuration."
            ),
        ),
    )

    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .calibration import WATFLOODModelOptimizer
    from .extractor import WATFLOODResultExtractor
    from .postprocessor import WATFLOODPostProcessor
    from .preprocessor import WATFLOODPreProcessor
    from .runner import WATFLOODRunner
