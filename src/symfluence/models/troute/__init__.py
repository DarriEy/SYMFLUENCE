# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TRoute Model Utilities.

This package contains components for the t-route model integration:
- Preprocessor: Handles spatial and data preprocessing
- Runner: Manages model execution
- Postprocessor: Extracts routed streamflow results
- Extractor: Advanced result analysis utilities
- Config: Configuration adapter with auto-generated defaults
- Plotter: 4-panel routing diagnostics visualization

t-route is NOAA's channel routing model that provides:
- Multiple routing methods (Muskingum-Cunge, diffusive wave)
- Integration with NWM and other hydrologic models
- Support for large-scale river network routing
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import TRouteConfigAdapter

# Lazy import mapping — execution, extraction, and plotting classes pull the
# routing/netCDF/matplotlib stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'TRouteResultExtractor': ('.extractor', 'TRouteResultExtractor'),
    'TRoutePlotter': ('.plotter', 'TRoutePlotter'),
    'TRoutePostProcessor': ('.postprocessor', 'TRoutePostProcessor'),
    'TRoutePreProcessor': ('.preprocessor', 'TRoutePreProcessor'),
    'TRouteRunner': ('.runner', 'TRouteRunner'),
}


def __getattr__(name: str):
    """Lazy import handler for TRoute module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['TRouteConfigAdapter'])


__all__ = [
    'TRoutePreProcessor',
    'TRouteRunner',
    'TRoutePostProcessor',
    'TRouteResultExtractor',
    'TRouteConfigAdapter',
    'TRoutePlotter',
]

# Register all TRoute components via unified registry
from symfluence.core.registry import model_manifest


def register() -> None:
    """Register TROUTE components with the unified registry.

    Execution, extraction, and plotting classes are registered lazily —
    imported on first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "TROUTE",
        config_adapter=TRouteConfigAdapter,
        build_instructions_module="symfluence.models.troute.build_instructions",
    )
    base = 'symfluence.models.troute'
    R.preprocessors.add_lazy("TROUTE", f"{base}.preprocessor.TRoutePreProcessor")
    R.runners.add_lazy("TROUTE", f"{base}.runner.TRouteRunner", runner_method='run_troute')
    R.postprocessors.add_lazy("TROUTE", f"{base}.postprocessor.TRoutePostProcessor")
    R.result_extractors.add_lazy("TROUTE", f"{base}.extractor.TRouteResultExtractor")
    R.plotters.add_lazy("TROUTE", f"{base}.plotter.TRoutePlotter")


if TYPE_CHECKING:
    from .extractor import TRouteResultExtractor
    from .plotter import TRoutePlotter
    from .postprocessor import TRoutePostProcessor
    from .preprocessor import TRoutePreProcessor
    from .runner import TRouteRunner
