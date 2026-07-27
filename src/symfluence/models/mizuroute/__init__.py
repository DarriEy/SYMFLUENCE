# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MizuRoute River Routing Model.

This module implements integration with mizuRoute, a large-scale river routing
model developed at NCAR (National Center for Atmospheric Research). MizuRoute
routes runoff from hydrological models through river networks to produce
streamflow at any location in the network.

Routing Schemes:
    1. **Impulse Response Function (IRF)**: Unit hydrograph approach that
       convolves upstream runoff with a gamma-shaped transfer function.
       Fast and suitable for large-scale applications.

    2. **Kinematic Wave Tracking (KWT)**: Tracks individual runoff pulses
       through the network using kinematic wave approximation. More
       physically based but computationally intensive.

    3. **Diffusive Wave (DW)**: Solves the diffusive wave equation for
       each river segment. Most accurate for backwater effects but slowest.

Design Rationale:
    MizuRoute addresses the need for consistent routing across models:
    - Hydrological models often have inconsistent or simplified routing
    - MizuRoute provides unified routing for any runoff source
    - Enables routing SUMMA, FUSE, GR, HYPE, or any model through same network
    - Supports continental-scale applications (used in NWM, ISIMIP)

Spatial Structure:
    - Segments: River reach elements with properties (length, slope, width)
    - HRUs: Hydrologic response units contributing runoff to segments
    - Network Topology: Upstream-downstream connectivity (segId, downSegId)
    - Remapping: Maps source model HRUs to mizuRoute HRUs when grids differ

Key Components:
    MizuRoutePreProcessor: Network topology setup, remapping file generation
    MizuRouteRunner: Model execution with routing scheme selection
    MizuRouteConfigMixin: Configuration access helpers for coupled models

Configuration Parameters:
    SETTINGS_MIZU_TOPOLOGY: Path to network topology NetCDF file
    SETTINGS_MIZU_WITHIN_BASIN: Within-basin routing option
    SETTINGS_MIZU_NEEDS_REMAP: Whether HRU remapping is required
    SETTINGS_MIZU_OUTPUT_VARS: Variables to output (streamflow, etc.)
    MIZU_FROM_MODEL: Source model for runoff (SUMMA, FUSE, GR, HYPE, etc.)

Typical Workflow:
    1. Generate river network topology from stream shapefile
    2. Create HRU-to-segment mapping
    3. Generate remapping file if source model uses different spatial units
    4. Configure routing scheme (IRF recommended for large domains)
    5. Run source hydrological model to generate runoff
    6. Execute mizuRoute via MizuRouteRunner
    7. Extract routed streamflow at gauge locations

Integration Patterns:
    - Coupled: Run as ROUTING_MODEL after HYDROLOGICAL_MODEL
    - Standalone: Route pre-computed runoff files
    - Multi-model: Route outputs from multiple hydrological models

Limitations and Considerations:
    - Network topology must be prepared in advance (river segments, connectivity)
    - Remapping adds preprocessing complexity when model grids differ
    - KWT and DW schemes are slower but more accurate than IRF
    - Lake/reservoir routing requires additional configuration
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .mixins import MizuRouteConfigMixin

# Lazy import mapping — execution and extraction classes pull the
# geospatial/netCDF stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'MizuRoutePreProcessor': ('.preprocessor', 'MizuRoutePreProcessor'),
    'MizuRouteRunner': ('.runner', 'MizuRouteRunner'),
    'MizuRouteResultExtractor': ('.extractor', 'MizuRouteResultExtractor'),
}


def __getattr__(name: str):
    """Lazy import handler for mizuRoute module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        list(_LAZY_IMPORTS.keys()) + ['MizuRouteConfigMixin', 'MizuRouteConfigAdapter']
    )


__all__ = [
    'MizuRoutePreProcessor',
    'MizuRouteRunner',
    'MizuRouteConfigMixin',
]

# Register all mizuRoute components via unified registry
from symfluence.core.registry import model_manifest

from .config import MizuRouteConfigAdapter


def register() -> None:
    """Register MIZUROUTE components with the unified registry.

    Execution and extraction classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "MIZUROUTE",
        config_adapter=MizuRouteConfigAdapter,
        build_instructions_module="symfluence.models.mizuroute.build_instructions",
    )
    base = 'symfluence.models.mizuroute'
    R.preprocessors.add_lazy("MIZUROUTE", f"{base}.preprocessor.MizuRoutePreProcessor")
    R.base_settings.add("MIZUROUTE", "symfluence.models.mizuroute")
    R.runners.add_lazy("MIZUROUTE", f"{base}.runner.MizuRouteRunner", runner_method='run_mizuroute')
    R.result_extractors.add_lazy("MIZUROUTE", f"{base}.extractor.MizuRouteResultExtractor")
    # Calibration bounds are owned by this package (service-decomposition
    # item 2): registering here means plugin discovery is what makes them
    # servable, so a bound change never needs a core release.
    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .extractor import MizuRouteResultExtractor
    from .preprocessor import MizuRoutePreProcessor
    from .runner import MizuRouteRunner
