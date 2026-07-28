# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MESH (Modélisation Environmentale - Surface and Hydrology) Model.

This module implements MESH, a coupled land surface-hydrology model developed
by Environment and Climate Change Canada. MESH combines the Canadian Land
Surface Scheme (CLASS) or SVS with WATFLOOD routing for operational
hydrological prediction, particularly in cold regions.

Model Architecture:
    MESH couples two main components:

    1. **Land Surface Scheme** (CLASS or SVS):
       - Energy balance: radiation, sensible/latent heat fluxes
       - Snow processes: multi-layer snow, metamorphism, melt
       - Soil processes: heat conduction, moisture dynamics, freeze-thaw
       - Vegetation: phenology, transpiration, interception

    2. **Routing Component** (WATFLOOD-derived):
       - Overland flow routing
       - Channel routing with Manning's equation
       - Lake and wetland storage
       - Gridded or GRU-based spatial structure

Design Rationale:
    MESH addresses Canadian operational needs:
    - Process-based for scenario analysis (climate, land use)
    - Cold region processes (permafrost, snow redistribution, ice)
    - Operational use in Canadian flood forecasting
    - Energy balance critical for snowmelt timing

Spatial Structure:
    - GRUs (Grouped Response Units): Tiles with similar hydrological response
    - Tiles: Land cover types within each GRU
    - Grid: Optional regular grid for spatially distributed simulations

Key Components:
    MESHPreProcessor: DDB preparation, forcing setup, parameter files
    MESHRunner: Model execution and simulation management
    MESHPostProcessor: Output extraction and result formatting

Configuration Parameters:
    MESH_SPATIAL_MODE: Spatial setup ('auto', 'lumped', 'distributed')
    MESH_FORCING_PATH: Path to forcing data files
    MESH_FORCING_VARS: Forcing variable names
    MESH_FORCING_UNITS: Forcing variable units

Typical Workflow:
    1. Prepare drainage database (DDB) with GRU/tile definitions
    2. Process forcing data (hourly or sub-hourly for energy balance)
    3. Set up CLASS/SVS parameters and initial conditions
    4. Configure WATFLOOD routing parameters
    5. Execute MESH via MESHRunner
    6. Extract results (streamflow, SWE, soil moisture, energy fluxes)

Limitations and Considerations:
    - Requires MESH executable (compiled from source)
    - CLASS/SVS have different parameter requirements
    - Energy balance requires radiation and wind data (not just P/T)
    - Canadian datasets (CaPA, RDRS) well-supported
    - DDB preparation can be complex for new domains
    - Primarily tested for Canadian applications
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and extraction classes pull the
# geospatial/observation stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'MESHPostProcessor': ('.postprocessor', 'MESHPostProcessor'),
    'MESHPreProcessor': ('.preprocessor', 'MESHPreProcessor'),
    'MESHRunner': ('.runner', 'MESHRunner'),
    'visualize_mesh': ('.visualizer', 'visualize_mesh'),
    'MESHResultExtractor': ('.extractor', 'MESHResultExtractor'),
}


def __getattr__(name: str):
    """Lazy import handler for MESH module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['MESHConfigAdapter'])


__all__ = [
    'MESHPreProcessor',
    'MESHRunner',
    'MESHPostProcessor',
    'visualize_mesh'
]

# Register all MESH components via unified registry
from symfluence.core.registry import model_manifest

from .config import MESHConfigAdapter


def register() -> None:
    """Register MESH components with the unified registry.

    Execution and extraction classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "MESH",
        config_adapter=MESHConfigAdapter,
        build_instructions_module="symfluence.models.mesh.build_instructions",
    )
    base = 'symfluence.models.mesh'
    R.preprocessors.add_lazy("MESH", f"{base}.preprocessor.MESHPreProcessor")
    R.base_settings.add("MESH", "symfluence.models.mesh")
    R.runners.add_lazy("MESH", f"{base}.runner.MESHRunner", runner_method='run_mesh')
    R.postprocessors.add_lazy("MESH", f"{base}.postprocessor.MESHPostProcessor")
    R.visualizers.add_lazy("MESH", f"{base}.visualizer.visualize_mesh")
    R.result_extractors.add_lazy("MESH", f"{base}.extractor.MESHResultExtractor")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "MESH",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            requires_routing={
                # MESH has internal routing (WATFLOOD/PDMROF).
                SpatialMode.DISTRIBUTED: False,
                SpatialMode.SEMI_DISTRIBUTED: False,
                # Lumped uses noroute mode (RFF+DRAINSOL proxy).
                SpatialMode.LUMPED: False,
            },
            warning_message=None,  # Lumped mode fully supported
        ),
    )

    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .extractor import MESHResultExtractor
    from .postprocessor import MESHPostProcessor
    from .preprocessor import MESHPreProcessor
    from .runner import MESHRunner
    from .visualizer import visualize_mesh
