# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NGEN (Next Generation Water Resources Modeling Framework).

This module implements integration with NOAA's NextGen framework, a modular
hydrological modeling system built on the Basic Model Interface (BMI) standard.
NGEN enables flexible coupling of different model formulations for rainfall-runoff,
evapotranspiration, snow, and routing processes.

Model Architecture:
    NGEN uses a plug-and-play architecture where BMI-compliant modules are coupled:

    1. **Rainfall-Runoff Modules**:
       - CFE (Conceptual Functional Equivalent): Simplified NWM conceptual model
       - TOPMODEL: Topography-based saturated area model
       - LSTM: Neural network surrogate (via external coupling)

    2. **Land Surface Modules**:
       - Noah-OWP-M: Noah land surface model adapted for OWP (Office of Water Prediction)
       - Includes soil heat, soil moisture, snow, and canopy processes

    3. **Evapotranspiration Modules**:
       - PET: Multiple PET formulations (Penman-Monteith, Priestley-Taylor, etc.)

    4. **Routing**: Internal NGEN routing or external coupling to routing models

Design Rationale:
    NGEN addresses the need for flexible, modular water prediction:
    - BMI standard enables swapping modules without code changes
    - Supports multi-scale modeling from catchment to continental
    - Foundation for NOAA's next-generation National Water Model
    - Enables hybrid physics-ML approaches via BMI

Spatial Structure:
    - Catchments: Hydrologic units defined by hydrofabric (typically NHDPlus-based)
    - Nexuses: Connection points between catchments for routing
    - Realization: Configuration defining which modules run where

Key Components:
    NgenPreProcessor: Hydrofabric processing, forcing preparation
    NgenConfigGenerator: Generates module configs (CFE, PET, Noah) and realization JSON
    NgenRunner: Model execution with catchment parallelization
    NgenPostProcessor: Output aggregation and result extraction

Configuration Parameters:
    NGEN_MODULES_TO_CALIBRATE: Which modules to calibrate (default: 'CFE')
    NGEN_CFE_PARAMS_TO_CALIBRATE: CFE parameters
        (default: 'maxsmc,satdk,bb,slop')
        maxsmc: Maximum soil moisture content
        satdk: Saturated hydraulic conductivity
        bb: Soil pore size distribution index
        slop: Slope of the water table
    NGEN_NOAH_PARAMS_TO_CALIBRATE: Noah parameters
        (default: 'refkdt,slope,smcmax,dksat')
    NGEN_PET_PARAMS_TO_CALIBRATE: PET parameters
        (default: 'wind_speed_measurement_height_m')
    NGEN_ACTIVE_CATCHMENT_ID: Specific catchment for single-catchment runs

Typical Workflow:
    1. Initialize NgenPreProcessor with configuration and hydrofabric
    2. Generate module configurations via NgenConfigGenerator
    3. Create realization JSON defining module coupling
    4. Prepare forcing data in NGEN-compatible format
    5. Execute NGEN via NgenRunner
    6. Extract and aggregate results via NgenPostProcessor

Limitations and Considerations:
    - Requires NGEN executable and BMI module libraries
    - Hydrofabric (catchment/nexus network) must be pre-generated
    - CFE is simplified; full NWM fidelity requires Noah-OWP-M
    - Multi-catchment runs benefit from parallel execution
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution, extraction, and plotting classes pull the
# geospatial/matplotlib stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'NgenConfigGenerator': ('.config_generator', 'NgenConfigGenerator'),
    'NgenPostProcessor': ('.postprocessor', 'NgenPostProcessor'),
    'NgenPreProcessor': ('.preprocessor', 'NgenPreProcessor'),
    'NgenRunner': ('.runner', 'NgenRunner'),
    'visualize_ngen': ('.visualizer', 'visualize_ngen'),
    'NGENResultExtractor': ('.extractor', 'NGENResultExtractor'),
    'NGENPlotter': ('.plotter', 'NGENPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for NGEN module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['NgenConfigAdapter'])


__all__ = [
    'NgenPreProcessor',
    'NgenRunner',
    'NgenPostProcessor',
    'NgenConfigGenerator',
    'visualize_ngen'
]

# Register all NGEN components via unified registry
from symfluence.core.registry import model_manifest

from .config import NgenConfigAdapter


def register() -> None:
    """Register NGEN components with the unified registry.

    Execution, extraction, and plotting classes are registered lazily —
    imported on first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "NGEN",
        config_adapter=NgenConfigAdapter,
        build_instructions_module="symfluence.models.ngen.build_instructions",
    )
    base = 'symfluence.models.ngen'
    R.preprocessors.add_lazy("NGEN", f"{base}.preprocessor.NgenPreProcessor")
    R.runners.add_lazy("NGEN", f"{base}.runner.NgenRunner", runner_method='run_ngen')
    R.postprocessors.add_lazy("NGEN", f"{base}.postprocessor.NgenPostProcessor")
    R.visualizers.add_lazy("NGEN", f"{base}.visualizer.visualize_ngen")
    R.result_extractors.add_lazy("NGEN", f"{base}.extractor.NGENResultExtractor")
    R.plotters.add_lazy("NGEN", f"{base}.plotter.NGENPlotter")

    # Spatial capabilities are owned by this package (service-decomposition
    # item 2): declared at plugin-discovery time so core carries no per-model
    # spatial knowledge and a capability change never needs a core release.
    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "NGEN",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            requires_routing={
                SpatialMode.DISTRIBUTED: True,  # Uses t-route for routing
                SpatialMode.SEMI_DISTRIBUTED: True,
                SpatialMode.LUMPED: False,
            },
            warning_message=None,
        ),
    )

    # Calibration bounds are owned by this package (service-decomposition
    # item 2): registering here means plugin discovery is what makes them
    # servable, so a bound change never needs a core release.
    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .config_generator import NgenConfigGenerator
    from .extractor import NGENResultExtractor
    from .plotter import NGENPlotter
    from .postprocessor import NgenPostProcessor
    from .preprocessor import NgenPreProcessor
    from .runner import NgenRunner
    from .visualizer import visualize_ngen
