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
    NgenPostprocessor: Output aggregation and result extraction

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
    6. Extract and aggregate results via NgenPostprocessor

Limitations and Considerations:
    - Requires NGEN executable and BMI module libraries
    - Hydrofabric (catchment/nexus network) must be pre-generated
    - CFE is simplified; full NWM fidelity requires Noah-OWP-M
    - Multi-catchment runs benefit from parallel execution
"""

from .config_generator import NgenConfigGenerator
from .postprocessor import NgenPostprocessor
from .preprocessor import NgenPreProcessor
from .runner import NgenRunner
from .visualizer import visualize_ngen

__all__ = [
    'NgenPreProcessor',
    'NgenRunner',
    'NgenPostprocessor',
    'NgenConfigGenerator',
    'visualize_ngen'
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry

from .config import NgenConfigAdapter

ModelRegistry.register_config_adapter('NGEN')(NgenConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import NGENResultExtractor

ModelRegistry.register_result_extractor('NGEN')(NGENResultExtractor)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import NGENPlotter  # noqa: F401
