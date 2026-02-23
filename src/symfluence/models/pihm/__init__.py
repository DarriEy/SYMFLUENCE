"""PIHM (Penn State Integrated Hydrologic Model) Integration.

This module implements PIHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install pihm`
- Preprocessing (generates PIHM input files for lumped single-element mesh)
- Model execution (standalone via mm-pihm)
- Result extraction (river flux, groundwater head)
- SUMMA → PIHM coupling (recharge → baseflow)
- Postprocessing (combined surface + subsurface flow)

PIHM is a finite-volume, unstructured-mesh, fully-coupled
surface-subsurface model solving Richards equation + diffusion wave
overland flow + 1D channel routing. Uses SUNDIALS CVODE solver.

Configuration Parameters:
    PIHM_INSTALL_PATH: Path to PIHM installation
    PIHM_EXE: Executable name (default: pihm)
    PIHM_K_SAT: Saturated hydraulic conductivity (m/s)
    PIHM_POROSITY: Total porosity
    PIHM_VG_ALPHA: van Genuchten alpha (1/m)
    PIHM_VG_N: van Genuchten n
    PIHM_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Qu, Y. & Duffy, C.J. (2007): A semidiscrete finite volume
    formulation for multiprocess watershed simulation.
    Water Resources Research 43(8).

    https://github.com/PSUmodeling/MM-PIHM
"""
from .config import PIHMConfigAdapter
from .extractor import PIHMResultExtractor
from .plotter import PIHMPlotter
from .postprocessor import PIHMPostProcessor
from .preprocessor import PIHMPreProcessor
from .runner import PIHMRunner

__all__ = [
    "PIHMPreProcessor",
    "PIHMRunner",
    "PIHMResultExtractor",
    "PIHMPostProcessor",
    "PIHMConfigAdapter",
    "PIHMPlotter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Components are registered via decorators in their respective modules:
# - PIHMPreProcessor: @ModelRegistry.register_preprocessor("PIHM")
# - PIHMRunner: @ModelRegistry.register_runner("PIHM")
# - PIHMResultExtractor: @ModelRegistry.register_result_extractor("PIHM")
# - PIHMPostProcessor: @ModelRegistry.register_postprocessor("PIHM")
# PIHMConfigAdapter needs explicit registration (no decorator)
from symfluence.models.registry import ModelRegistry

ModelRegistry.register_config_adapter('PIHM')(PIHMConfigAdapter)
