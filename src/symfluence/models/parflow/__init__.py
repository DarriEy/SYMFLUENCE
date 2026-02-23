"""ParFlow Integrated Hydrologic Model Integration.

This module implements ParFlow support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install parflow`
- Preprocessing (generates ParFlow .pfidb run files via pftools API)
- Model execution (standalone variably-saturated + overland flow)
- Result extraction (pressure head, saturation, overland flow from .pfb)
- SUMMA -> ParFlow coupling (recharge -> subsurface flow)
- Postprocessing (combined surface + subsurface flow)

ParFlow is a parallel integrated hydrologic model that solves
variably-saturated flow (Richards equation) and overland flow.
In SYMFLUENCE it is used as an alternative to MODFLOW for coupled
land surface + groundwater simulations with full vadose zone support.

Configuration Parameters:
    PARFLOW_INSTALL_PATH: Path to ParFlow installation
    PARFLOW_EXE: Executable name (default: parflow)
    PARFLOW_DIR: ParFlow install root (sets PARFLOW_DIR env var)
    PARFLOW_K_SAT: Saturated hydraulic conductivity (m/hr)
    PARFLOW_POROSITY: Porosity (dimensionless)
    PARFLOW_VG_ALPHA: van Genuchten alpha (1/m)
    PARFLOW_VG_N: van Genuchten n (dimensionless, > 1)
    PARFLOW_TOP/BOT: Domain top/bottom elevation (m)
    PARFLOW_MANNINGS_N: Manning's roughness for overland flow
    PARFLOW_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Kollet, S.J. & Maxwell, R.M. (2006): Integrated surface-groundwater
    flow modeling: A free-surface overland flow boundary condition in a
    parallel groundwater flow model. Advances in Water Resources 29(7).

    https://github.com/parflow/parflow
"""
from .config import ParFlowConfigAdapter
from .extractor import ParFlowResultExtractor
from .plotter import ParFlowPlotter
from .postprocessor import ParFlowPostProcessor
from .preprocessor import ParFlowPreProcessor
from .runner import ParFlowRunner

__all__ = [
    "ParFlowPreProcessor",
    "ParFlowRunner",
    "ParFlowResultExtractor",
    "ParFlowPostProcessor",
    "ParFlowConfigAdapter",
    "ParFlowPlotter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Components are registered via decorators in their respective modules:
# - ParFlowPreProcessor: @ModelRegistry.register_preprocessor("PARFLOW")
# - ParFlowRunner: @ModelRegistry.register_runner("PARFLOW")
# - ParFlowResultExtractor: @ModelRegistry.register_result_extractor("PARFLOW")
# - ParFlowPostProcessor: @ModelRegistry.register_postprocessor("PARFLOW")
# ParFlowConfigAdapter needs explicit registration (no decorator)
from symfluence.models.registry import ModelRegistry

ModelRegistry.register_config_adapter('PARFLOW')(ParFlowConfigAdapter)
