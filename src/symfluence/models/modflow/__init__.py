"""MODFLOW 6 (USGS Modular Groundwater Flow Model) Integration.

This module implements MODFLOW 6 support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install modflow`
- Preprocessing (generates MODFLOW 6 text input files)
- Model execution (standalone single-cell lumped model)
- Result extraction (head, drain discharge)
- SUMMA → MODFLOW coupling (recharge → baseflow)
- Postprocessing (combined surface + subsurface flow)

MODFLOW 6 is the USGS modular groundwater flow model. In SYMFLUENCE
it is used as a lumped single-cell groundwater box coupled with land
surface models to produce physically-based baseflow separation.

Configuration Parameters:
    MODFLOW_INSTALL_PATH: Path to MODFLOW 6 installation
    MODFLOW_EXE: Executable name (default: mf6)
    MODFLOW_K: Hydraulic conductivity (m/d)
    MODFLOW_SY: Specific yield
    MODFLOW_TOP/BOT: Aquifer top/bottom elevation (m)
    MODFLOW_DRAIN_ELEVATION: Drain outlet elevation (m)
    MODFLOW_DRAIN_CONDUCTANCE: Drain conductance (m2/d)
    MODFLOW_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Langevin, C.D., et al. (2017): Documentation for the MODFLOW 6
    Groundwater Flow Model. USGS Techniques and Methods 6-A55.

    https://github.com/MODFLOW-ORG/modflow6
"""
from .preprocessor import MODFLOWPreProcessor
from .runner import MODFLOWRunner
from .extractor import MODFLOWResultExtractor
from .postprocessor import MODFLOWPostProcessor
from .config import MODFLOWConfigAdapter
from .plotter import MODFLOWPlotter

__all__ = [
    "MODFLOWPreProcessor",
    "MODFLOWRunner",
    "MODFLOWResultExtractor",
    "MODFLOWPostProcessor",
    "MODFLOWConfigAdapter",
    "MODFLOWPlotter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Components are registered via decorators in their respective modules:
# - MODFLOWPreProcessor: @ModelRegistry.register_preprocessor("MODFLOW")
# - MODFLOWRunner: @ModelRegistry.register_runner("MODFLOW")
# - MODFLOWResultExtractor: @ModelRegistry.register_result_extractor("MODFLOW")
# - MODFLOWPostProcessor: @ModelRegistry.register_postprocessor("MODFLOW")
# MODFLOWConfigAdapter needs explicit registration (no decorator)
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('MODFLOW')(MODFLOWConfigAdapter)
