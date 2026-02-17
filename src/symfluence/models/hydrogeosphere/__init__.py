"""HydroGeoSphere (HGS) Integration.

This module implements HydroGeoSphere support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install hydrogeosphere`
- Preprocessing (generates HGS input files for lumped 1x1 grid)
- Model execution (grok + hgs two-step)
- Result extraction (hydrograph, head, water balance)
- SUMMA → HGS coupling (recharge → baseflow)
- Postprocessing (combined surface + subsurface flow)

HydroGeoSphere is a control-volume finite-element, fully-coupled 3D
variably-saturated subsurface + 2D overland flow + 1D channel flow
model. Commercial code from Aquanty with research licenses available.

Configuration Parameters:
    HGS_INSTALL_PATH: Path to HGS installation
    HGS_EXE: HGS solver executable (default: hgs)
    HGS_GROK_EXE: Grok preprocessor executable (default: grok)
    HGS_K_SAT: Saturated hydraulic conductivity (m/s)
    HGS_POROSITY: Total porosity
    HGS_VG_ALPHA: van Genuchten alpha (1/m)
    HGS_VG_N: van Genuchten n
    HGS_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Therrien, R., et al. (2010): HydroGeoSphere — A Three-dimensional
    Numerical Model Describing Fully-integrated Subsurface and Surface
    Flow and Solute Transport. Groundwater Simulations Group.

    https://www.aquanty.com/hydrogeosphere
"""
from .preprocessor import HGSPreProcessor
from .runner import HGSRunner
from .extractor import HGSResultExtractor
from .postprocessor import HGSPostProcessor
from .config import HGSConfigAdapter
from .plotter import HGSPlotter

__all__ = [
    "HGSPreProcessor",
    "HGSRunner",
    "HGSResultExtractor",
    "HGSPostProcessor",
    "HGSConfigAdapter",
    "HGSPlotter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Components are registered via decorators in their respective modules:
# - HGSPreProcessor: @ModelRegistry.register_preprocessor("HYDROGEOSPHERE")
# - HGSRunner: @ModelRegistry.register_runner("HYDROGEOSPHERE")
# - HGSResultExtractor: @ModelRegistry.register_result_extractor("HYDROGEOSPHERE")
# - HGSPostProcessor: @ModelRegistry.register_postprocessor("HYDROGEOSPHERE")
# HGSConfigAdapter needs explicit registration (no decorator)
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('HYDROGEOSPHERE')(HGSConfigAdapter)
