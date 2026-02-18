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
from .preprocessor import WATFLOODPreProcessor
from .runner import WATFLOODRunner
from .extractor import WATFLOODResultExtractor
from .postprocessor import WATFLOODPostProcessor
from .config import WATFLOODConfigAdapter

__all__ = [
    "WATFLOODPreProcessor",
    "WATFLOODRunner",
    "WATFLOODResultExtractor",
    "WATFLOODPostProcessor",
    "WATFLOODConfigAdapter",
]

# Register build instructions
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass

# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

ModelRegistry.register_preprocessor('WATFLOOD')(WATFLOODPreProcessor)
ModelRegistry.register_runner('WATFLOOD')(WATFLOODRunner)
ModelRegistry.register_result_extractor('WATFLOOD')(WATFLOODResultExtractor)
ModelRegistry.register_config_adapter('WATFLOOD')(WATFLOODConfigAdapter)

# Register calibration components
try:
    from .calibration import WATFLOODModelOptimizer  # noqa: F401
except ImportError:
    pass
