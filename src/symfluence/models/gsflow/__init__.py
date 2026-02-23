"""GSFLOW (coupled PRMS + MODFLOW-NWT) Hydrological Model.

GSFLOW is a USGS coupled groundwater-surface-water model that integrates
PRMS (surface/soil processes) with MODFLOW-NWT (saturated zone) via SFR
and UZF packages for bidirectional exchange.

Supports three operation modes:
- PRMS: Surface processes only
- MODFLOW: Groundwater only
- COUPLED: Full bidirectional PRMS↔MODFLOW-NWT exchange (default)

References:
    Markstrom, S.L., et al. (2008): GSFLOW---Coupled Ground-Water and
    Surface-Water Flow Model. USGS Techniques and Methods 6-D1.
"""
from .config import GSFLOWConfigAdapter
from .extractor import GSFLOWResultExtractor
from .postprocessor import GSFLOWPostProcessor
from .preprocessor import GSFLOWPreProcessor
from .runner import GSFLOWRunner

__all__ = [
    "GSFLOWPreProcessor",
    "GSFLOWRunner",
    "GSFLOWResultExtractor",
    "GSFLOWPostProcessor",
    "GSFLOWConfigAdapter",
]

# Register build instructions
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass

# Register components with ModelRegistry
from symfluence.models.registry import ModelRegistry

ModelRegistry.register_preprocessor('GSFLOW')(GSFLOWPreProcessor)
ModelRegistry.register_runner('GSFLOW')(GSFLOWRunner)
ModelRegistry.register_result_extractor('GSFLOW')(GSFLOWResultExtractor)
ModelRegistry.register_config_adapter('GSFLOW')(GSFLOWConfigAdapter)

# Register calibration components
try:
    from .calibration import GSFLOWModelOptimizer  # noqa: F401
except ImportError:
    pass
