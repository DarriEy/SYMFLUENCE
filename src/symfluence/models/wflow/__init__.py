"""Wflow (wflow_sbm) Distributed Hydrological Model."""
from .config import WflowConfigAdapter
from .extractor import WflowResultExtractor
from .postprocessor import WflowPostProcessor
from .preprocessor import WflowPreProcessor
from .runner import WflowRunner

__all__ = [
    "WflowPreProcessor",
    "WflowRunner",
    "WflowResultExtractor",
    "WflowPostProcessor",
    "WflowConfigAdapter",
]

from symfluence.core.registry import model_manifest

model_manifest(
    "WFLOW",
    preprocessor=WflowPreProcessor,
    runner=WflowRunner,
    result_extractor=WflowResultExtractor,
    config_adapter=WflowConfigAdapter,
    build_instructions_module="symfluence.models.wflow.build_instructions",
)

try:
    from .calibration import WflowModelOptimizer  # noqa: F401
except ImportError:
    pass
