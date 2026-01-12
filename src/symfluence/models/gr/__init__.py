"""
GR (GR4J/CemaNeige) hydrological model package.

This package contains components for running and managing GR family models.
Supports both lumped and distributed spatial modes with optional mizuRoute routing.
"""

from .preprocessor import GRPreProcessor
from .runner import GRRunner
from .postprocessor import GRPostprocessor
from .visualizer import visualize_gr

__all__ = [
    'GRPreProcessor',
    'GRRunner',
    'GRPostprocessor',
    'visualize_gr'
]

# Register defaults with DefaultsRegistry (import triggers registration via decorator)
from . import defaults  # noqa: F401

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
from .config import GRConfigAdapter
ModelRegistry.register_config_adapter('GR')(GRConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import GRResultExtractor
ModelRegistry.register_result_extractor('GR')(GRResultExtractor)
