"""
MESH (Modélisation Environmentale Communautaire - Surface and Hydrology) package.

This package contains components for running and managing MESH model simulations.
"""

from .preprocessor import MESHPreProcessor
from .runner import MESHRunner
from .postprocessor import MESHPostProcessor
from .visualizer import visualize_mesh

__all__ = [
    'MESHPreProcessor',
    'MESHRunner',
    'MESHPostProcessor',
    'visualize_mesh'
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
from .config import MESHConfigAdapter
ModelRegistry.register_config_adapter('MESH')(MESHConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import MESHResultExtractor
ModelRegistry.register_result_extractor('MESH')(MESHResultExtractor)
