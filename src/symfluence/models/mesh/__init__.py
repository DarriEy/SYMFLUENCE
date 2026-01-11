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
