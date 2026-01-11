"""
MizuRoute Model Utilities.

This package contains components for the mizuRoute model integration:
- Preprocessor: Handles spatial and data preprocessing
- Runner: Manages model execution
- Mixins: Configuration access helpers

Refactored from single file to modular structure.
"""

from .preprocessor import MizuRoutePreProcessor
from .runner import MizuRouteRunner
from .mixins import MizuRouteConfigMixin

__all__ = [
    'MizuRoutePreProcessor',
    'MizuRouteRunner',
    'MizuRouteConfigMixin',
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional
