"""
TRoute Model Utilities.

This package contains components for the t-route model integration:
- Preprocessor: Handles spatial and data preprocessing
- Runner: Manages model execution

Refactored from single file to modular structure.
"""

from .preprocessor import TRoutePreProcessor
from .runner import TRouteRunner

__all__ = [
    'TRoutePreProcessor',
    'TRouteRunner',
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional
