"""
NOAA NextGen Framework (ngen) Module.

This package contains components for the NGen model integration:
- Preprocessor: Handles spatial and data preprocessing
- Runner: Manages model execution
- Postprocessor: Handles output extraction and analysis

Refactored from single file to modular structure.
"""

from .preprocessor import NgenPreProcessor
from .runner import NgenRunner
from .postprocessor import NgenPostprocessor

__all__ = [
    'NgenPreProcessor',
    'NgenRunner',
    'NgenPostprocessor',
]
