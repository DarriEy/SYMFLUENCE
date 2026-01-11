"""
NOAA NextGen Framework (ngen) Module.

This package contains components for the NGen model integration:
- Preprocessor: Handles spatial and data preprocessing
- ConfigGenerator: Modular config file generation (CFE, PET, NOAH, realization)
- Runner: Manages model execution
- Postprocessor: Handles output extraction and analysis

Refactored from single file to modular structure.
"""

from .preprocessor import NgenPreProcessor
from .runner import NgenRunner
from .postprocessor import NgenPostprocessor
from .config_generator import NgenConfigGenerator
from .visualizer import visualize_ngen

__all__ = [
    'NgenPreProcessor',
    'NgenRunner',
    'NgenPostprocessor',
    'NgenConfigGenerator',
    'visualize_ngen'
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import NGENPlotter  # noqa: F401
