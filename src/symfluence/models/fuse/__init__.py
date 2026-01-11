"""
FUSE model module.

This module provides a clean, modular structure for the FUSE (Framework for
Understanding Structural Errors) hydrological model integration.

The module has been refactored from a single large file (fuse_utils.py, 3094 lines)
into separate components for better maintainability and organization.

Main Components:
    - FUSEPreProcessor: Handles FUSE preprocessing and setup (preprocessor.py, ~350 lines)
    - FUSERunner: Executes FUSE model simulations (runner.py, 1194 lines)
    - FUSEPostprocessor: Processes FUSE model outputs (postprocessor.py, 95 lines)
    - FuseStructureAnalyzer: Analyzes FUSE model structure ensembles (structure_analyzer.py)

Manager Classes (Refactored 2026-01-01):
    - FuseForcingProcessor: Handles forcing data transformations (forcing_processor.py, ~350 lines)
    - FuseElevationBandManager: Manages elevation band creation (elevation_band_manager.py, ~290 lines)
    - FuseSyntheticDataGenerator: Generates synthetic hydrographs (synthetic_data_generator.py, ~170 lines)

Refactoring History:
    - 2025-12-31: Initial refactoring from fuse_utils.py
    - 2026-01-01: Manager extraction and modularization (reduced preprocessor from 1441 → 350 lines)

For backward compatibility, all public classes are re-exported at the package level.
"""

# Import main classes
from .preprocessor import FUSEPreProcessor
from .runner import FUSERunner
from .postprocessor import FUSEPostprocessor
from .structure_analyzer import FuseStructureAnalyzer
from .visualizer import visualize_fuse

# Import manager classes (for advanced usage)
from .forcing_processor import FuseForcingProcessor
from .elevation_band_manager import FuseElevationBandManager
from .synthetic_data_generator import FuseSyntheticDataGenerator

__all__ = [
    # Main classes (public API)
    'FUSEPreProcessor',
    'FUSERunner',
    'FUSEPostprocessor',
    'FuseStructureAnalyzer',
    # Manager classes (advanced usage)
    'FuseForcingProcessor',
    'FuseElevationBandManager',
    'FuseSyntheticDataGenerator',
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Register analysis components with AnalysisRegistry
from symfluence.evaluation.analysis_registry import AnalysisRegistry

# Register FUSE decision analyzer (structure ensemble analysis)
AnalysisRegistry.register_decision_analyzer('FUSE')(FuseStructureAnalyzer)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import FUSEPlotter  # noqa: F401
