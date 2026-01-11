"""
SUMMA (Structure for Unifying Multiple Modeling Alternatives) package.

This package contains components for running and managing SUMMA model simulations.
"""

from .preprocessor import SummaPreProcessor
from .runner import SummaRunner
from .postprocessor import SUMMAPostprocessor
from .structure_analyzer import SummaStructureAnalyzer
from .visualizer import visualize_summa
from .forcing_processor import SummaForcingProcessor
from .config_manager import SummaConfigManager
from .attributes_manager import SummaAttributesManager

__all__ = [
    'SummaPreProcessor',
    'SummaRunner',
    'SUMMAPostprocessor',
    'SummaStructureAnalyzer',
    'SummaForcingProcessor',
    'SummaConfigManager',
    'SummaAttributesManager'
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Register analysis components with AnalysisRegistry
from symfluence.evaluation.analysis_registry import AnalysisRegistry

# Register SUMMA decision analyzer (structure ensemble analysis)
AnalysisRegistry.register_decision_analyzer('SUMMA')(SummaStructureAnalyzer)

# Register plotter with PlotterRegistry (import triggers registration via decorator)
from .plotter import SUMMAPlotter  # noqa: F401
