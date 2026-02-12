"""
Evaluation module for SYMFLUENCE model performance assessment.

This package provides tools for evaluating hydrological model outputs against
observational data, including streamflow, snow, soil moisture, and other
environmental variables.

Key components:
    EvaluationRegistry: Central registry for evaluation configurations
    AnalysisRegistry: Registry for analysis types and methods
    BaseStructureEnsembleAnalyzer: Multi-model ensemble analysis
    OutputFileLocator: Utility for locating model output files

Example:
    >>> from symfluence.evaluation import EvaluationRegistry
    >>> registry = EvaluationRegistry()
    >>> registry.register_evaluator('streamflow', streamflow_evaluator)
"""
from .registry import EvaluationRegistry
from .analysis_registry import AnalysisRegistry
from . import evaluators
from .structure_ensemble import BaseStructureEnsembleAnalyzer
from .output_file_locator import OutputFileLocator, get_output_file_locator
from .metric_transformer import MetricTransformer

__all__ = [
    "EvaluationRegistry",
    "AnalysisRegistry",
    "evaluators",
    "BaseStructureEnsembleAnalyzer",
    "OutputFileLocator",
    "get_output_file_locator",
    "MetricTransformer",
]
