"""
Base Evaluator for SYMFLUENCE

Note: This module is maintained for backward compatibility. 
New evaluators should inherit from symfluence.evaluation.evaluators.ModelEvaluator.
"""

from .evaluators import ModelEvaluator as BaseEvaluator

__all__ = ["BaseEvaluator"]
