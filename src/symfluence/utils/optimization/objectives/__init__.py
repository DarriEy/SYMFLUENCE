"""
Objective Functions

Objective function classes that calculate scalar optimization targets from
evaluation results. Objectives can combine multiple metrics and variables
for multi-criteria calibration.

Components:
- BaseObjective: Abstract base class for all objectives
- ObjectiveRegistry: Registry for objective function implementations
- MultivariateObjective: Combines multiple variables into single objective

Usage:
    >>> from symfluence.utils.optimization.objectives import ObjectiveRegistry
    >>> objective = ObjectiveRegistry.get_objective('MULTIVARIATE', config, logger)
    >>> score = objective.calculate(evaluation_results)
"""

from .base import BaseObjective
from .registry import ObjectiveRegistry
from .multivariate import MultivariateObjective

__all__ = [
    'BaseObjective',
    'ObjectiveRegistry',
    'MultivariateObjective',
]
