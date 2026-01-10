#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Algorithm Interface

Abstract base class for optimization algorithms using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import numpy as np


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms.

    Algorithms receive evaluation callbacks from the optimizer and return
    optimization results. This allows algorithms to be easily swapped
    while maintaining a consistent interface.
    """

    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize the algorithm.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Common algorithm parameters
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.population_size = config.get('POPULATION_SIZE', 30)
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        self.penalty_score = -999.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name (e.g., 'DDS', 'PSO', 'NSGA-II')."""
        pass

    @abstractmethod
    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        evaluate_population_objectives: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the optimization algorithm.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Callback to evaluate a single normalized solution
            evaluate_population: Callback to evaluate a population of normalized solutions
            denormalize_params: Callback to convert normalized params to dict
            record_iteration: Callback to record iteration results
            update_best: Callback to update best solution
            log_progress: Callback to log optimization progress
            evaluate_population_objectives: Optional callback for multi-objective evaluation
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dictionary containing:
                - best_solution: Best normalized solution found
                - best_score: Best fitness score
                - best_params: Best parameters as dictionary
                - history: Optimization history
        """
        pass

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to [0, 1] bounds."""
        return np.clip(x, 0, 1)

    def _reflect_at_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Reflect solutions at bounds instead of clipping.

        This often produces better exploration than simple clipping.
        """
        result = x.copy()
        for i in range(len(result)):
            while result[i] < 0 or result[i] > 1:
                if result[i] < 0:
                    result[i] = -result[i]
                if result[i] > 1:
                    result[i] = 2.0 - result[i]
        return result
