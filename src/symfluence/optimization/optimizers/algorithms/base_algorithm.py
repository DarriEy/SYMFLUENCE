#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Algorithm Interface

Abstract base class for optimization algorithms using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, Union, TYPE_CHECKING
import numpy as np

from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class OptimizationAlgorithm(ConfigMixin, ABC):
    """
    Abstract base class for optimization algorithms.

    Algorithms receive evaluation callbacks from the optimizer and return
    optimization results. This allows algorithms to be easily swapped
    while maintaining a consistent interface.
    """

    def __init__(self, config: Union['SymfluenceConfig', Dict[str, Any]], logger):
        """
        Initialize the algorithm.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
        """
        # Import here to avoid circular imports
        from symfluence.core.config.models import SymfluenceConfig

        # Auto-convert dict to typed config for backward compatibility
        if isinstance(config, dict):
            try:
                self._config = SymfluenceConfig(**config)
            except Exception:
                # Fallback for partial configs (e.g., in tests)
                self._config = config
        else:
            self._config = config

        self.logger = logger

        # Common algorithm parameters - use _get_config_value for typed access
        self.max_iterations = self._get_config_value(
            lambda: self.config.optimization.iterations,
            default=100,
            dict_key='NUMBER_OF_ITERATIONS'
        )
        self.population_size = self._get_config_value(
            lambda: self.config.optimization.population_size,
            default=30,
            dict_key='POPULATION_SIZE'
        )
        self.target_metric = self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )
        self.penalty_score = ModelDefaults.PENALTY_SCORE

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
