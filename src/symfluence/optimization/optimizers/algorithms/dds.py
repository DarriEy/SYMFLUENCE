#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDS (Dynamically Dimensioned Search) Algorithm

A simple and effective algorithm for calibrating computationally expensive
hydrological models. DDS progressively focuses the search from global to local
as iterations progress.

Reference:
    Tolson, B.A. and Shoemaker, C.A. (2007). Dynamically dimensioned search
    algorithm for computationally efficient watershed model calibration.
    Water Resources Research, 43(1).
"""

from typing import Dict, Any, Callable, Optional
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class DDSAlgorithm(OptimizationAlgorithm):
    """Dynamically Dimensioned Search optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "DDS"

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
        compute_gradient: Optional[Callable] = None,
        gradient_mode: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run DDS optimization.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        self.logger.info(f"Starting DDS optimization with {n_params} parameters")

        # DDS perturbation range
        r = self._get_config_value(lambda: self.config.optimization.dds.r, default=0.2, dict_key='DDS_R')

        # Initialize with random starting point
        x_best = np.random.uniform(0, 1, n_params)
        f_best = evaluate_solution(x_best, 0)

        # Record initial state
        params_dict = denormalize_params(x_best)
        record_iteration(0, f_best, params_dict)
        update_best(f_best, params_dict, 0)

        # DDS main loop
        for iteration in range(1, self.max_iterations + 1):
            # Calculate probability of perturbation (decreases with iterations)
            p = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            p = max(1.0 / n_params, p)  # Ensure at least one parameter is perturbed

            # Select parameters to perturb
            perturb_mask = np.random.random(n_params) < p

            # Ensure at least one parameter is perturbed
            if not perturb_mask.any():
                perturb_mask[np.random.randint(n_params)] = True

            # Generate candidate solution
            x_new = x_best.copy()

            for i in range(n_params):
                if perturb_mask[i]:
                    perturbation = r * np.random.standard_normal()
                    x_new[i] = x_best[i] + perturbation

                    # Reflect at boundaries
                    if x_new[i] < 0:
                        x_new[i] = -x_new[i]
                    if x_new[i] > 1:
                        x_new[i] = 2 - x_new[i]

                    # Clip to bounds
                    x_new[i] = np.clip(x_new[i], 0, 1)

            # Evaluate candidate
            f_new = evaluate_solution(x_new, 0)

            # Update if better (DDS is greedy)
            if f_new > f_best:
                x_best = x_new
                f_best = f_new

            # Record results
            params_dict = denormalize_params(x_best)
            record_iteration(iteration, f_best, params_dict)
            update_best(f_best, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, f_best)

        return {
            'best_solution': x_best,
            'best_score': f_best,
            'best_params': denormalize_params(x_best)
        }
