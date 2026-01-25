#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SCE-UA (Shuffled Complex Evolution - University of Arizona) Algorithm

A global optimization algorithm that combines the strengths of the simplex
procedure with competitive evolution. Widely used in hydrological modeling.

Reference:
    Duan, Q., Sorooshian, S., and Gupta, V.K. (1992). Effective and Efficient
    Global Optimization for Conceptual Rainfall-Runoff Models.
    Water Resources Research, 28(4), 1015-1031.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class SCEUAAlgorithm(OptimizationAlgorithm):
    """Shuffled Complex Evolution algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "SCE-UA"

    def optimize(  # type: ignore[override]
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        evaluate_population_objectives: Optional[Callable] = None,
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run SCE-UA optimization.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            log_initial_population: Optional callback to log initial population
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        self.logger.info(f"Starting SCE-UA optimization with {n_params} parameters")

        # SCE-UA parameters
        n_complexes = max(2, self.population_size // 10)
        n_per_complex = 2 * n_params + 1
        pop_size = n_complexes * n_per_complex

        # Initialize population
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        population = np.random.uniform(0, 1, (pop_size, n_params))
        fitness = evaluate_population(population, 0)

        # Sort by fitness (descending for maximization)
        sorted_idx = np.argsort(-fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        best_pos = population[0].copy()
        best_fit = fitness[0]

        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # SCE-UA main loop
        for iteration in range(1, self.max_iterations + 1):
            # Partition into complexes
            for complex_idx in range(n_complexes):
                complex_members = list(range(complex_idx, pop_size, n_complexes))
                sub_complex = population[complex_members]
                sub_fitness = fitness[complex_members]

                # Evolve sub-complex (simplified CCE step)
                for _ in range(n_per_complex):
                    # Select simplex
                    simplex_size = n_params + 1
                    simplex_idx = np.random.choice(
                        len(complex_members), simplex_size, replace=False
                    )

                    # Generate new point (reflection)
                    worst_idx = simplex_idx[np.argmin(sub_fitness[simplex_idx])]
                    others = [i for i in simplex_idx if i != worst_idx]
                    centroid = np.mean(sub_complex[others], axis=0)

                    # Reflection
                    new_point = 2 * centroid - sub_complex[worst_idx]
                    new_point = np.clip(new_point, 0, 1)

                    # Evaluate
                    new_fitness = evaluate_solution(new_point, 0)

                    if new_fitness > sub_fitness[worst_idx]:
                        sub_complex[worst_idx] = new_point
                        sub_fitness[worst_idx] = new_fitness

                # Update main population
                population[complex_members] = sub_complex
                fitness[complex_members] = sub_fitness

            # Shuffle and sort
            sorted_idx = np.argsort(-fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update best
            if fitness[0] > best_fit:
                best_pos = population[0].copy()
                best_fit = fitness[0]

            # Record results
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict)
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit)

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos)
        }
