#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSO (Particle Swarm Optimization) Algorithm

A population-based metaheuristic that simulates social behavior of bird
flocking or fish schooling. Particles move through the search space guided
by their own best positions and the swarm's best position.

Reference:
    Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class PSOAlgorithm(OptimizationAlgorithm):
    """Particle Swarm Optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "PSO"

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
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run PSO optimization.

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
        self.logger.info(f"Starting PSO optimization with {n_params} parameters")

        n_particles = self.population_size

        # PSO parameters
        w = self.config.get('PSO_INERTIA', 0.7)  # Inertia weight
        c1 = self.config.get('PSO_COGNITIVE', 1.5)  # Cognitive coefficient
        c2 = self.config.get('PSO_SOCIAL', 1.5)  # Social coefficient
        v_max = self.config.get('PSO_V_MAX', 0.2)  # Maximum velocity

        # Initialize swarm
        self.logger.info(f"Evaluating initial population ({n_particles} particles)...")
        positions = np.random.uniform(0, 1, (n_particles, n_params))
        velocities = np.random.uniform(-v_max, v_max, (n_particles, n_params))

        # Evaluate initial population
        fitness = evaluate_population(positions, 0)

        # Initialize personal and global bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        global_best_idx = np.argmax(fitness)
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = fitness[global_best_idx]

        # Record initial best
        params_dict = denormalize_params(global_best_pos)
        record_iteration(0, global_best_fit, params_dict)
        update_best(global_best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_particles, global_best_fit)

        # PSO main loop
        for iteration in range(1, self.max_iterations + 1):
            # Update velocities
            r1 = np.random.random((n_particles, n_params))
            r2 = np.random.random((n_particles, n_params))

            cognitive = c1 * r1 * (personal_best_pos - positions)
            social = c2 * r2 * (global_best_pos - positions)

            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, 0, 1)

            # Evaluate
            fitness = evaluate_population(positions, iteration)

            # Update personal bests
            improved = fitness > personal_best_fit
            n_improved = np.sum(improved)
            personal_best_pos[improved] = positions[improved]
            personal_best_fit[improved] = fitness[improved]

            # Update global best
            if np.max(fitness) > global_best_fit:
                global_best_idx = np.argmax(fitness)
                global_best_pos = positions[global_best_idx].copy()
                global_best_fit = fitness[global_best_idx]

            # Record results
            params_dict = denormalize_params(global_best_pos)
            record_iteration(iteration, global_best_fit, params_dict)
            update_best(global_best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, global_best_fit, n_improved, n_particles)

        return {
            'best_solution': global_best_pos,
            'best_score': global_best_fit,
            'best_params': denormalize_params(global_best_pos)
        }
