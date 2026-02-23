#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSO (Particle Swarm Optimization) Algorithm

A population-based metaheuristic that simulates social behavior of bird
flocking or fish schooling. Particles move through the search space guided
by their own best positions and the swarm's best position.

Reference:
    Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks.

    Shi, Y. and Eberhart, R. (1998). A modified particle swarm optimizer.
    IEEE International Conference on Evolutionary Computation.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import PSODefaults


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
        compute_gradient: Optional[Callable] = None,
        gradient_mode: str = 'auto',
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run PSO optimization.

        PSO velocity update equation (Kennedy & Eberhart 1995):
            v_i(t+1) = w * v_i(t) + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)

        where:
        - w: inertia weight (momentum from previous velocity)
        - c1: cognitive coefficient (attraction to personal best)
        - c2: social coefficient (attraction to global best)
        - r1, r2: random values in [0, 1] for stochastic behavior

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

        # PSO parameters using standardized config access
        # Inertia weight (w): controls influence of previous velocity
        # w=0.7 provides good balance between exploration and exploitation
        # Lower values (0.4) favor exploitation; higher (0.9) favor exploration
        # (Shi & Eberhart 1998, "Inertia Weight Approach")
        w = self._get_config_value(
            lambda: self.config.optimization.pso_inertia,
            default=PSODefaults.INERTIA,
            dict_key='PSO_INERTIA'
        )

        # Cognitive coefficient (c1): attraction to personal best
        # c1=1.5 gives moderate self-confidence. Typical range: 1.0-2.0
        # (Kennedy & Eberhart 1995, Equation 1)
        c1 = self._get_config_value(
            lambda: self.config.optimization.pso_cognitive,
            default=PSODefaults.COGNITIVE,
            dict_key='PSO_COGNITIVE'
        )

        # Social coefficient (c2): attraction to global best
        # c2=1.5 gives moderate social influence. Typical range: 1.0-2.0
        # Often c1=c2 for balanced behavior
        # (Kennedy & Eberhart 1995, Equation 1)
        c2 = self._get_config_value(
            lambda: self.config.optimization.pso_social,
            default=PSODefaults.SOCIAL,
            dict_key='PSO_SOCIAL'
        )

        # Maximum velocity: limits velocity to prevent overshooting
        # v_max=0.2 limits movement to 20% of search space per iteration
        # Prevents oscillation and improves convergence stability
        # (Kennedy & Eberhart 1995, Section "Vmax")
        v_max = self._get_config_value(
            lambda: self.config.optimization.pso_v_max,
            default=PSODefaults.V_MAX,
            dict_key='PSO_V_MAX'
        )

        # Validate coefficients for convergence
        valid, warning = PSODefaults.validate_coefficients(w, c1, c2)
        if not valid:
            self.logger.warning(f"PSO coefficients may cause instability: {warning}")

        # Initialize swarm
        self.logger.info(f"Evaluating initial population ({n_particles} particles)...")
        positions = np.random.uniform(0, 1, (n_particles, n_params))
        velocities = np.random.uniform(-v_max, v_max, (n_particles, n_params))

        # Evaluate initial population
        fitness = evaluate_population(positions, 0)

        # Handle NaN/Inf fitness values
        invalid_mask = ~np.isfinite(fitness)
        if invalid_mask.any():
            self.logger.warning(
                f"Initial evaluation: {invalid_mask.sum()} particles "
                f"returned invalid fitness, assigning penalty"
            )
            fitness[invalid_mask] = float('-inf')

        # Initialize personal and global bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        global_best_idx = np.argmax(fitness)
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = fitness[global_best_idx]

        # Record initial best with enhanced tracking
        params_dict = denormalize_params(global_best_pos)
        record_iteration(0, global_best_fit, params_dict, {
            'mean_score': float(np.mean(fitness)),
            'std_score': float(np.std(fitness)),
            'n_improved': 0,
            'best_particle_idx': int(global_best_idx),
        })
        update_best(global_best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_particles, global_best_fit)

        # PSO main loop
        for iteration in range(1, self.max_iterations + 1):
            try:
                # Update velocities using standard PSO equation
                # v_i(t+1) = w * v_i(t) + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
                r1 = np.random.random((n_particles, n_params))
                r2 = np.random.random((n_particles, n_params))

                # Cognitive component: attraction to personal best
                cognitive = c1 * r1 * (personal_best_pos - positions)

                # Social component: attraction to global best
                social = c2 * r2 * (global_best_pos - positions)

                # Velocity update with inertia
                velocities = w * velocities + cognitive + social

                # Velocity clamping to prevent overshooting
                # This is crucial for convergence in bounded spaces
                velocities = np.clip(velocities, -v_max, v_max)

                # Update positions
                positions = positions + velocities

                # Bound positions to [0, 1] search space
                positions = np.clip(positions, 0, 1)

                # Evaluate new positions
                fitness = evaluate_population(positions, iteration)

                # Handle NaN/Inf fitness values
                invalid_mask = ~np.isfinite(fitness)
                if invalid_mask.any():
                    self.logger.warning(
                        f"Iteration {iteration}: {invalid_mask.sum()} particles "
                        f"returned invalid fitness, keeping previous best"
                    )
                    fitness[invalid_mask] = float('-inf')

                # Update personal bests (only if fitness improves)
                improved = fitness > personal_best_fit
                n_improved = np.sum(improved)
                personal_best_pos[improved] = positions[improved]
                personal_best_fit[improved] = fitness[improved]

                # Update global best
                if np.max(personal_best_fit) > global_best_fit:
                    global_best_idx = np.argmax(personal_best_fit)
                    global_best_pos = personal_best_pos[global_best_idx].copy()
                    global_best_fit = personal_best_fit[global_best_idx]

            except (ValueError, FloatingPointError) as e:
                self.logger.warning(
                    f"Error in iteration {iteration}: {e}. "
                    f"Continuing with previous state."
                )
                n_improved = 0

            # Record results with enhanced tracking for response surface analysis
            params_dict = denormalize_params(global_best_pos)
            record_iteration(iteration, global_best_fit, params_dict, {
                'mean_score': float(np.mean(fitness)),
                'std_score': float(np.std(fitness)),
                'n_improved': int(n_improved),
                'best_particle_idx': int(global_best_idx),
            })
            update_best(global_best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, global_best_fit, n_improved, n_particles)

        return {
            'best_solution': global_best_pos,
            'best_score': global_best_fit,
            'best_params': denormalize_params(global_best_pos)
        }
