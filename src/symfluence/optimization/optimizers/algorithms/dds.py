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
        self.logger.debug(f"Starting DDS optimization with {n_params} parameters")

        # Checkpoint callback for crash recovery (saves every N iterations)
        save_checkpoint = kwargs.get('save_checkpoint')
        checkpoint_interval = self._get_config_value(
            lambda: self.config.optimization.dds.checkpoint_interval,
            default=50, dict_key='DDS_CHECKPOINT_INTERVAL'
        )

        # DDS perturbation range (default 0.2, higher values explore more)
        r = self._get_config_value(lambda: self.config.optimization.dds.r, default=0.2, dict_key='DDS_R')

        # Minimum perturbation probability to ensure exploration even late in optimization
        p_min = self._get_config_value(
            lambda: self.config.optimization.dds.p_min, default=0.05, dict_key='DDS_P_MIN'
        )

        # Initialize with random starting point or provided initial guess
        initial_guess = kwargs.get('initial_guess')
        if initial_guess is not None and len(initial_guess) == n_params:
            x_best = np.array(initial_guess, dtype=float)
            self.logger.info(
                f"DDS using provided initial guess ({n_params} params, "
                f"mean={x_best.mean():.4f}, range=[{x_best.min():.4f}, {x_best.max():.4f}])"
            )
        else:
            x_best = np.random.uniform(0, 1, n_params)
            self.logger.info(
                f"DDS using RANDOM start (initial_guess={'None' if initial_guess is None else f'len={len(initial_guess)}'}, "
                f"n_params={n_params})"
            )

        f_best = evaluate_solution(x_best, 0)

        # Record initial state
        params_dict = denormalize_params(x_best)
        record_iteration(0, f_best, params_dict)
        update_best(f_best, params_dict, 0)

        # Stagnation detection settings
        stagnation_threshold = self._get_config_value(
            lambda: self.config.optimization.dds.stagnation_threshold,
            default=100, dict_key='DDS_STAGNATION_THRESHOLD'
        )
        iterations_since_improvement = 0

        # Initialize current perturbation range
        current_r = r

        # DDS main loop
        for iteration in range(1, self.max_iterations + 1):
            # Stagnation detection: if no improvement for many iterations,
            # increase perturbation to explore more
            if iterations_since_improvement >= stagnation_threshold:
                current_r = min(r * 2.0, 0.5)  # Double r but cap at 0.5
                if iterations_since_improvement == stagnation_threshold:
                    self.logger.debug(
                        f"DDS stagnation detected at iteration {iteration} "
                        f"(no improvement for {iterations_since_improvement} iterations). "
                        f"Increasing perturbation to {current_r:.3f}"
                    )
            else:
                current_r = r

            # Calculate probability of perturbation (decreases with iterations)
            # Use p_min to maintain minimum exploration capability throughout optimization
            p = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            p = max(p_min, max(1.0 / n_params, p))  # Ensure minimum perturbation probability

            # Select parameters to perturb
            perturb_mask = np.random.random(n_params) < p

            # Ensure at least one parameter is perturbed
            if not perturb_mask.any():
                perturb_mask[np.random.randint(n_params)] = True

            # Generate candidate solution
            x_new = x_best.copy()

            for i in range(n_params):
                if perturb_mask[i]:
                    perturbation = current_r * np.random.standard_normal()
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
            # Don't count crashes toward stagnation — penalty scores indicate
            # model failure (e.g. CLASS energy balance crash), not search
            # exhaustion.  Counting them triggers perturbation doubling which
            # generates more extreme parameter combos, causing more crashes.
            is_crash = (f_new <= self.penalty_score)
            if f_new > f_best:
                x_best = x_new
                f_best = f_new
                iterations_since_improvement = 0
            elif not is_crash:
                iterations_since_improvement += 1

            # Record results
            params_dict = denormalize_params(x_best)
            record_iteration(iteration, f_best, params_dict)
            update_best(f_best, params_dict, iteration)

            # Log progress every 10 iterations or at the end to reduce log spam
            if iteration % 10 == 0 or iteration == self.max_iterations:
                log_progress(self.name, iteration, f_best)

            # Periodic checkpoint for crash recovery
            if save_checkpoint and iteration % checkpoint_interval == 0:
                save_checkpoint(self.name, iteration)

        return {
            'best_solution': x_best,
            'best_score': f_best,
            'best_params': denormalize_params(x_best)
        }
