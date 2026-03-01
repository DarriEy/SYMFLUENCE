#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Async DDS (Asynchronous Parallel Dynamically Dimensioned Search) Algorithm

An asynchronous parallel variant of DDS that maintains a pool of best solutions
and generates batches of trials by selecting from the pool. More efficient
for parallel execution than synchronous DDS.
"""

import random
from typing import Any, Callable, Dict, Optional

import numpy as np

from .base_algorithm import OptimizationAlgorithm


class AsyncDDSAlgorithm(OptimizationAlgorithm):
    """Asynchronous Parallel DDS optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "AsyncDDS"

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
        num_processes: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Asynchronous DDS optimization.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            num_processes: Number of parallel processes
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        self.logger.debug(f"Starting Async DDS optimization with {n_params} parameters")

        # Async DDS parameters
        import sys

        # On macOS, spawn-based multiprocessing is brittle for very fast model runs.
        # Default to serial pool/batch to avoid deadlocks/hangs unless user overrides.
        default_pool = 1 if sys.platform == "darwin" else min(20, num_processes * 2)
        default_batch = 1 if sys.platform == "darwin" else num_processes

        dds_r = self._get_config_value(lambda: self.config.optimization.dds.r, default=0.2, dict_key='DDS_R')
        pool_size = self._get_config_value(lambda: self.config.optimization.dds.async_pool_size, default=default_pool, dict_key='ASYNC_DDS_POOL_SIZE')
        batch_size = self._get_config_value(lambda: self.config.optimization.dds.async_batch_size, default=default_batch, dict_key='ASYNC_DDS_BATCH_SIZE')
        max_stagnation = self._get_config_value(lambda: self.config.optimization.dds.max_stagnation_batches, default=10, dict_key='MAX_STAGNATION_BATCHES')

        # Calculate target evaluations
        total_target_evaluations = self.max_iterations * num_processes
        target_batches = max(1, total_target_evaluations // batch_size) if self.max_iterations > 0 else 0

        # Solution pool tracking
        solution_pool = []  # List of (solution, score, batch_num) tuples
        pool_scores = []
        total_evaluations = 0
        stagnation_counter = 0
        best_score = float('-inf')
        best_solution = None

        self.logger.info("Async DDS configuration:")
        self.logger.info(f"  Pool size: {pool_size}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Target batches: {target_batches}")
        self.logger.info(f"  MPI processes: {num_processes}")

        # Initialize solution pool
        self.logger.info(f"Evaluating initial pool ({pool_size} solutions)...")
        initial_population = np.random.uniform(0, 1, (pool_size, n_params))

        # Inject initial guess if provided
        initial_guess = kwargs.get('initial_guess')
        if initial_guess is not None:
            if isinstance(initial_guess, (list, np.ndarray)) and len(initial_guess) == n_params:
                self.logger.info("Injecting initial guess into population")
                initial_population[0] = np.array(initial_guess)
            else:
                self.logger.warning(f"Initial guess provided but shape mismatch: expected {n_params}, got {len(initial_guess) if hasattr(initial_guess, '__len__') else 'unknown'}")

        initial_fitness = evaluate_population(initial_population, 0)

        # Log initial pool scores for debugging
        valid_initial = [s for s in initial_fitness if s is not None and s != self.penalty_score]
        if valid_initial:
            self.logger.info(
                f"Initial pool scores: min={min(valid_initial):.4f}, max={max(valid_initial):.4f}"
            )
        else:
            self.logger.warning("Initial pool: No valid scores!")

        for i, (solution, score) in enumerate(zip(initial_population, initial_fitness)):
            if score is not None and score != self.penalty_score:
                solution_pool.append((solution.copy(), score, 0))
                pool_scores.append(score)
                total_evaluations += 1

                if score > best_score:
                    best_score = score
                    best_solution = solution.copy()

        # Sort pool by score (best first)
        if solution_pool:
            combined = list(zip(solution_pool, pool_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            solution_pool = [item[0] for item in combined]
            pool_scores = [item[1] for item in combined]

        if not solution_pool:
            self.logger.error("No valid solutions in initial pool - all evaluations failed")
            return {
                'best_solution': None,
                'best_score': self.penalty_score,
                'best_params': {}
            }

        # Record initial best
        params_dict = denormalize_params(best_solution)
        record_iteration(0, best_score, params_dict)
        update_best(best_score, params_dict, 0)
        self.logger.info(f"Initial pool complete | Best score: {best_score:.4f}")

        # Main batch loop
        for batch_num in range(1, target_batches + 1):
            # Check convergence
            if total_evaluations >= total_target_evaluations:
                self.logger.info(f"Reached target evaluations: {total_evaluations}")
                break
            if stagnation_counter >= max_stagnation:
                self.logger.info(f"Stopping due to stagnation ({stagnation_counter} batches)")
                break

            # Generate batch of trials from pool
            trials = []
            for i in range(batch_size):
                # Tournament selection from pool
                tournament_size = min(3, len(solution_pool))
                candidates = random.sample(range(len(solution_pool)), tournament_size)
                parent_idx = min(candidates, key=lambda idx: -pool_scores[idx])  # Best wins
                parent = solution_pool[parent_idx][0].copy()

                # DDS perturbation
                prob_select = max(
                    1.0 - np.log(total_evaluations + i + 1) / np.log(total_target_evaluations),
                    1.0 / n_params
                )

                trial = parent.copy()
                perturb_mask = np.random.random(n_params) < prob_select
                if not perturb_mask.any():
                    perturb_mask[np.random.randint(n_params)] = True

                for j in range(n_params):
                    if perturb_mask[j]:
                        perturbation = np.random.normal(0, dds_r)
                        trial[j] = parent[j] + perturbation

                        # Reflect at bounds
                        if trial[j] < 0:
                            trial[j] = -trial[j]
                        elif trial[j] > 1:
                            trial[j] = 2.0 - trial[j]
                        trial[j] = np.clip(trial[j], 0, 1)

                trials.append(trial)

            # Evaluate batch
            trial_population = np.array(trials)
            trial_fitness = evaluate_population(trial_population, batch_num)

            # Update pool with batch results
            improvements = 0
            for trial, score in zip(trials, trial_fitness):
                if score is None or score == self.penalty_score:
                    continue

                total_evaluations += 1

                # Check for improvement
                is_improvement = False
                if score > best_score:
                    best_score = score
                    best_solution = trial.copy()
                    stagnation_counter = 0
                    is_improvement = True
                elif len(solution_pool) < pool_size:
                    is_improvement = True
                elif pool_scores and score > min(pool_scores):
                    is_improvement = True

                if is_improvement:
                    improvements += 1

                # Add to pool
                solution_pool.append((trial.copy(), score, batch_num))
                pool_scores.append(score)

            # Trim pool to size
            if len(solution_pool) > pool_size:
                combined = list(zip(solution_pool, pool_scores))
                combined.sort(key=lambda x: x[1], reverse=True)
                solution_pool = [item[0] for item in combined[:pool_size]]
                pool_scores = [item[1] for item in combined[:pool_size]]

            if improvements == 0:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Record results
            params_dict = denormalize_params(best_solution)
            record_iteration(batch_num, best_score, params_dict)
            update_best(best_score, params_dict, batch_num)

            # Log progress every 10 batches or at the end to reduce log spam
            if batch_num % 10 == 0 or batch_num >= self.max_iterations // batch_size:
                log_progress(self.name, batch_num, best_score, improvements, batch_size)

        self.logger.info("AsyncDDS completed")
        self.logger.info(f"Total evaluations: {total_evaluations}")
        self.logger.info(f"Final pool size: {len(solution_pool)}")

        return {
            'best_solution': best_solution,
            'best_score': best_score,
            'best_params': denormalize_params(best_solution)
        }
