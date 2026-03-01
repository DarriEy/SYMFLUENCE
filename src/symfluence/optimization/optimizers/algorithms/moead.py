#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)

A multi-objective optimization algorithm that decomposes a multi-objective
problem into a set of scalar subproblems using weight vectors. Each subproblem
is optimized simultaneously using information from neighboring subproblems.

Key Features:
    - Decomposes multi-objective problem into scalar subproblems
    - Uses neighborhood structure for efficient information sharing
    - Maintains diversity through evenly distributed weight vectors
    - Effective for many-objective optimization

Reference:
    Zhang, Q. and Li, H. (2007). MOEA/D: A Multiobjective Evolutionary Algorithm
    Based on Decomposition. IEEE Transactions on Evolutionary Computation,
    11(6), 712-731.

    Li, H. and Zhang, Q. (2009). Multiobjective Optimization Problems With
    Complicated Pareto Sets, MOEA/D and NSGA-II. IEEE Transactions on
    Evolutionary Computation, 13(2), 284-302.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import MOEADDefaults


class MOEADAlgorithm(OptimizationAlgorithm):
    """MOEA/D Multi-Objective Evolutionary Algorithm based on Decomposition."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "MOEA/D"

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
        Run MOEA/D optimization.

        For single-objective problems, MOEA/D reduces to a standard EA.
        For multi-objective, it decomposes using Tchebycheff or weighted sum.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            evaluate_population_objectives: Multi-objective evaluation callback
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        # Check if multi-objective
        is_multi_objective = evaluate_population_objectives is not None

        if is_multi_objective:
            objective_names = kwargs.get('objective_names', ['KGE', 'NSE'])
            return self._optimize_multi_objective(
                n_params, evaluate_population_objectives, objective_names, denormalize_params,
                record_iteration, update_best, log_progress, log_initial_population
            )
        else:
            # Single objective - use weighted sum decomposition with single weight
            return self._optimize_single_objective(
                n_params, evaluate_solution, evaluate_population, denormalize_params,
                record_iteration, update_best, log_progress, log_initial_population
            )

    def _optimize_single_objective(
        self,
        n_params: int,
        evaluate_solution: Callable,
        evaluate_population: Callable,
        denormalize_params: Callable,
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        log_initial_population: Optional[Callable]
    ) -> Dict[str, Any]:
        """Single-objective optimization using MOEA/D framework."""
        self.logger.info(f"Starting MOEA/D (single-objective mode) with {n_params} parameters")

        # MOEA/D parameters using standardized config access
        pop_size = self.population_size

        # Neighborhood size - number of neighbors for each subproblem
        # (Zhang & Li 2007, Section III-A)
        n_neighbors = self._get_config_value(
            lambda: self.config.optimization.moead_neighbors,
            default=min(MOEADDefaults.NEIGHBORS, pop_size - 1),
            dict_key='MOEAD_NEIGHBORS'
        )

        # Crossover rate for DE reproduction (Zhang & Li 2007, Section III-B)
        cr = self._get_config_value(
            lambda: self.config.optimization.moead_cr,
            default=MOEADDefaults.CR,
            dict_key='MOEAD_CR'
        )

        # DE scaling factor (Zhang & Li 2007, Section III-B)
        f = self._get_config_value(
            lambda: self.config.optimization.moead_f,
            default=MOEADDefaults.F,
            dict_key='MOEAD_F'
        )

        # Mutation rate for polynomial mutation
        mutation_rate = self._get_config_value(
            lambda: self.config.optimization.moead_mutation,
            default=MOEADDefaults.MUTATION,
            dict_key='MOEAD_MUTATION'
        )

        # Validate parameters
        valid, msg = MOEADDefaults.validate_decomposition('tchebycheff')
        if not valid:
            self.logger.warning(f"MOEA/D validation: {msg}")

        # Initialize population
        population = np.random.uniform(0, 1, (pop_size, n_params))
        fitness = evaluate_population(population, 0)

        # For single objective, all weight vectors point to same direction
        # But we still use neighborhood for local search
        # Create neighborhoods based on solution similarity
        neighborhoods = self._create_neighborhoods_by_distance(population, n_neighbors)

        # Track best
        best_idx = np.argmax(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # Main MOEA/D loop
        for iteration in range(1, self.max_iterations + 1):
            n_improved = 0

            for i in range(pop_size):
                # Select parents from neighborhood
                neighbors = neighborhoods[i]

                # DE-style reproduction
                if len(neighbors) >= 2:
                    r1, r2 = np.random.choice(neighbors, 2, replace=False)
                else:
                    r1, r2 = np.random.choice(pop_size, 2, replace=False)

                # Generate offspring via DE
                if np.random.random() < cr:
                    offspring = population[i] + f * (population[r1] - population[r2])
                else:
                    offspring = population[i].copy()

                # Polynomial mutation
                for j in range(n_params):
                    if np.random.random() < mutation_rate:
                        offspring[j] += np.random.normal(0, 0.1)

                offspring = np.clip(offspring, 0, 1)

                # Evaluate offspring
                offspring_fit = evaluate_solution(offspring, iteration)

                # Update neighbors if offspring is better
                for j in neighbors:
                    if offspring_fit > fitness[j]:
                        population[j] = offspring.copy()
                        fitness[j] = offspring_fit
                        n_improved += 1

                # Update global best
                if offspring_fit > best_fit:
                    best_fit = offspring_fit
                    best_pos = offspring.copy()

            # Update neighborhoods periodically
            if iteration % 10 == 0:
                neighborhoods = self._create_neighborhoods_by_distance(population, n_neighbors)

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict, {'n_improved': n_improved})
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit, n_improved, pop_size)

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'final_population': population,
            'final_fitness': fitness,
        }

    def _optimize_multi_objective(
        self,
        n_params: int,
        evaluate_objectives: Callable,
        objective_names: List[str],
        denormalize_params: Callable,
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        log_initial_population: Optional[Callable]
    ) -> Dict[str, Any]:
        """Multi-objective optimization using MOEA/D decomposition."""
        self.logger.info(f"Starting MOEA/D (multi-objective mode) with {n_params} parameters")

        # MOEA/D parameters using standardized config access
        pop_size = self.population_size

        # Neighborhood size (Zhang & Li 2007, Section III-A)
        n_neighbors = self._get_config_value(
            lambda: self.config.optimization.moead_neighbors,
            default=min(MOEADDefaults.NEIGHBORS, pop_size - 1),
            dict_key='MOEAD_NEIGHBORS'
        )

        # Crossover rate for DE reproduction (Zhang & Li 2007, Section III-B)
        cr = self._get_config_value(
            lambda: self.config.optimization.moead_cr,
            default=MOEADDefaults.CR,
            dict_key='MOEAD_CR'
        )

        # DE scaling factor (Zhang & Li 2007, Section III-B)
        f = self._get_config_value(
            lambda: self.config.optimization.moead_f,
            default=MOEADDefaults.F,
            dict_key='MOEAD_F'
        )

        # Mutation rate for polynomial mutation
        mutation_rate = self._get_config_value(
            lambda: self.config.optimization.moead_mutation,
            default=MOEADDefaults.MUTATION,
            dict_key='MOEAD_MUTATION'
        )

        # Max replacements per offspring (Zhang & Li 2007, Section III-C)
        nr = self._get_config_value(
            lambda: self.config.optimization.moead_nr,
            default=MOEADDefaults.NR,
            dict_key='MOEAD_NR'
        )

        # Decomposition approach (Zhang & Li 2007, Section II-B)
        decomposition = self._get_config_value(
            lambda: self.config.optimization.moead_decomposition,
            default=MOEADDefaults.DECOMPOSITION,
            dict_key='MOEAD_DECOMPOSITION'
        )

        # Validate decomposition method
        valid, msg = MOEADDefaults.validate_decomposition(decomposition)
        if not valid:
            self.logger.warning(f"MOEA/D validation: {msg}")

        # Determine number of objectives from first evaluation
        test_solution = np.random.uniform(0, 1, n_params)
        test_objectives = evaluate_objectives(test_solution.reshape(1, -1), objective_names, 0)[0]
        n_objectives = len(test_objectives)

        self.logger.info(f"Detected {n_objectives} objectives, decomposition={decomposition}")

        # Generate uniformly distributed weight vectors
        weights = self._generate_weight_vectors(pop_size, n_objectives)
        actual_pop_size = len(weights)

        # Create neighborhoods based on weight vector similarity
        neighborhoods = self._create_neighborhoods_by_weights(weights, n_neighbors)

        # Initialize population
        population = np.random.uniform(0, 1, (actual_pop_size, n_params))
        # Evaluate entire population at once to get proper individual_id assignment
        objectives = evaluate_objectives(population, objective_names, 0)

        # Reference point (ideal point) - best value for each objective
        z_ideal = np.max(objectives, axis=0)

        # Track best (using first objective as primary)
        best_idx = np.argmax(objectives[:, 0])
        best_pos = population[best_idx].copy()
        best_fit = objectives[best_idx, 0]

        # External archive for Pareto front
        archive: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(actual_pop_size):
            self._update_archive(archive, population[i], objectives[i])

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, actual_pop_size, best_fit)

        # Main MOEA/D loop
        for iteration in range(1, self.max_iterations + 1):
            n_improved = 0

            for i in range(actual_pop_size):
                neighbors = neighborhoods[i]

                # Select parents from neighborhood
                if len(neighbors) >= 2:
                    r1, r2 = np.random.choice(neighbors, 2, replace=False)
                else:
                    r1, r2 = np.random.choice(actual_pop_size, 2, replace=False)

                # DE reproduction
                if np.random.random() < cr:
                    offspring = population[i] + f * (population[r1] - population[r2])
                else:
                    offspring = population[i].copy()

                # Mutation
                for j in range(n_params):
                    if np.random.random() < mutation_rate:
                        offspring[j] += np.random.normal(0, 0.1)

                offspring = np.clip(offspring, 0, 1)

                # Evaluate offspring
                offspring_obj = evaluate_objectives(offspring.reshape(1, -1), objective_names, iteration)[0]

                # Update ideal point
                z_ideal = np.maximum(z_ideal, offspring_obj)

                # Update neighbors using decomposition (limit to nr replacements)
                shuffled_neighbors = list(neighbors)
                np.random.shuffle(shuffled_neighbors)
                n_replaced = 0
                for j in shuffled_neighbors:
                    if n_replaced >= nr:
                        break
                    if self._is_better_decomposed(
                        offspring_obj, objectives[j], weights[j], z_ideal, decomposition
                    ):
                        population[j] = offspring.copy()
                        objectives[j] = offspring_obj.copy()
                        n_improved += 1
                        n_replaced += 1

                # Update archive
                self._update_archive(archive, offspring, offspring_obj)

                # Update global best (first objective)
                if offspring_obj[0] > best_fit:
                    best_fit = offspring_obj[0]
                    best_pos = offspring.copy()

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {'archive_size': len(archive), 'n_improved': n_improved}
            )
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit, n_improved, actual_pop_size)

        # Extract Pareto front from archive
        pareto_solutions = np.array([x for x, _ in archive])
        pareto_objectives = np.array([obj for _, obj in archive])

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'pareto_front': pareto_solutions,
            'pareto_objectives': pareto_objectives,
            'final_population': population,
            'final_objectives': objectives,
        }

    def _generate_weight_vectors(self, n_vectors: int, n_objectives: int) -> np.ndarray:
        """Generate uniformly distributed weight vectors."""
        if n_objectives == 2:
            # Simple linear spacing for bi-objective
            weights = np.zeros((n_vectors, 2))
            for i in range(n_vectors):
                weights[i, 0] = i / (n_vectors - 1)
                weights[i, 1] = 1 - weights[i, 0]
            return weights
        else:
            # Simplex-lattice design for many objectives
            from itertools import combinations_with_replacement
            H = n_vectors  # Number of divisions
            # Generate all combinations
            indices = list(combinations_with_replacement(range(n_objectives), H))
            weights = []
            for idx in indices[:n_vectors]:
                w = np.zeros(n_objectives)
                for i in idx:
                    w[i] += 1
                w = w / H
                weights.append(w)
            return np.array(weights)

    def _create_neighborhoods_by_weights(
        self,
        weights: np.ndarray,
        n_neighbors: int
    ) -> List[List[int]]:
        """Create neighborhoods based on weight vector Euclidean distance."""
        n = len(weights)
        neighborhoods = []

        for i in range(n):
            distances = np.linalg.norm(weights - weights[i], axis=1)
            neighbors = np.argsort(distances)[1:n_neighbors + 1].tolist()
            neighborhoods.append(neighbors)

        return neighborhoods

    def _create_neighborhoods_by_distance(
        self,
        population: np.ndarray,
        n_neighbors: int
    ) -> List[List[int]]:
        """Create neighborhoods based on solution distance."""
        n = len(population)
        neighborhoods = []

        for i in range(n):
            distances = np.linalg.norm(population - population[i], axis=1)
            neighbors = np.argsort(distances)[1:n_neighbors + 1].tolist()
            neighborhoods.append(neighbors)

        return neighborhoods

    def _is_better_decomposed(
        self,
        obj1: np.ndarray,
        obj2: np.ndarray,
        weight: np.ndarray,
        z_ideal: np.ndarray,
        decomposition: str
    ) -> bool:
        """Check if obj1 is better than obj2 under decomposition approach.

        For maximization problems (higher objective values are better):
        - Tchebycheff: minimize max_i { w_i * (z*_i - f_i(x)) }
          Lower scalarized value means closer to ideal point = better
        - Weighted sum: maximize sum_i { w_i * f_i(x) }
          Higher weighted sum = better
        """
        if decomposition == 'tchebycheff':
            # Tchebycheff approach (maximization)
            # g = max weighted gap from ideal point; lower is better
            g1 = np.max(weight * (z_ideal - obj1))
            g2 = np.max(weight * (z_ideal - obj2))
            return g1 < g2  # Lower gap from ideal is better
        else:
            # Weighted sum approach
            g1 = np.sum(weight * obj1)
            g2 = np.sum(weight * obj2)
            return g1 > g2

    def _update_archive(
        self,
        archive: List[Tuple[np.ndarray, np.ndarray]],
        solution: np.ndarray,
        objectives: np.ndarray,
        max_size: int = 100
    ) -> None:
        """Update external archive with non-dominated solutions."""
        # Check if solution is dominated by any archive member
        dominated = False
        to_remove = []

        for i, (_, obj) in enumerate(archive):
            if np.all(obj >= objectives) and np.any(obj > objectives):
                # Solution is dominated
                dominated = True
                break
            if np.all(objectives >= obj) and np.any(objectives > obj):
                # Solution dominates archive member
                to_remove.append(i)

        if not dominated:
            # Remove dominated members
            for i in sorted(to_remove, reverse=True):
                archive.pop(i)
            # Add new solution
            archive.append((solution.copy(), objectives.copy()))

            # Trim archive if too large
            if len(archive) > max_size:
                # Remove most crowded solution
                archive.pop(np.random.randint(len(archive)))
