#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II)

A multi-objective evolutionary algorithm that uses non-dominated sorting
and crowding distance for maintaining diversity in the Pareto front.

Reference:
    Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions
    on Evolutionary Computation, 6(2), 182-197.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class NSGA2Algorithm(OptimizationAlgorithm):
    """NSGA-II multi-objective optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "NSGA-II"

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
        objective_names: Optional[List[str]] = None,
        multiobjective: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run NSGA-II optimization.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            evaluate_population_objectives: Callback for multi-objective evaluation
            objective_names: List of objective metric names
            multiobjective: Whether to use multi-objective evaluation
            **kwargs: Additional parameters

        Returns:
            Optimization results dictionary
        """
        self.logger.debug(f"Starting NSGA-II optimization with {n_params} parameters")

        pop_size = self.population_size
        objective_names = objective_names or ['KGE', 'NSE']
        num_objectives = len(objective_names)

        # NSGA-II parameters
        crossover_rate = self.config.get('NSGA2_CROSSOVER_RATE', 0.9)
        mutation_rate = self.config.get('NSGA2_MUTATION_RATE', 0.5)
        eta_c = self.config.get('NSGA2_ETA_C', 15)  # Crossover distribution index
        eta_m = self.config.get('NSGA2_ETA_M', 10)  # Mutation distribution index

        # Initialize population
        self.logger.debug(f"Initializing population ({pop_size} individuals)...")
        population = np.random.uniform(0, 1, (pop_size, n_params))
        objectives = np.full((pop_size, num_objectives), np.nan)

        # Evaluate initial population
        if multiobjective and evaluate_population_objectives:
            objectives = evaluate_population_objectives(population, objective_names, 0)
        else:
            fitness = evaluate_population(population, 0)
            objectives[:, 0] = fitness
            if num_objectives > 1:
                objectives[:, 1] = fitness

        # Perform NSGA-II selection
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        # Find representative solution (best on first objective)
        best_idx = np.argmax(objectives[:, 0])
        best_solution = population[best_idx].copy()
        best_fitness = objectives[best_idx, 0]
        best_secondary = objectives[best_idx, 1] if num_objectives > 1 else None

        # Record initial best
        extra_metrics = {}
        if best_secondary is not None:
            extra_metrics[f"obj2_{objective_names[1]}"] = float(best_secondary)
        params_dict = denormalize_params(best_solution)
        record_iteration(0, best_fitness, params_dict, additional_metrics=extra_metrics or None)
        update_best(best_fitness, params_dict, 0)

        if best_secondary is not None:
            self.logger.info(
                "Initial population complete | Best obj1 (%s): %.4f | Best obj2 (%s): %.4f",
                objective_names[0], best_fitness, objective_names[1], best_secondary
            )
        else:
            self.logger.info(f"Initial population complete | Best obj1: {best_fitness:.4f}")

        # Main NSGA-II loop
        for generation in range(1, self.max_iterations + 1):
            # Generate offspring through selection, crossover, and mutation
            offspring = np.zeros_like(population)
            for i in range(0, pop_size, 2):
                # Tournament selection
                p1_idx = self._tournament_selection(ranks, crowding_distances, pop_size)
                p2_idx = self._tournament_selection(ranks, crowding_distances, pop_size)
                p1, p2 = population[p1_idx], population[p2_idx]

                # Crossover
                if np.random.random() < crossover_rate:
                    c1, c2 = self._sbx_crossover(p1, p2, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                offspring[i] = self._polynomial_mutation(c1, eta_m, mutation_rate)
                if i + 1 < pop_size:
                    offspring[i + 1] = self._polynomial_mutation(c2, eta_m, mutation_rate)

            # Evaluate offspring
            if multiobjective and evaluate_population_objectives:
                offspring_objectives = evaluate_population_objectives(
                    offspring, objective_names, generation
                )
            else:
                offspring_fitness = evaluate_population(offspring, generation)
                offspring_objectives = np.full((pop_size, num_objectives), np.nan)
                offspring_objectives[:, 0] = offspring_fitness
                if num_objectives > 1:
                    offspring_objectives[:, 1] = offspring_fitness

            # Combine parent and offspring populations
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, offspring_objectives])

            # Environmental selection
            selected_indices = self._environmental_selection(combined_obj, pop_size)
            population = combined_pop[selected_indices]
            objectives = combined_obj[selected_indices]

            # Update ranks and crowding distances
            ranks = self._fast_non_dominated_sort(objectives)
            crowding_distances = self._calculate_crowding_distance(objectives, ranks)

            # Update best solution
            current_best_idx = np.argmax(objectives[:, 0])
            if objectives[current_best_idx, 0] > best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = objectives[current_best_idx, 0]
                best_secondary = objectives[current_best_idx, 1] if num_objectives > 1 else None

            # Record results
            params_dict = denormalize_params(best_solution)
            extra_metrics = {}
            if best_secondary is not None:
                extra_metrics[f"obj2_{objective_names[1]}"] = float(best_secondary)
            record_iteration(
                generation, best_fitness, params_dict,
                additional_metrics=extra_metrics or None
            )
            update_best(best_fitness, params_dict, generation)

            # Log progress
            log_progress(
                self.name, generation, best_fitness,
                secondary_score=best_secondary,
                secondary_label=objective_names[1] if best_secondary is not None else None
            )

        return {
            'best_solution': best_solution,
            'best_score': best_fitness,
            'best_params': denormalize_params(best_solution),
            'pareto_front': population[ranks == 0] if ranks is not None else None,
            'pareto_objectives': objectives[ranks == 0] if ranks is not None else None
        }

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> np.ndarray:
        """Fast non-dominated sorting for NSGA-II."""
        pop_size = len(objectives)
        ranks = np.zeros(pop_size, dtype=int)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]

        # Find domination relationships
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Assign ranks
        current_front = np.where(domination_count == 0)[0]
        rank = 0
        while len(current_front) > 0:
            ranks[current_front] = rank
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = np.array(next_front)
            rank += 1

        return ranks

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (maximization)."""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _calculate_crowding_distance(
        self, objectives: np.ndarray, ranks: np.ndarray
    ) -> np.ndarray:
        """Calculate crowding distance for each solution."""
        pop_size = len(objectives)
        num_objectives = objectives.shape[1]
        crowding_distance = np.zeros(pop_size)

        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(rank_indices) <= 2:
                crowding_distance[rank_indices] = np.inf
                continue

            for obj_idx in range(num_objectives):
                obj_values = objectives[rank_indices, obj_idx]
                sorted_indices = np.argsort(obj_values)
                sorted_rank_indices = rank_indices[sorted_indices]

                # Boundary solutions get infinite distance
                crowding_distance[sorted_rank_indices[0]] = np.inf
                crowding_distance[sorted_rank_indices[-1]] = np.inf

                # Calculate crowding distance for middle solutions
                obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        crowding_distance[sorted_rank_indices[i]] += (
                            (obj_values[sorted_indices[i + 1]] - obj_values[sorted_indices[i - 1]]) / obj_range
                        )

        return crowding_distance

    def _tournament_selection(
        self,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        pop_size: int
    ) -> int:
        """Tournament selection for NSGA-II."""
        candidates = np.random.choice(pop_size, 2, replace=False)
        best_idx = candidates[0]

        for candidate in candidates[1:]:
            if (ranks[candidate] < ranks[best_idx] or
                (ranks[candidate] == ranks[best_idx] and
                 crowding_distances[candidate] > crowding_distances[best_idx])):
                best_idx = candidate

        return best_idx

    def _environmental_selection(
        self, objectives: np.ndarray, target_size: int
    ) -> np.ndarray:
        """Select best individuals for next generation."""
        ranks = self._fast_non_dominated_sort(objectives)
        crowding_distances = self._calculate_crowding_distance(objectives, ranks)

        selected_indices = []
        for rank in np.unique(ranks):
            rank_indices = np.where(ranks == rank)[0]
            if len(selected_indices) + len(rank_indices) <= target_size:
                selected_indices.extend(rank_indices)
            else:
                # Sort by crowding distance and select best
                remaining = target_size - len(selected_indices)
                rank_crowding = crowding_distances[rank_indices]
                sorted_indices = np.argsort(rank_crowding)[::-1]
                selected_indices.extend(rank_indices[sorted_indices[:remaining]])
                break

        return np.array(selected_indices)

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        eta_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        c1, c2 = p1.copy(), p2.copy()
        n_params = len(p1)

        for i in range(n_params):
            if np.random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-9:
                if p1[i] < p2[i]:
                    y1, y2 = p1[i], p2[i]
                else:
                    y1, y2 = p2[i], p1[i]

                rand = np.random.random()
                beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta_c + 1.0)

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                c1[i] = np.clip(c1[i], 0, 1)
                c2[i] = np.clip(c2[i], 0, 1)

        return c1, c2

    def _polynomial_mutation(
        self,
        solution: np.ndarray,
        eta_m: float,
        mutation_rate: float
    ) -> np.ndarray:
        """Polynomial mutation."""
        mutated = solution.copy()
        n_params = len(solution)

        for i in range(n_params):
            if np.random.random() < mutation_rate:
                y = mutated[i]
                delta1 = y - 0.0
                delta2 = 1.0 - y

                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutated[i] = y + deltaq
                mutated[i] = np.clip(mutated[i], 0, 1)

        return mutated
