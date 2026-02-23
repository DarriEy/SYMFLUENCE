"""
NSGA-II Operators

Provides genetic operators and selection mechanisms for NSGA-II.
"""

from typing import List, Tuple

import numpy as np


class NSGA2Operators:
    """
    Genetic operators and selection mechanisms for NSGA-II.

    Provides:
    - Fast non-dominated sorting
    - Crowding distance calculation
    - Tournament selection
    - Environmental selection
    - Simulated Binary Crossover (SBX)
    - Polynomial mutation
    """

    @staticmethod
    def fast_non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
        """
        Perform fast non-dominated sorting.

        Args:
            objectives: Objective values array (n_individuals x n_objectives)

        Returns:
            Array of ranks for each individual (0 = Pareto front)
        """
        pop_size = len(objectives)
        ranks = np.zeros(pop_size, dtype=int)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions: List[List[int]] = [[] for _ in range(pop_size)]

        # Find domination relationships
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if NSGA2Operators._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif NSGA2Operators._dominates(objectives[j], objectives[i]):
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

    @staticmethod
    def _dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 dominates obj2 (maximization).

        Args:
            obj1: First objective vector
            obj2: Second objective vector

        Returns:
            True if obj1 dominates obj2
        """
        return bool(np.all(obj1 >= obj2)) and bool(np.any(obj1 > obj2))

    @staticmethod
    def calculate_crowding_distance(
        objectives: np.ndarray,
        ranks: np.ndarray
    ) -> np.ndarray:
        """
        Calculate crowding distance for each solution.

        Args:
            objectives: Objective values array (n_individuals x n_objectives)
            ranks: Rank array from non-dominated sorting

        Returns:
            Array of crowding distances
        """
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

    @staticmethod
    def tournament_selection(
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        pop_size: int
    ) -> int:
        """
        Tournament selection for NSGA-II.

        Selects based on rank first, then crowding distance.

        Args:
            ranks: Rank array
            crowding_distances: Crowding distance array
            pop_size: Population size

        Returns:
            Selected individual index
        """
        candidates = np.random.choice(pop_size, 2, replace=False)
        best_idx = candidates[0]

        for candidate in candidates[1:]:
            if (ranks[candidate] < ranks[best_idx] or
                (ranks[candidate] == ranks[best_idx] and
                 crowding_distances[candidate] > crowding_distances[best_idx])):
                best_idx = candidate

        return best_idx

    @staticmethod
    def environmental_selection(
        objectives: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """
        Select best individuals for next generation.

        Args:
            objectives: Objective values array (n_individuals x n_objectives)
            target_size: Number of individuals to select

        Returns:
            Array of selected individual indices
        """
        ranks = NSGA2Operators.fast_non_dominated_sort(objectives)
        crowding_distances = NSGA2Operators.calculate_crowding_distance(objectives, ranks)

        selected_indices: List[int] = []
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

    @staticmethod
    def sbx_crossover(
        p1: np.ndarray,
        p2: np.ndarray,
        eta_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).

        Args:
            p1: First parent (normalized [0, 1])
            p2: Second parent (normalized [0, 1])
            eta_c: Crossover distribution index

        Returns:
            Tuple of two offspring
        """
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

    @staticmethod
    def polynomial_mutation(
        solution: np.ndarray,
        eta_m: float,
        mutation_rate: float
    ) -> np.ndarray:
        """
        Polynomial mutation.

        Args:
            solution: Solution to mutate (normalized [0, 1])
            eta_m: Mutation distribution index
            mutation_rate: Probability of mutating each variable

        Returns:
            Mutated solution
        """
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
