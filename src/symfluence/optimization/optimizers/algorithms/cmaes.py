#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

A state-of-the-art evolutionary algorithm for derivative-free optimization.
CMA-ES adapts the covariance matrix of a multivariate normal distribution
to efficiently search the parameter space, making it highly effective for
ill-conditioned and non-separable optimization problems.

Key Features:
    - Self-adaptive step size and search direction
    - Handles parameter correlations through covariance matrix adaptation
    - Invariant to rotation and scaling of the search space
    - No user-defined parameters except population size (robust defaults)

Reference:
    Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review.
    In Towards a New Evolutionary Computation, pp. 75-102.

    Hansen, N. and Ostermeier, A. (2001). Completely Derandomized Self-Adaptation
    in Evolution Strategies. Evolutionary Computation, 9(2), 159-195.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class CMAESAlgorithm(OptimizationAlgorithm):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "CMA-ES"

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
        Run CMA-ES optimization.

        CMA-ES maintains a multivariate normal distribution N(m, σ²C) where:
        - m is the mean (current best estimate)
        - σ is the global step size
        - C is the covariance matrix (adapted to capture parameter correlations)

        The algorithm iteratively:
        1. Samples λ candidate solutions from N(m, σ²C)
        2. Evaluates and ranks candidates by fitness
        3. Updates m toward weighted mean of best μ solutions
        4. Adapts σ using cumulative step-size adaptation (CSA)
        5. Adapts C using rank-μ and rank-one updates

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
        self.logger.info(f"Starting CMA-ES optimization with {n_params} parameters")

        # Population size (λ) - default heuristic
        lambda_ = self.population_size
        if lambda_ < 4:
            lambda_ = 4 + int(3 * np.log(n_params))
            self.logger.info(f"Adjusted population size to {lambda_} for {n_params} parameters")

        # Number of parents (μ) - typically λ/2
        mu = lambda_ // 2

        # Recombination weights (log-linear weighting)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()  # Variance-effective selection mass

        # Strategy parameter defaults (from Hansen's recommendations)
        # Step-size control
        c_sigma = (mu_eff + 2) / (n_params + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n_params + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n_params) / (n_params + 4 + 2 * mu_eff / n_params)
        c_1 = 2 / ((n_params + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n_params + 2) ** 2 + mu_eff))

        # Expected length of N(0,I) distributed random vector
        chi_n = np.sqrt(n_params) * (1 - 1 / (4 * n_params) + 1 / (21 * n_params ** 2))

        # Initialize state
        # Mean at center of normalized space [0, 1]
        mean = np.full(n_params, 0.5)

        # Initial step size (σ) - covers about 1/3 of the range
        sigma = 0.3

        # Covariance matrix (identity initially)
        C = np.eye(n_params)

        # Evolution paths
        p_sigma = np.zeros(n_params)  # Step-size evolution path
        p_c = np.zeros(n_params)  # Covariance matrix evolution path

        # Eigendecomposition (for sampling)
        B = np.eye(n_params)  # Eigenvectors
        D = np.ones(n_params)  # Sqrt of eigenvalues
        invsqrt_C = np.eye(n_params)
        eigeneval = 0  # Last eigendecomposition count

        # Track best solution
        best_pos = mean.copy()
        best_fit = float('-inf')
        eval_count = 0

        # Main optimization loop
        for generation in range(1, self.max_iterations + 1):
            # Generate λ offspring by sampling from N(mean, σ²C)
            # x_k = mean + σ * B * D * z_k, where z_k ~ N(0, I)
            population = np.zeros((lambda_, n_params))
            z_samples = np.zeros((lambda_, n_params))

            for k in range(lambda_):
                z = np.random.randn(n_params)
                z_samples[k] = z
                y = B @ (D * z)  # Transform by sqrt(C)
                x = mean + sigma * y
                # Clip to bounds [0, 1]
                population[k] = np.clip(x, 0, 1)

            # Evaluate population
            fitness = evaluate_population(population, generation)
            eval_count += lambda_

            # Sort by fitness (descending - we maximize)
            sorted_indices = np.argsort(-fitness)

            # Update best
            if fitness[sorted_indices[0]] > best_fit:
                best_fit = fitness[sorted_indices[0]]
                best_pos = population[sorted_indices[0]].copy()

            # Select μ best individuals
            selected_indices = sorted_indices[:mu]

            # Compute weighted mean of selected points
            old_mean = mean.copy()
            mean = np.zeros(n_params)
            for i, idx in enumerate(selected_indices):
                mean += weights[i] * population[idx]
            mean = np.clip(mean, 0, 1)

            # Update evolution paths
            # p_sigma: cumulative step-size adaptation path
            mean_shift = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (invsqrt_C @ mean_shift)

            # Heaviside function for stalling detection
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * generation)) < (1.4 + 2 / (n_params + 1)) * chi_n else 0

            # p_c: covariance matrix adaptation path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_shift

            # Adapt covariance matrix C
            # Rank-one update
            rank_one = np.outer(p_c, p_c)

            # Rank-μ update
            rank_mu = np.zeros((n_params, n_params))
            for i, idx in enumerate(selected_indices):
                y_i = (population[idx] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            # Combined update
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu

            # Enforce symmetry
            C = (C + C.T) / 2

            # Adapt step size σ
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Bound sigma to prevent explosion/collapse
            sigma = max(1e-10, min(sigma, 1.0))

            # Update eigendecomposition periodically
            if eval_count - eigeneval > lambda_ / (c_1 + c_mu) / n_params / 10:
                eigeneval = eval_count
                try:
                    # Ensure C is positive definite
                    C = (C + C.T) / 2
                    eigenvalues, B = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-20)  # Prevent negative eigenvalues
                    D = np.sqrt(eigenvalues)
                    invsqrt_C = B @ np.diag(1 / D) @ B.T
                except np.linalg.LinAlgError:
                    self.logger.warning("Eigendecomposition failed, resetting covariance matrix")
                    C = np.eye(n_params)
                    B = np.eye(n_params)
                    D = np.ones(n_params)
                    invsqrt_C = np.eye(n_params)

            # Record iteration
            params_dict = denormalize_params(best_pos)
            n_improved = int(np.sum(fitness[selected_indices] > np.median(fitness)))
            record_iteration(generation, best_fit, params_dict, {'sigma': sigma, 'n_improved': n_improved})
            update_best(best_fit, params_dict, generation)

            # Log progress
            log_progress(self.name, generation, best_fit, n_improved, lambda_)

            # Early stopping: if sigma becomes very small, we've converged
            if sigma < 1e-12:
                self.logger.info(f"CMA-ES converged at generation {generation} (σ < 1e-12)")
                break

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'final_sigma': sigma,
            'evaluations': eval_count
        }
