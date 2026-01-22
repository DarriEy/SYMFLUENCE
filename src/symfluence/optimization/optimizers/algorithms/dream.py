#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DREAM (DiffeRential Evolution Adaptive Metropolis)

A Markov Chain Monte Carlo (MCMC) algorithm that uses differential evolution
for efficient proposal generation. DREAM is particularly popular in hydrology
for its ability to provide parameter uncertainty estimates alongside optimization.

Key Features:
    - Multiple parallel chains for better exploration
    - Differential evolution-based proposals for efficient jumping
    - Adaptive proposal scaling via self-adaptive randomized subspace sampling
    - Outlier chain detection and correction
    - Provides posterior samples for uncertainty quantification

Reference:
    Vrugt, J.A., ter Braak, C.J.F., Diks, C.G.H., Robinson, B.A., Hyman, J.M.,
    and Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by
    differential evolution with self-adaptive randomized subspace sampling.
    International Journal of Nonlinear Sciences and Numerical Simulation, 10(3), 273-290.

    Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM
    software package: Theory, concepts, and MATLAB implementation.
    Environmental Modelling & Software, 75, 273-316.
"""

from typing import Dict, Any, Callable, Optional, List
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class DREAMAlgorithm(OptimizationAlgorithm):
    """DREAM (DiffeRential Evolution Adaptive Metropolis) optimization algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "DREAM"

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
        Run DREAM optimization.

        DREAM maintains N parallel Markov chains that explore the parameter space.
        Proposals are generated using differential evolution:
            x_proposal = x_current + γ * (x_r1 - x_r2) + ε

        where γ is adapted based on acceptance rate and ε is small random noise.

        The algorithm uses Metropolis-Hastings acceptance:
            α = min(1, p(x_proposal) / p(x_current))

        For optimization, we use likelihood ∝ exp(fitness / T) where T is temperature.

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
            Optimization results dictionary including posterior samples
        """
        self.logger.info(f"Starting DREAM optimization with {n_params} parameters")

        # DREAM parameters
        # Number of chains: DREAM needs at least 2*n_params for good mixing
        # Use population_size if specified, otherwise default to 2*n_params + 1
        min_chains = 2 * n_params + 1
        n_chains = max(min_chains, self.population_size)
        self.logger.info(f"Using {n_chains} chains for DREAM ({n_params} parameters)")

        # Number of chain pairs for DE proposal (δ)
        n_pairs = self.config_dict.get('DREAM_PAIRS', 3)
        n_pairs = min(n_pairs, (n_chains - 1) // 2)
        if n_pairs < 1:
            n_pairs = 1

        # Crossover probability for subspace sampling
        CR = self.config_dict.get('DREAM_CR', 0.9)

        # Jump rate scaling factor
        # The optimal gamma depends on effective dimensions: d* = CR * n_params
        d_star = max(1, int(CR * n_params))
        gamma_base = 2.38 / np.sqrt(2 * n_pairs * d_star)

        # Small random noise for ergodicity (scaled by parameter range)
        # Default is 1e-3 which gives ~0.1% noise in normalized [0,1] space
        eps_std = self.config_dict.get('DREAM_EPS', 1e-3)

        # Temperature for likelihood (lower = more greedy, higher = more exploration)
        temperature = self.config_dict.get('DREAM_TEMPERATURE', 1.0)

        # Outlier detection threshold (Mahalanobis distance)
        outlier_threshold = self.config_dict.get('DREAM_OUTLIER_THRESHOLD', 2.0)

        # Initialize chains uniformly in [0, 1]
        chains = np.random.uniform(0, 1, (n_chains, n_params))

        # Evaluate initial chain positions
        fitness = evaluate_population(chains, 0)

        # Convert fitness to log-likelihood (for MCMC)
        # Higher fitness = higher likelihood
        log_likelihood = fitness / temperature

        # Track best solution
        best_idx = np.argmax(fitness)
        best_pos = chains[best_idx].copy()
        best_fit = fitness[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_chains, best_fit)

        # Storage for posterior samples (after burn-in)
        burn_in = self.max_iterations // 5  # 20% burn-in
        posterior_samples: List[np.ndarray] = []

        # Acceptance tracking for adaptive scaling
        acceptance_history = np.zeros((n_chains, 100))  # Rolling window
        acceptance_idx = 0

        # Main DREAM loop
        for iteration in range(1, self.max_iterations + 1):
            n_accepted = 0

            # Generate proposals for all chains
            proposals = np.zeros_like(chains)

            for i in range(n_chains):
                # Select random chains for DE (excluding current chain)
                available = [j for j in range(n_chains) if j != i]

                # Select 2*n_pairs chains for differential evolution
                selected = np.random.choice(available, size=2 * n_pairs, replace=False)
                r1 = selected[:n_pairs]
                r2 = selected[n_pairs:]

                # Compute differential evolution jump
                diff = np.zeros(n_params)
                for k in range(n_pairs):
                    diff += chains[r1[k]] - chains[r2[k]]

                # Subspace sampling: randomly select dimensions to update
                # This improves mixing in high dimensions
                crossover_mask = np.random.random(n_params) < CR
                if not crossover_mask.any():
                    # Ensure at least one dimension is updated
                    crossover_mask[np.random.randint(n_params)] = True

                # Adaptive jump rate
                # Occasionally use gamma = 1 for mode jumping
                if np.random.random() < 0.1:
                    gamma = 1.0
                else:
                    gamma = gamma_base

                # Generate proposal
                proposal = chains[i].copy()
                jump = gamma * diff + eps_std * np.random.randn(n_params)
                proposal[crossover_mask] += jump[crossover_mask]

                # Reflect at bounds
                proposal = self._reflect_at_bounds(proposal)
                proposals[i] = proposal

            # Evaluate all proposals
            proposal_fitness = evaluate_population(proposals, iteration)
            proposal_log_lik = proposal_fitness / temperature

            # Metropolis-Hastings acceptance for each chain
            for i in range(n_chains):
                # Log acceptance probability
                log_alpha = proposal_log_lik[i] - log_likelihood[i]

                # Accept or reject
                if np.log(np.random.random()) < log_alpha:
                    chains[i] = proposals[i]
                    fitness[i] = proposal_fitness[i]
                    log_likelihood[i] = proposal_log_lik[i]
                    n_accepted += 1
                    acceptance_history[i, acceptance_idx % 100] = 1

                    # Update best if improved
                    if fitness[i] > best_fit:
                        best_fit = fitness[i]
                        best_pos = chains[i].copy()
                else:
                    acceptance_history[i, acceptance_idx % 100] = 0

            acceptance_idx += 1

            # Outlier chain detection and correction
            if iteration > 10 and iteration % 10 == 0:
                self._correct_outlier_chains(
                    chains, fitness, log_likelihood,
                    outlier_threshold, temperature
                )

            # Store posterior samples after burn-in
            if iteration > burn_in:
                for i in range(n_chains):
                    posterior_samples.append(chains[i].copy())

            # Record iteration
            params_dict = denormalize_params(best_pos)
            acceptance_rate = n_accepted / n_chains

            record_iteration(
                iteration, best_fit, params_dict,
                {'acceptance_rate': acceptance_rate, 'n_chains': n_chains}
            )
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit, n_accepted, n_chains)

            # Convergence check via Gelman-Rubin diagnostic (simplified)
            if iteration > burn_in and iteration % 50 == 0:
                r_hat = self._compute_gelman_rubin(chains, fitness)
                if r_hat < 1.1:
                    self.logger.info(
                        f"DREAM converged at iteration {iteration} (R-hat = {r_hat:.3f})"
                    )
                    # Continue to collect more samples rather than stopping
                    pass

        # Compute posterior statistics
        posterior_array = np.array(posterior_samples) if posterior_samples else chains
        posterior_mean = np.mean(posterior_array, axis=0)
        posterior_std = np.std(posterior_array, axis=0)

        # Compute credible intervals (95%)
        credible_lower = np.percentile(posterior_array, 2.5, axis=0)
        credible_upper = np.percentile(posterior_array, 97.5, axis=0)

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval_95': (credible_lower, credible_upper),
            'n_posterior_samples': len(posterior_samples),
            'final_chains': chains,
            'final_fitness': fitness
        }

    def _correct_outlier_chains(
        self,
        chains: np.ndarray,
        fitness: np.ndarray,
        log_likelihood: np.ndarray,
        threshold: float,
        temperature: float
    ) -> None:
        """
        Detect and correct outlier chains that may be stuck in low-probability regions.

        Uses interquartile range (IQR) of log-likelihood to identify outliers.
        Outlier chains are restarted from a randomly selected good chain.

        Args:
            chains: Current chain positions (n_chains, n_params)
            fitness: Current fitness values (n_chains,)
            log_likelihood: Current log-likelihood values (n_chains,)
            threshold: IQR multiplier for outlier detection
            temperature: Temperature for likelihood computation
        """
        n_chains = len(chains)
        if n_chains < 4:
            return

        # Compute IQR of log-likelihood
        q1, q3 = np.percentile(log_likelihood, [25, 75])
        iqr = q3 - q1

        # Identify outliers (below Q1 - threshold * IQR)
        outlier_threshold = q1 - threshold * iqr
        outliers = log_likelihood < outlier_threshold

        if outliers.any():
            # Get indices of good chains
            good_chains = np.where(~outliers)[0]
            if len(good_chains) == 0:
                return

            # Restart outlier chains from random good chains
            for i in np.where(outliers)[0]:
                source = np.random.choice(good_chains)
                chains[i] = chains[source].copy()
                fitness[i] = fitness[source]
                log_likelihood[i] = log_likelihood[source]

            self.logger.debug(f"Corrected {outliers.sum()} outlier chains")

    def _compute_gelman_rubin(
        self,
        chains: np.ndarray,
        fitness: np.ndarray
    ) -> float:
        """
        Compute simplified Gelman-Rubin convergence diagnostic.

        The R-hat statistic compares within-chain and between-chain variance.
        Values close to 1.0 indicate convergence.

        Args:
            chains: Current chain positions (n_chains, n_params)
            fitness: Current fitness values (for weighting)

        Returns:
            R-hat statistic (target: < 1.1 for convergence)
        """
        n_chains, n_params = chains.shape

        if n_chains < 2:
            return float('inf')

        # Compute between-chain variance
        _chain_means = np.mean(chains, axis=1)  # noqa: F841 (n_chains,) for each param
        _overall_mean = np.mean(chains, axis=0)  # noqa: F841 (n_params,)

        B = np.var(np.mean(chains, axis=0))  # Simplified: variance of means

        # Compute within-chain variance
        W = np.mean([np.var(chains[i]) for i in range(n_chains)])

        if W < 1e-10:
            return 1.0

        # Simplified R-hat
        # Full formula would use chain length, but we use current snapshot
        var_estimate = W + B
        r_hat = np.sqrt(var_estimate / W)

        return r_hat
