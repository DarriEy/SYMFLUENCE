#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian Optimization

A surrogate-based global optimization method that uses a probabilistic model
(Gaussian Process) to guide the search. Particularly effective for expensive
objective functions where each evaluation is costly.

Key Features:
    - Builds a surrogate model of the objective function
    - Uses acquisition functions to balance exploration vs exploitation
    - Efficient for expensive-to-evaluate objectives
    - Provides uncertainty estimates

Reference:
    Snoek, J., Larochelle, H., and Adams, R.P. (2012). Practical Bayesian
    Optimization of Machine Learning Algorithms. NIPS.

    Jones, D.R., Schonlau, M., and Welch, W.J. (1998). Efficient Global
    Optimization of Expensive Black-Box Functions. Journal of Global
    Optimization, 13(4), 455-492.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np

from .base_algorithm import OptimizationAlgorithm
from .config_schema import BODefaults


class BayesianOptimizationAlgorithm(OptimizationAlgorithm):
    """Bayesian Optimization using Gaussian Process surrogate."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "BAYESIAN-OPT"

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
        Run Bayesian Optimization.

        The algorithm:
        1. Starts with initial random samples to build surrogate
        2. Fits a Gaussian Process (GP) to observed data
        3. Uses acquisition function to select next point
        4. Evaluates objective and updates GP
        5. Repeats until budget exhausted

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
        self.logger.info(f"Starting Bayesian Optimization with {n_params} parameters")

        # BO parameters using standardized config access
        # Number of initial samples for building the GP surrogate
        # Rule of thumb: max(5, 2*n_params) ensures reasonable initial coverage
        # (Snoek 2012, Section 3)
        n_initial = self._get_config_value(
            lambda: self.config.optimization.bo_initial_samples,
            default=max(5, BODefaults.INITIAL_SAMPLES_FACTOR * n_params),
            dict_key='BO_INITIAL_SAMPLES'
        )

        # Acquisition function: 'ei' (Expected Improvement), 'ucb', or 'pi'
        # EI is the most commonly used and robust choice
        # (Jones 1998, Section 4)
        acquisition = self._get_config_value(
            lambda: self.config.optimization.bo_acquisition,
            default=BODefaults.ACQUISITION,
            dict_key='BO_ACQUISITION'
        )

        # Exploration parameter for EI/PI acquisition
        # xi=0.01 provides slight exploration bias
        # Higher values increase exploration
        # (Snoek 2012, Section 2.1)
        xi = self._get_config_value(
            lambda: self.config.optimization.bo_xi,
            default=BODefaults.XI,
            dict_key='BO_XI'
        )

        # UCB parameter (kappa)
        # kappa=2.576 corresponds to 99% confidence interval
        # Higher values increase exploration
        # (Srinivas et al. 2010, GP-UCB)
        kappa = self._get_config_value(
            lambda: self.config.optimization.bo_kappa,
            default=BODefaults.KAPPA,
            dict_key='BO_KAPPA'
        )

        # Number of restarts for acquisition function optimization
        # More restarts improve chance of finding global optimum of acquisition
        # (Snoek 2012, Section 3.2)
        n_restarts = self._get_config_value(
            lambda: self.config.optimization.bo_restarts,
            default=BODefaults.RESTARTS,
            dict_key='BO_RESTARTS'
        )

        # Validate BO configuration
        valid, warning = BODefaults.validate_acquisition(acquisition)
        if not valid:
            self.logger.warning(f"BO acquisition validation: {warning}")

        self.logger.info(
            f"BO settings: {n_initial} initial samples, acquisition={acquisition}, "
            f"xi={xi}, kappa={kappa}"
        )

        # Storage for observations
        X_observed: List[np.ndarray] = []
        y_observed: List[float] = []

        # Initial sampling (Latin Hypercube for better coverage)
        initial_samples = self._latin_hypercube_sample(n_initial, n_params)
        initial_fitness = evaluate_population(initial_samples, 0)

        for i in range(n_initial):
            X_observed.append(initial_samples[i])
            y_observed.append(initial_fitness[i])

        # Track best
        best_idx = np.argmax(y_observed)
        best_pos = X_observed[best_idx].copy()
        best_fit = y_observed[best_idx]

        # Record initial state
        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        if log_initial_population:
            log_initial_population(self.name, n_initial, best_fit)

        # Main BO loop
        for iteration in range(1, self.max_iterations + 1):
            # Convert to arrays
            X = np.array(X_observed)
            y = np.array(y_observed)

            # Fit GP surrogate
            gp_mean, gp_std, gp_params = self._fit_gp(X, y)

            # Optimize acquisition function to find next point
            next_point = self._optimize_acquisition(
                gp_mean, gp_std, y, n_params, acquisition, xi, kappa, n_restarts
            )

            # Evaluate new point
            next_fitness = evaluate_solution(next_point, iteration)

            # Update observations
            X_observed.append(next_point)
            y_observed.append(next_fitness)

            # Update best
            if next_fitness > best_fit:
                best_fit = next_fitness
                best_pos = next_point.copy()

            # Record iteration
            params_dict = denormalize_params(best_pos)
            record_iteration(
                iteration, best_fit, params_dict,
                {'gp_lengthscale': gp_params.get('lengthscale', 0)}
            )
            update_best(best_fit, params_dict, iteration)

            # Log progress
            log_progress(self.name, iteration, best_fit, 1, 1)

        self.logger.info(f"Bayesian Optimization complete: {len(y_observed)} evaluations")

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos),
            'X_observed': np.array(X_observed),
            'y_observed': np.array(y_observed),
            'n_evaluations': len(y_observed),
        }

    def _latin_hypercube_sample(self, n_samples: int, n_params: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n_samples, n_params))
        for j in range(n_params):
            intervals = np.arange(n_samples)
            np.random.shuffle(intervals)
            for i in range(n_samples):
                low = intervals[i] / n_samples
                high = (intervals[i] + 1) / n_samples
                samples[i, j] = np.random.uniform(low, high)
        return samples

    def _fit_gp(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Callable, Callable, Dict]:
        """
        Fit a simple Gaussian Process surrogate.

        Uses RBF kernel with automatic lengthscale estimation.

        Args:
            X: Observed inputs (n_obs, n_params)
            y: Observed outputs (n_obs,)

        Returns:
            Tuple of (mean_function, std_function, hyperparameters)
        """
        n_obs, n_params = X.shape

        # Normalize y for numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_normalized = (y - y_mean) / y_std

        # Estimate lengthscale from data spread
        lengthscale = np.std(X, axis=0) + 1e-8

        # Noise variance (nugget for numerical stability)
        noise_var = 1e-6

        def rbf_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
            """RBF (squared exponential) kernel."""
            # Scaled squared distance
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            scaled_diff = diff / lengthscale
            sq_dist = np.sum(scaled_diff ** 2, axis=2)
            return np.exp(-0.5 * sq_dist)

        # Compute kernel matrix
        K = rbf_kernel(X, X) + noise_var * np.eye(n_obs)

        # Cholesky decomposition for stable inversion
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if Cholesky fails
            alpha = np.linalg.lstsq(K, y_normalized, rcond=None)[0]
            L = None

        def predict_mean(X_new: np.ndarray) -> np.ndarray:
            """Predict mean at new points."""
            K_star = rbf_kernel(X_new.reshape(-1, n_params), X)
            mean_normalized = K_star @ alpha
            return mean_normalized * y_std + y_mean

        def predict_std(X_new: np.ndarray) -> np.ndarray:
            """Predict standard deviation at new points."""
            X_new = X_new.reshape(-1, n_params)
            K_star = rbf_kernel(X_new, X)
            K_star_star = rbf_kernel(X_new, X_new)

            if L is not None:
                v = np.linalg.solve(L, K_star.T)
                var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
            else:
                var = np.diag(K_star_star) - np.diag(K_star @ np.linalg.lstsq(K, K_star.T, rcond=None)[0])

            var = np.maximum(var, 1e-10)  # Ensure positive
            return np.sqrt(var) * y_std

        params = {'lengthscale': np.mean(lengthscale), 'noise_var': noise_var}
        return predict_mean, predict_std, params

    def _optimize_acquisition(
        self,
        gp_mean: Callable,
        gp_std: Callable,
        y_observed: np.ndarray,
        n_params: int,
        acquisition: str,
        xi: float,
        kappa: float,
        n_restarts: int
    ) -> np.ndarray:
        """
        Optimize acquisition function to find next evaluation point.

        Args:
            gp_mean: GP mean prediction function
            gp_std: GP std prediction function
            y_observed: Observed fitness values
            n_params: Number of parameters
            acquisition: Acquisition function type
            xi: Exploration parameter
            kappa: UCB parameter
            n_restarts: Number of random restarts

        Returns:
            Next point to evaluate
        """
        y_best = np.max(y_observed)

        def acquisition_function(x: np.ndarray) -> float:
            """Compute acquisition value (to maximize)."""
            x = np.clip(x, 0, 1)
            mu = gp_mean(x)[0]
            sigma = gp_std(x)[0]

            if sigma < 1e-10:
                return mu

            if acquisition == 'ei':
                # Expected Improvement
                z = (mu - y_best - xi) / sigma
                ei = sigma * (z * self._norm_cdf(z) + self._norm_pdf(z))
                return ei
            elif acquisition == 'ucb':
                # Upper Confidence Bound
                return mu + kappa * sigma
            elif acquisition == 'pi':
                # Probability of Improvement
                z = (mu - y_best - xi) / sigma
                return self._norm_cdf(z)
            else:
                return mu  # Default to mean

        # Multi-start optimization of acquisition function
        best_acq = float('-inf')
        best_x = np.random.uniform(0, 1, n_params)

        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(0, 1, n_params)

            # Simple gradient-free optimization (coordinate descent)
            x = x0.copy()
            for _ in range(50):  # Local optimization steps
                for d in range(n_params):
                    # Line search along dimension d
                    best_val = acquisition_function(x)
                    for delta in [-0.1, -0.05, 0.05, 0.1]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + delta, 0, 1)
                        val = acquisition_function(x_new)
                        if val > best_val:
                            best_val = val
                            x = x_new

            acq_val = acquisition_function(x)
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = x.copy()

        return best_x

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
