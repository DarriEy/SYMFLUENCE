#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adam Gradient-Based Optimization Algorithm

Uses finite difference gradient computation for derivative-free optimization.
Adam is particularly effective for smooth optimization landscapes and can
converge faster than population-based methods in some cases.

Reference:
    Kingma, D.P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization.
    ICLR 2015.
"""

from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class AdamAlgorithm(OptimizationAlgorithm):
    """Adam gradient-based optimization algorithm using finite differences."""

    @property
    def name(self) -> str:
        return "ADAM"

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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Adam optimization with finite difference gradients.

        Args:
            n_params: Number of parameters
            evaluate_solution: Callback to evaluate a single solution
            evaluate_population: Callback to evaluate a population
            denormalize_params: Callback to denormalize parameters
            record_iteration: Callback to record iteration
            update_best: Callback to update best solution
            log_progress: Callback to log progress
            **kwargs: Additional parameters (steps, lr, beta1, beta2)

        Returns:
            Optimization results dictionary
        """
        # Adam hyperparameters from config or kwargs
        steps = kwargs.get('steps', self.config.get('ADAM_STEPS', self.max_iterations))
        lr = kwargs.get('lr', self.config.get('ADAM_LR', 0.01))
        beta1 = kwargs.get('beta1', self.config.get('ADAM_BETA1', 0.9))
        beta2 = kwargs.get('beta2', self.config.get('ADAM_BETA2', 0.999))
        eps = kwargs.get('eps', self.config.get('ADAM_EPS', 1e-8))
        gradient_epsilon = self.config.get('GRADIENT_EPSILON', 1e-4)
        gradient_clip = self.config.get('GRADIENT_CLIP_VALUE', 1.0)

        self.logger.info(f"Starting Adam optimization with {n_params} parameters")
        self.logger.info(f"  Steps: {steps}, LR: {lr}, Beta1: {beta1}, Beta2: {beta2}")

        # Initialize at midpoint of normalized space
        x = np.full(n_params, 0.5)

        # Adam state
        m = np.zeros(n_params)  # First moment
        v = np.zeros(n_params)  # Second moment

        # Track best
        best_x = x.copy()
        best_fitness = float('-inf')

        for step in range(steps):
            # Compute gradients using finite differences
            fitness, gradient = self._compute_gradients(
                x, evaluate_solution, gradient_epsilon
            )

            # Clip gradient
            gradient = self._clip_gradient(gradient, gradient_clip)

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()

            # Record iteration
            params_dict = denormalize_params(best_x)
            record_iteration(step, best_fitness, params_dict)
            update_best(best_fitness, params_dict, step)

            # Adam update
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))

            # Update parameters (gradient ascent for maximization)
            x = x + lr * m_hat / (np.sqrt(v_hat) + eps)

            # Clip to [0, 1]
            x = np.clip(x, 0, 1)

            # Log progress
            if step % 10 == 0:
                log_progress(self.name, step, best_fitness)

        return {
            'best_solution': best_x,
            'best_score': best_fitness,
            'best_params': denormalize_params(best_x)
        }

    def _compute_gradients(
        self,
        x: np.ndarray,
        evaluate_func: Callable,
        epsilon: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute gradients using central finite differences.

        Args:
            x: Current parameter values (normalized)
            evaluate_func: Function to evaluate fitness
            epsilon: Perturbation size

        Returns:
            Tuple of (current fitness, gradient array)
        """
        n_params = len(x)
        gradient = np.zeros(n_params)

        # Evaluate at current point
        f_center = evaluate_func(x, 0)

        # Compute central differences
        for i in range(n_params):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] = min(1.0, x[i] + epsilon)
            x_minus[i] = max(0.0, x[i] - epsilon)

            f_plus = evaluate_func(x_plus, 0)
            f_minus = evaluate_func(x_minus, 0)

            # Central difference (for maximization, gradient points uphill)
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)

        return f_center, gradient

    def _clip_gradient(self, gradient: np.ndarray, clip_value: float) -> np.ndarray:
        """Clip gradient to prevent exploding gradients."""
        norm = np.linalg.norm(gradient)
        if norm > clip_value:
            gradient = gradient * (clip_value / norm)
        return gradient
