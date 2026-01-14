#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adam Gradient-Based Optimization Algorithm.

ADAM (Adaptive Moment Estimation) is a gradient-based optimizer that uses
finite-difference gradients and adaptive learning rates. It combines first-moment
(momentum) and second-moment (RMSprop) estimation for efficient convergence.

Useful for hydrological model calibration when:
- Objective function is relatively smooth (limited noise)
- Parameter space is not highly multi-modal
- Gradient computation via finite differences is feasible
- Number of function evaluations is limited

Note: Uses central finite differences for gradient computation, making it
derivative-free but requiring ~2N function evaluations per iteration (N = n_params).

References:
    Kingma, D.P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization.
    In Proceedings of the 3rd International Conference on Learning Representations (ICLR).
"""

from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class AdamAlgorithm(OptimizationAlgorithm):
    """Adam gradient-based optimization using finite-difference gradients.

    ADAM maintains first and second moment estimates of the gradient:
    - m = exponential moving average of gradients (momentum)
    - v = exponential moving average of squared gradients (adaptive learning rate)
    - Uses bias correction to account for initialization bias

    Algorithm Overview:
        1. Initialize parameters at normalized space midpoint (0.5)
        2. For each step:
           a. Compute finite-difference gradients (central differences)
           b. Update first moment: m ← β1*m + (1-β1)*∇f
           c. Update second moment: v ← β2*v + (1-β2)*∇f²
           d. Bias correct: m̂ = m / (1 - β1^t), v̂ = v / (1 - β2^t)
           e. Update parameters: x ← x + α * m̂ / (√v̂ + ε)
           f. Clip to [0,1] bounds
        3. Return best solution found

    Gradient Computation:
        Uses central finite differences: ∇f_i = (f(x+ε*e_i) - f(x-ε*e_i)) / (2ε)
        - e_i: unit vector in dimension i
        - ε: perturbation size (typically 1e-4)
        - Cost: 2*n_params function evaluations per step

    Hyperparameters:
        - α (lr): Learning rate (default: 0.01)
        - β1: First moment decay (default: 0.9)
        - β2: Second moment decay (default: 0.999)
        - ε: Numerical stability (default: 1e-8)
        - steps: Maximum iterations (default: uses max_iterations)
    """

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
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
        """Run ADAM optimization with finite-difference gradients.

        Initializes parameters at normalized space midpoint and iteratively
        improves using adaptive moment estimation and gradient ascent.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Function to evaluate single parameter vector
                              Call: score = evaluate_solution(x_normalized, step_id)
            evaluate_population: Function to evaluate population
                                (unused in ADAM, single-solution method)
            denormalize_params: Function to convert normalized [0,1] to actual parameters
            record_iteration: Function to record iteration results
            update_best: Function to update best solution found
            log_progress: Function to log progress messages
            evaluate_population_objectives: Unused for ADAM
            **kwargs: Optional hyperparameters:
                     - steps: Number of iterations (default: max_iterations from config)
                     - lr: Learning rate (default: 0.01)
                     - beta1: First moment decay (default: 0.9)
                     - beta2: Second moment decay (default: 0.999)
                     - eps: Numerical stability constant (default: 1e-8)

        Returns:
            Dict with keys:
            - best_solution: Best parameter vector found (normalized [0,1])
            - best_score: Highest objective value achieved
            - best_params: Denormalized best parameters (dictionary)
        """
        # Adam hyperparameters from config or kwargs
        steps = kwargs.get('steps', self.config_dict.get('ADAM_STEPS', self.max_iterations))
        lr = kwargs.get('lr', self.config_dict.get('ADAM_LR', 0.01))
        beta1 = kwargs.get('beta1', self.config.get('ADAM_BETA1', 0.9))
        beta2 = kwargs.get('beta2', self.config.get('ADAM_BETA2', 0.999))
        eps = kwargs.get('eps', self.config_dict.get('ADAM_EPS', 1e-8))
        gradient_epsilon = self.config_dict.get('GRADIENT_EPSILON', 1e-4)
        gradient_clip = self.config_dict.get('GRADIENT_CLIP_VALUE', 1.0)

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
        """Compute gradients using central finite differences.

        Central finite difference formula (more accurate than forward/backward):
            ∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2*ε)

        where e_i is unit vector with 1 in dimension i and 0 elsewhere.

        Procedure:
            1. Evaluate fitness at current point (f_center)
            2. For each parameter dimension:
               a. Perturb +ε in that dimension
               b. Evaluate fitness at perturbed point
               c. Perturb -ε in that dimension
               d. Evaluate fitness at perturbed point
               e. Compute central difference
            3. Return current fitness and gradient array

        Cost: 2*n_params + 1 function evaluations

        Args:
            x: Current parameter values (normalized [0,1])
            evaluate_func: Function to evaluate fitness: f = evaluate_func(x, step_id)
            epsilon: Perturbation size (typically 1e-4)

        Returns:
            Tuple of (f_center, gradient_array):
            - f_center: Fitness at current point
            - gradient_array: Approximate gradient (shape: n_params)
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
        """Clip gradient norm to prevent exploding gradients.

        Gradient clipping prevents numerical instability when finite-difference
        gradients become very large (e.g., near discontinuities or noise).

        Algorithm:
            1. Compute gradient norm: ||g|| = √(Σ g_i²)
            2. If ||g|| > clip_value:
                 g_clipped = g * (clip_value / ||g||)
            3. Otherwise: g_clipped = g

        This rescales the gradient vector to have maximum norm of clip_value,
        preserving direction but reducing magnitude.

        Args:
            gradient: Gradient vector (shape: n_params)
            clip_value: Maximum L2 norm allowed (typically 1.0)

        Returns:
            np.ndarray: Clipped gradient with ||g|| ≤ clip_value
        """
        norm = np.linalg.norm(gradient)
        if norm > clip_value:
            gradient = gradient * (clip_value / norm)
        return gradient
