#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""L-BFGS Gradient-Based Optimization Algorithm.

Limited-memory BFGS (L-BFGS) is a quasi-Newton optimization method that
approximates the inverse Hessian matrix using a limited history of past
gradients and position changes. This enables quasi-Newton methods to scale
to high-dimensional problems without storing full Hessian matrix.

Effective for hydrological model calibration when:
- Objective function is relatively smooth with limited noise
- Parameter space has low multi-modality
- Memory-efficient method is needed (Hessian approximation via history only)
- Convergence speed is important (quasi-Newton vs first-order methods)

Note: Uses central finite differences for gradient computation, making it
derivative-free but requiring ~2N function evaluations per iteration (N = n_params).

References:
    Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage.
    Mathematics of Computation, 35(151), 773-782.

    Liu, D.C. and Nocedal, J. (1989). On the limited memory BFGS method for
    large scale optimization. Mathematical Programming, 45, 503-528.
"""

from typing import Dict, Any, Callable, Optional, Tuple, List
import numpy as np

from .base_algorithm import OptimizationAlgorithm


class LBFGSAlgorithm(OptimizationAlgorithm):
    """L-BFGS quasi-Newton optimization using finite-difference gradients.

    L-BFGS approximates the Hessian inverse using only a limited history of
    gradients and position changes, enabling efficient quasi-Newton optimization
    without storing the full Hessian matrix.

    Algorithm Overview:
        1. Initialize parameters at normalized space midpoint (0.5)
        2. Compute initial gradient via finite differences
        3. For each step:
           a. Compute search direction using L-BFGS two-loop recursion
           b. Line search: find step size satisfying Wolfe conditions
           c. Update position
           d. Compute new gradient
           e. Store position and gradient differences in history
           f. Maintain limited history (e.g., last 10 updates)
        4. Terminate when gradient norm < 1e-6 (convergence) or max steps reached
        5. Return best solution found

    Two-Loop Recursion:
        Efficiently computes search direction p = H*g where H is approximate
        Hessian inverse, using only stored history (s_k, y_k) pairs without
        explicitly forming Hessian.

    Line Search (Wolfe Conditions):
        Ensures step size satisfies:
        1. Sufficient decrease (Armijo): f(x_new) ≥ f(x) + c1*α*⟨∇f,d⟩
        2. Curvature condition (strong Wolfe): |⟨∇f(x_new),d⟩| ≤ c2*|⟨∇f,d⟩|

    Hyperparameters:
        - α (lr): Initial step size for line search (default: 0.1)
        - history_size: # of (s,y) pairs to retain (default: 10)
        - steps: Maximum iterations (default: uses max_iterations)
        - c1, c2: Wolfe condition parameters (default: 1e-4, 0.9)
    """

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "LBFGS"

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
        """Run L-BFGS optimization with finite-difference gradients and line search.

        Implements full L-BFGS algorithm with Wolfe line search and gradient
        convergence detection.

        Args:
            n_params: Number of parameters to optimize
            evaluate_solution: Function to evaluate single parameter vector
                              Call: score = evaluate_solution(x_normalized, step_id)
            evaluate_population: Unused in L-BFGS (single-solution method)
            denormalize_params: Function to convert normalized [0,1] to actual parameters
            record_iteration: Function to record iteration results
            update_best: Function to update best solution found
            log_progress: Function to log progress messages
            evaluate_population_objectives: Unused for L-BFGS
            **kwargs: Optional hyperparameters:
                     - steps: Maximum iterations (default: max_iterations from config)
                     - lr: Initial step size for line search (default: 0.1)
                     - history_size: # of (s,y) pairs retained (default: 10)
                     - c1: Armijo parameter (default: 1e-4)
                     - c2: Wolfe curvature parameter (default: 0.9)

        Returns:
            Dict with keys:
            - best_solution: Best parameter vector found (normalized [0,1])
            - best_score: Highest objective value achieved
            - best_params: Denormalized best parameters (dictionary)
        """
        # L-BFGS hyperparameters from config or kwargs
        steps = kwargs.get('steps', self.config.get('LBFGS_STEPS', self.max_iterations))
        lr = kwargs.get('lr', self.config.get('LBFGS_LR', 0.1))
        history_size = kwargs.get('history_size', self.config.get('LBFGS_HISTORY_SIZE', 10))
        c1 = kwargs.get('c1', self.config.get('LBFGS_C1', 1e-4))  # Armijo condition
        c2 = kwargs.get('c2', self.config.get('LBFGS_C2', 0.9))   # Wolfe condition
        gradient_epsilon = self.config.get('GRADIENT_EPSILON', 1e-4)
        gradient_clip = self.config.get('GRADIENT_CLIP_VALUE', 1.0)

        self.logger.info(f"Starting L-BFGS optimization with {n_params} parameters")
        self.logger.info(f"  Steps: {steps}, LR: {lr}, History size: {history_size}")

        # Initialize at midpoint of normalized space
        x = np.full(n_params, 0.5)

        # L-BFGS history
        s_history: List[np.ndarray] = []  # Position differences
        y_history: List[np.ndarray] = []  # Gradient differences

        # Track best
        best_x = x.copy()
        best_fitness = float('-inf')

        # Initial gradient
        fitness, gradient = self._compute_gradients(x, evaluate_solution, gradient_epsilon)
        gradient = self._clip_gradient(gradient, gradient_clip)

        for step in range(steps):
            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()

            # Record iteration
            params_dict = denormalize_params(best_x)
            record_iteration(step, best_fitness, params_dict)
            update_best(best_fitness, params_dict, step)

            # Compute search direction using L-BFGS two-loop recursion
            direction = self._lbfgs_direction(gradient, s_history, y_history)

            # Line search
            step_size, new_fitness, new_gradient = self._line_search(
                x, direction, fitness, gradient, evaluate_solution,
                lr, c1, c2, gradient_epsilon
            )

            if step_size is None:
                # Line search failed, use gradient descent
                self.logger.warning(f"L-BFGS line search failed at step {step}, using gradient descent")
                step_size = lr / (step + 1)
                x_new = x + step_size * gradient  # gradient ascent
                x_new = np.clip(x_new, 0, 1)
                new_fitness, new_gradient = self._compute_gradients(
                    x_new, evaluate_solution, gradient_epsilon
                )
            else:
                x_new = x + step_size * direction
                x_new = np.clip(x_new, 0, 1)

            new_gradient = self._clip_gradient(new_gradient, gradient_clip)

            # Update history
            s = x_new - x
            y = new_gradient - gradient

            if np.dot(y, s) > 1e-10:  # Curvature condition
                s_history.append(s)
                y_history.append(y)

                if len(s_history) > history_size:
                    s_history.pop(0)
                    y_history.pop(0)

            # Update state
            x = x_new
            fitness = new_fitness
            gradient = new_gradient

            # Log progress
            if step % 10 == 0:
                log_progress(self.name, step, best_fitness)

            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                self.logger.info(f"L-BFGS converged at step {step}")
                break

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

    def _lbfgs_direction(
        self,
        gradient: np.ndarray,
        s_history: List[np.ndarray],
        y_history: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.

        Args:
            gradient: Current gradient
            s_history: History of position differences
            y_history: History of gradient differences

        Returns:
            Search direction (for gradient ascent)
        """
        q = gradient.copy()
        m = len(s_history)
        alphas = []

        # First loop (backward)
        for i in range(m - 1, -1, -1):
            rho_i = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-10)
            alpha_i = rho_i * np.dot(s_history[i], q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_history[i]

        alphas.reverse()

        # Initial Hessian approximation
        if m > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / (
                np.dot(y_history[-1], y_history[-1]) + 1e-10
            )
        else:
            gamma = 1.0

        r = gamma * q

        # Second loop (forward)
        for i in range(m):
            rho_i = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-10)
            beta_i = rho_i * np.dot(y_history[i], r)
            r = r + (alphas[i] - beta_i) * s_history[i]

        return r  # For maximization, this is the ascent direction

    def _line_search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        f_x: float,
        grad_x: np.ndarray,
        evaluate_func: Callable,
        initial_step: float,
        c1: float,
        c2: float,
        gradient_epsilon: float,
        max_iter: int = 20
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """
        Backtracking line search with Wolfe conditions.

        Returns:
            Tuple of (step_size, new_fitness, new_gradient)
            step_size is None if line search failed
        """
        step_size = initial_step
        directional_deriv = np.dot(grad_x, direction)

        if directional_deriv <= 0:
            # Not an ascent direction
            return None, f_x, grad_x

        for _ in range(max_iter):
            x_new = np.clip(x + step_size * direction, 0, 1)
            f_new, grad_new = self._compute_gradients(x_new, evaluate_func, gradient_epsilon)

            # Armijo condition (sufficient increase for maximization)
            if f_new >= f_x + c1 * step_size * directional_deriv:
                # Curvature condition
                new_directional_deriv = np.dot(grad_new, direction)
                if new_directional_deriv >= c2 * directional_deriv:
                    return step_size, f_new, grad_new

            step_size *= 0.5

            if step_size < 1e-10:
                break

        return None, f_x, grad_x
