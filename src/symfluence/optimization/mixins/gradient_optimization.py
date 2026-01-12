"""
Gradient Optimization Mixin

Provides gradient-based optimization methods (ADAM, L-BFGS) via finite differences.
These methods are useful for smooth optimization landscapes and can converge
faster than population-based methods in some cases.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class GradientOptimizationMixin:
    """
    Mixin class providing gradient-based optimization via finite differences.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger
    - self.param_manager: Parameter manager with normalize/denormalize methods
    - self._evaluate_solution: Method to evaluate a parameter set

    Provides:
    - Adam optimizer
    - L-BFGS optimizer
    - Finite difference gradient computation
    - Gradient clipping
    """

    # =========================================================================
    # Configuration
    # =========================================================================

    @property
    def gradient_epsilon(self) -> float:
        """Epsilon for finite difference gradient computation."""
        return self.config.get('GRADIENT_EPSILON', 1e-4)

    @property
    def gradient_clip_value(self) -> float:
        """Maximum gradient magnitude for clipping."""
        return self.config.get('GRADIENT_CLIP_VALUE', 1.0)

    # =========================================================================
    # Gradient computation
    # =========================================================================

    def compute_fd_gradients(
        self,
        x: np.ndarray,
        evaluate_func: Callable[[np.ndarray], float],
        epsilon: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute gradients using central finite differences.

        Args:
            x: Current parameter values (normalized)
            evaluate_func: Function to evaluate fitness given parameters
            epsilon: Perturbation size (default: self.gradient_epsilon)

        Returns:
            Tuple of (current fitness, gradient array)
        """
        if epsilon is None:
            epsilon = self.gradient_epsilon

        n_params = len(x)
        gradient = np.zeros(n_params)

        # Evaluate at current point
        f_center = evaluate_func(x)

        # Compute central differences
        for i in range(n_params):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] = min(1.0, x[i] + epsilon)
            x_minus[i] = max(0.0, x[i] - epsilon)

            f_plus = evaluate_func(x_plus)
            f_minus = evaluate_func(x_minus)

            # Central difference (for maximization, gradient points uphill)
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)

        return f_center, gradient

    def compute_fd_gradients_forward(
        self,
        x: np.ndarray,
        f_x: float,
        evaluate_func: Callable[[np.ndarray], float],
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute gradients using forward finite differences (faster, less accurate).

        Args:
            x: Current parameter values (normalized)
            f_x: Function value at x (avoids recomputation)
            evaluate_func: Function to evaluate fitness
            epsilon: Perturbation size

        Returns:
            Gradient array
        """
        if epsilon is None:
            epsilon = self.gradient_epsilon

        n_params = len(x)
        gradient = np.zeros(n_params)

        for i in range(n_params):
            x_plus = x.copy()
            x_plus[i] = min(1.0, x[i] + epsilon)

            f_plus = evaluate_func(x_plus)
            gradient[i] = (f_plus - f_x) / epsilon

        return gradient

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        Clip gradient to prevent exploding gradients.

        Args:
            gradient: Gradient array

        Returns:
            Clipped gradient
        """
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip_value:
            gradient = gradient * (self.gradient_clip_value / norm)
        return gradient

    # =========================================================================
    # Adam optimizer
    # =========================================================================

    def _run_adam(
        self,
        evaluate_func: Callable[[np.ndarray], float],
        initial_x: Optional[np.ndarray] = None,
        steps: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Run Adam optimization.

        Args:
            evaluate_func: Function to evaluate (normalized params -> fitness)
            initial_x: Initial normalized parameters (default: midpoint)
            steps: Number of optimization steps
            lr: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            eps: Small constant for numerical stability

        Returns:
            Tuple of (best parameters, best fitness, history)
        """
        n_params = len(self.param_manager.all_param_names)

        # Initialize
        if initial_x is None:
            x = np.full(n_params, 0.5)  # Start at midpoint
        else:
            x = initial_x.copy()

        # Adam state
        m = np.zeros(n_params)  # First moment
        v = np.zeros(n_params)  # Second moment

        # Track best
        best_x = x.copy()
        best_fitness = float('-inf')
        history = []

        for step in range(steps):
            # Compute gradients
            fitness, gradient = self.compute_fd_gradients(x, evaluate_func)

            # Clip gradient
            gradient = self.clip_gradient(gradient)

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()

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

            # Record history
            history.append({
                'step': step,
                'fitness': fitness,
                'best_fitness': best_fitness,
                'lr': lr,
                'grad_norm': np.linalg.norm(gradient),
            })

            if step % 10 == 0:
                self.logger.info(
                    f"Adam step {step}/{steps}: fitness={fitness:.4f}, "
                    f"best={best_fitness:.4f}, grad_norm={np.linalg.norm(gradient):.4f}"
                )

        return best_x, best_fitness, history

    # =========================================================================
    # L-BFGS optimizer
    # =========================================================================

    def _run_lbfgs(
        self,
        evaluate_func: Callable[[np.ndarray], float],
        initial_x: Optional[np.ndarray] = None,
        steps: int = 50,
        lr: float = 0.1,
        history_size: int = 10,
        c1: float = 1e-4,
        c2: float = 0.9
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Run L-BFGS optimization with line search.

        Args:
            evaluate_func: Function to evaluate (normalized params -> fitness)
            initial_x: Initial normalized parameters
            steps: Maximum number of steps
            lr: Initial step size
            history_size: Number of past gradients to store
            c1: Armijo condition constant
            c2: Wolfe condition constant

        Returns:
            Tuple of (best parameters, best fitness, history)
        """
        n_params = len(self.param_manager.all_param_names)

        # Initialize
        if initial_x is None:
            x = np.full(n_params, 0.5)
        else:
            x = initial_x.copy()

        # L-BFGS history
        s_history = []  # Position differences
        y_history = []  # Gradient differences

        # Track best
        best_x = x.copy()
        best_fitness = float('-inf')
        history = []

        # Initial gradient
        fitness, gradient = self.compute_fd_gradients(x, evaluate_func)
        gradient = self.clip_gradient(gradient)

        for step in range(steps):
            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()

            # Compute search direction using L-BFGS two-loop recursion
            direction = self._lbfgs_direction(gradient, s_history, y_history)

            # Line search
            step_size, new_fitness, new_gradient = self._line_search(
                x, direction, fitness, gradient, evaluate_func, lr, c1, c2
            )

            if step_size is None:
                # Line search failed, use gradient descent
                self.logger.warning(f"L-BFGS line search failed at step {step}, using gradient descent")
                step_size = lr / (step + 1)
                x_new = x + step_size * gradient  # gradient ascent
                x_new = np.clip(x_new, 0, 1)
                new_fitness, new_gradient = self.compute_fd_gradients(x_new, evaluate_func)
            else:
                x_new = x + step_size * direction
                x_new = np.clip(x_new, 0, 1)

            new_gradient = self.clip_gradient(new_gradient)

            # Update history
            s = x_new - x
            y = new_gradient - gradient

            if np.dot(y, s) > 1e-10:  # Curvature condition
                s_history.append(s)
                y_history.append(y)

                if len(s_history) > history_size:
                    s_history.pop(0)
                    y_history.pop(0)

            # Record history
            history.append({
                'step': step,
                'fitness': fitness,
                'best_fitness': best_fitness,
                'step_size': step_size or lr,
                'grad_norm': np.linalg.norm(gradient),
            })

            # Update state
            x = x_new
            fitness = new_fitness
            gradient = new_gradient

            if step % 10 == 0:
                self.logger.info(
                    f"L-BFGS step {step}/{steps}: fitness={fitness:.4f}, "
                    f"best={best_fitness:.4f}"
                )

            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                self.logger.info(f"L-BFGS converged at step {step}")
                break

        return best_x, best_fitness, history

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
            gamma = np.dot(s_history[-1], y_history[-1]) / (np.dot(y_history[-1], y_history[-1]) + 1e-10)
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
            f_new, grad_new = self.compute_fd_gradients(x_new, evaluate_func)

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
