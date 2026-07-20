# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Gradient Optimization Mixin

Provides gradient-based optimization methods (ADAM, L-BFGS) via finite differences.
These methods are useful for smooth optimization landscapes and can converge
faster than population-based methods in some cases.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from symfluence.core.mixins import ConfigMixin

logger = logging.getLogger(__name__)


class GradientOptimizationMixin(ConfigMixin):
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
        return self._get_config_value(
            lambda: self.config.optimization.gradient_epsilon,
            default=1e-4
        )

    @property
    def gradient_clip_value(self) -> float:
        """Maximum gradient magnitude for clipping."""
        return self._get_config_value(
            lambda: self.config.optimization.gradient_clip_value,
            default=1.0
        )

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

            # Divide by the step actually taken, not the requested one. The
            # perturbations above are clamped to the [0, 1] box, so for a
            # parameter sitting on a bound the real span is epsilon rather
            # than 2*epsilon — dividing by 2*epsilon there halves the
            # gradient and points the search along a systematically wrong
            # direction exactly where it is most likely to be stuck.
            step = x_plus[i] - x_minus[i]
            gradient[i] = (f_plus - f_minus) / step if step > 0 else 0.0

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
            # Step backwards when the forward step is clamped away entirely.
            # A parameter resting on the upper bound previously produced
            # (f_x - f_x) / epsilon == 0 — an exactly zero gradient, telling
            # the optimizer the parameter has no effect and freezing it on
            # the bound for the rest of the run.
            step = x_plus[i] - x[i]
            if step <= 0:
                x_minus = x.copy()
                x_minus[i] = max(0.0, x[i] - epsilon)
                step = x[i] - x_minus[i]
                if step <= 0:
                    gradient[i] = 0.0  # degenerate box: lower == upper
                    continue
                gradient[i] = (f_x - evaluate_func(x_minus)) / step
            else:
                gradient[i] = (f_plus - f_x) / step

        return gradient

    #: Fraction of failed line searches above which the search is reported as
    #: degenerate. Occasional failures are normal; a majority means the
    #: gradient is not a usable descent direction.
    _LINE_SEARCH_FAILURE_ALERT = 0.5
    #: Minimum steps before judging, so short runs do not trip the check.
    _LINE_SEARCH_MIN_STEPS = 20

    def _warn_if_line_search_degenerate(
        self, failures: int, steps: int, epsilon: float
    ) -> None:
        """Report once when line searches fail so often the method degrades.

        Every failure already logs, but a wall of identical warnings reads as
        noise, and the run still reports success — a paper calibration was
        observed taking the steepest-descent fallback on 110 of 125 steps,
        i.e. not running L-BFGS at all, with nothing saying so. The usual
        cause is a gradient dominated by evaluation noise: JAX defaults to
        float32 (~1e-7 relative), and a central difference over ``epsilon``
        divides that noise by ``2*epsilon``, so an epsilon below
        ``sqrt(float32 eps) ~= 3.5e-4`` amplifies it rather than resolving a
        slope.
        """
        if steps < self._LINE_SEARCH_MIN_STEPS:
            return
        if failures / steps < self._LINE_SEARCH_FAILURE_ALERT:
            return
        if getattr(self, '_line_search_degenerate_reported', False):
            return
        self._line_search_degenerate_reported = True
        float32_floor = float(np.sqrt(np.finfo(np.float32).eps))
        hint = ""
        if epsilon < float32_floor:
            hint = (
                f" gradient_epsilon={epsilon:g} is below sqrt(float32 eps)="
                f"{float32_floor:.1e}, so if the model evaluates in float32 "
                f"(the JAX default) the gradient is noise, not slope — raise "
                f"gradient_epsilon or evaluate in float64."
            )
        self.logger.error(
            "Line search has failed on %d of %d steps (%.0f%%); the optimizer "
            "is running as steepest descent, not L-BFGS, and its result should "
            "not be read as a converged quasi-Newton solution.%s",
            failures, steps, 100 * failures / steps, hint,
        )

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

        # L-BFGS is formulated for minimization, but evaluate_func returns fitness
        # to maximize. Minimize g = -fitness internally; grad_g = -grad_fitness.
        def eval_min(xi: np.ndarray) -> Tuple[float, np.ndarray]:
            fit, ascent_grad = self.compute_fd_gradients(xi, evaluate_func)
            return -fit, -ascent_grad

        # L-BFGS history (minimization space)
        s_history: List[np.ndarray] = []  # Position differences s_k = x_{k+1} - x_k
        y_history: List[np.ndarray] = []  # Min-space gradient differences y_k

        # Track best (reported in fitness / maximization terms)
        best_x = x.copy()
        best_fitness = float('-inf')
        history = []

        # Initial minimization-space objective and gradient
        g, gradient = eval_min(x)
        gradient = self.clip_gradient(gradient)
        line_search_failures = 0

        for step in range(steps):
            fitness = -g

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_x = x.copy()

            # Two-loop recursion returns r ≈ H·grad_g; descent direction is -r.
            direction = -self._lbfgs_direction(gradient, s_history, y_history)

            # Line search in minimization space
            step_size, g_new, new_gradient = self._line_search(
                x, direction, g, gradient, eval_min, lr, c1, c2
            )

            if step_size is None:
                # Line search failed, use steepest descent on g
                self.logger.warning(f"L-BFGS line search failed at step {step}, using steepest descent")
                line_search_failures += 1
                self._warn_if_line_search_degenerate(
                    line_search_failures, step + 1, epsilon=self.gradient_epsilon
                )
                step_size = lr / (step + 1)
                x_new = np.clip(x - step_size * gradient, 0, 1)  # descent along -grad_g
                g_new, new_gradient = eval_min(x_new)
            else:
                x_new = np.clip(x + step_size * direction, 0, 1)

            new_gradient = self.clip_gradient(new_gradient)

            # Update history (curvature condition y·s > 0 holds where g is convex,
            # i.e. near a maximum of fitness)
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
            g = g_new
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

        # Final point may be the best one found
        if -g > best_fitness:
            best_fitness = -g
            best_x = x.copy()

        return best_x, best_fitness, history

    def _lbfgs_direction(
        self,
        gradient: np.ndarray,
        s_history: List[np.ndarray],
        y_history: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute r = H·gradient via the L-BFGS two-loop recursion.

        Sign-agnostic: the caller forms the descent direction as -r. With an
        empty history it returns the (scaled) gradient, i.e. steepest descent.

        Args:
            gradient: Current minimization-space gradient
            s_history: History of position differences
            y_history: History of minimization-space gradient differences

        Returns:
            r = H·gradient (the caller negates this for the descent direction)
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

        return r  # H·gradient; caller negates for the descent direction

    def _line_search(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        g_x: float,
        grad_x: np.ndarray,
        eval_min: Callable,
        initial_step: float,
        c1: float,
        c2: float,
        max_iter: int = 20
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """
        Backtracking line search (weak Wolfe) on the minimization objective g.

        Returns:
            Tuple of (step_size, g_new, grad_new)
            step_size is None if line search failed
        """
        step_size = initial_step
        directional_deriv = np.dot(grad_x, direction)

        if directional_deriv >= 0:
            # Not a descent direction for g
            return None, g_x, grad_x

        for _ in range(max_iter):
            x_new = np.clip(x + step_size * direction, 0, 1)
            g_new, grad_new = eval_min(x_new)

            # Armijo condition (sufficient decrease for minimization)
            if g_new <= g_x + c1 * step_size * directional_deriv:
                # Weak-Wolfe curvature condition
                new_directional_deriv = np.dot(grad_new, direction)
                if new_directional_deriv >= c2 * directional_deriv:
                    return step_size, g_new, grad_new

            step_size *= 0.5

            if step_size < 1e-10:
                break

        return None, g_x, grad_x
