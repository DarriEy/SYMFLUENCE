# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Gradient-based optimizers for hydrological model calibration.

Provides JAX-compatible optimizers that work with any model supporting
automatic differentiation.

Classes:
    AdamW: Adam optimizer with decoupled weight decay

Example:
    >>> from symfluence.optimization.gradient import AdamW
    >>> optimizer = AdamW(lr=0.01, weight_decay=0.001)
    >>> for _ in range(100):
    ...     loss, grads = val_grad_fn(params)
    ...     params = optimizer.step(params, grads)
"""

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore[misc]


class AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    Implements the AdamW algorithm from "Decoupled Weight Decay Regularization"
    (Loshchilov & Hutter, 2019).

    Args:
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        weight_decay: Decoupled weight decay coefficient

    Example:
        >>> optimizer = AdamW(lr=0.01, weight_decay=0.001)
        >>> for _ in range(100):
        ...     loss, grads = val_grad_fn(params)
        ...     params = optimizer.step(params, grads)
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def step(self, params: Any, grads: Any) -> Any:
        """
        Perform one AdamW update step.

        Args:
            params: Current parameters (numpy or JAX array)
            grads: Gradients with respect to parameters

        Returns:
            Updated parameters
        """
        xp = jnp if HAS_JAX else np
        self.t += 1

        if self.m is None:
            self.m = xp.zeros_like(params)
            self.v = xp.zeros_like(params)

        # Update biased first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Update biased second moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Weight decay (decoupled from gradient)
        if self.weight_decay > 0:
            params = params * (1 - self.lr * self.weight_decay)

        # Adam update
        params = params - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

        return params

    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0


__all__ = ['AdamW']
