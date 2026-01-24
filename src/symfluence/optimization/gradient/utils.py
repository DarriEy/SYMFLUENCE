"""
Utility classes for gradient-based optimization.

Provides helper utilities for parameter management during optimization.

Classes:
    EMA: Exponential moving average of parameters

Example:
    >>> from symfluence.optimization.gradient import EMA
    >>> ema = EMA(decay=0.99)
    >>> for _ in range(100):
    ...     params = optimizer.step(params, grads)
    ...     ema.update(params)
    >>> final_params = ema.get()
"""

from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore[misc]


class EMA:
    """
    Exponential Moving Average of parameters.

    Maintains a shadow copy of parameters that is a running average,
    which often provides better generalization than the final iterate.

    Args:
        decay: EMA decay rate (higher = more smoothing)

    Example:
        >>> ema = EMA(decay=0.99)
        >>> for _ in range(100):
        ...     params = optimizer.step(params, grads)
        ...     ema.update(params)
        >>> final_params = ema.get()  # Use smoothed parameters
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.shadow = None
        self.num_updates = 0

    def update(self, params: Any):
        """
        Update shadow parameters with current parameters.

        Args:
            params: Current parameters to incorporate
        """
        xp = jnp if HAS_JAX else np

        if self.shadow is None:
            self.shadow = params.copy() if hasattr(params, 'copy') else xp.array(params)
        else:
            self.shadow = self.decay * self.shadow + (1 - self.decay) * params

        self.num_updates += 1

    def get(self) -> Any:
        """
        Get shadow (EMA) parameters.

        Returns:
            EMA parameters, or None if never updated
        """
        return self.shadow

    def get_bias_corrected(self) -> Any:
        """
        Get bias-corrected EMA parameters.

        Returns:
            Bias-corrected EMA parameters
        """
        if self.shadow is None:
            return None

        # Bias correction for early iterations
        correction = 1 - self.decay ** self.num_updates
        return self.shadow / correction

    def reset(self):
        """Reset EMA state."""
        self.shadow = None
        self.num_updates = 0


__all__ = ['EMA']
