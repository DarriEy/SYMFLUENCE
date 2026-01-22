"""
Learning rate schedulers for gradient-based optimization.

Provides learning rate scheduling strategies that work with any optimizer.

Classes:
    CosineAnnealingWarmRestarts: Cosine annealing with warm restarts (SGDR)
    CosineDecay: Simple cosine decay schedule

Example:
    >>> from symfluence.optimization.gradient import CosineAnnealingWarmRestarts
    >>> scheduler = CosineAnnealingWarmRestarts(lr_max=0.1, T_0=50)
    >>> for i in range(500):
    ...     lr = scheduler.get_lr(i)
    ...     optimizer.lr = lr
"""

import numpy as np


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing learning rate schedule with warm restarts.

    Implements SGDR (Stochastic Gradient Descent with Warm Restarts)
    from Loshchilov & Hutter (2017).

    The learning rate follows a cosine curve from lr_max to lr_min,
    then "restarts" back to lr_max. Each restart period can optionally
    be longer than the previous (T_mult > 1).

    Args:
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate
        T_0: Initial restart period (iterations)
        T_mult: Period multiplier after each restart
        warmup_steps: Linear warmup steps before cosine schedule

    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(
        ...     lr_max=0.1, lr_min=1e-5, T_0=50, T_mult=2
        ... )
        >>> for i in range(500):
        ...     lr = scheduler.get_lr(i)
        ...     optimizer.lr = lr
    """

    def __init__(
        self,
        lr_max: float = 0.1,
        lr_min: float = 1e-5,
        T_0: int = 50,
        T_mult: int = 2,
        warmup_steps: int = 10
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """
        Get learning rate for given step.

        Args:
            step: Current iteration number

        Returns:
            Learning rate for this step
        """
        # Linear warmup
        if step < self.warmup_steps:
            return self.lr_max * (step + 1) / self.warmup_steps

        # Cosine annealing within current cycle
        adjusted_step = step - self.warmup_steps

        # Find current cycle and position within it
        T_cum = 0
        T_i = self.T_0

        while T_cum + T_i <= adjusted_step:
            T_cum += T_i
            T_i = T_i * self.T_mult

        T_cur = adjusted_step - T_cum

        # Cosine annealing
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * T_cur / T_i)
        )

        return float(lr)


class CosineDecay:
    """
    Simple cosine decay learning rate schedule (no restarts).

    Args:
        lr_max: Initial/maximum learning rate
        lr_min: Final/minimum learning rate
        total_steps: Total number of steps for decay
        warmup_steps: Linear warmup steps before cosine decay

    Example:
        >>> scheduler = CosineDecay(lr_max=0.02, lr_min=0.001, total_steps=300)
        >>> for i in range(300):
        ...     lr = scheduler.get_lr(i)
    """

    def __init__(
        self,
        lr_max: float = 0.02,
        lr_min: float = 0.001,
        total_steps: int = 300,
        warmup_steps: int = 0
    ):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            return self.lr_max * (step + 1) / self.warmup_steps

        # Cosine decay
        adjusted_step = step - self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps

        progress = min(adjusted_step / decay_steps, 1.0)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * progress)
        )

        return float(lr)


__all__ = ['CosineAnnealingWarmRestarts', 'CosineDecay']
