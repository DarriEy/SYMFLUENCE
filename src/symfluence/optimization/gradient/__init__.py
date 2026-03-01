# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Gradient-based optimization utilities.

This subpackage provides model-agnostic utilities for gradient-based
optimization of hydrological models. These utilities are compatible with
JAX for automatic differentiation.

Components:
    - AdamW: Adam optimizer with decoupled weight decay
    - CosineAnnealingWarmRestarts: LR scheduler with warm restarts (SGDR)
    - CosineDecay: Simple cosine decay LR scheduler
    - EMA: Exponential moving average of parameters

Example:
    >>> from symfluence.optimization.gradient import AdamW, CosineAnnealingWarmRestarts, EMA
    >>>
    >>> optimizer = AdamW(lr=0.01, weight_decay=0.001)
    >>> scheduler = CosineAnnealingWarmRestarts(lr_max=0.1, T_0=50)
    >>> ema = EMA(decay=0.99)
    >>>
    >>> for i in range(n_iterations):
    ...     lr = scheduler.get_lr(i)
    ...     optimizer.lr = lr
    ...     params = optimizer.step(params, grads)
    ...     ema.update(params)
"""

from .optimizers import AdamW
from .schedulers import CosineAnnealingWarmRestarts, CosineDecay
from .utils import EMA

__all__ = [
    'AdamW',
    'CosineAnnealingWarmRestarts',
    'CosineDecay',
    'EMA',
]
