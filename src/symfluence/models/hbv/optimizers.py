"""
Gradient-based optimizers for HBV calibration.

Provides JAX-compatible optimizers for calibrating distributed HBV models:
- AdamW: Adam with decoupled weight decay
- CosineAnnealingWarmRestarts: LR scheduler with warm restarts
- EMA: Exponential moving average of parameters

Example:
    >>> optimizer = AdamW(lr=0.01)
    >>> scheduler = CosineAnnealingWarmRestarts(lr_max=0.1, T_0=50)
    >>> ema = EMA(decay=0.99)
    >>>
    >>> for i in range(n_iterations):
    ...     lr = scheduler.get_lr(i)
    ...     optimizer.lr = lr
    ...     params = optimizer.step(params, grads)
    ...     ema.update(params)
"""

from typing import Dict, Any

# Re-export model-agnostic gradient utilities from optimization module
# for backward compatibility
from symfluence.optimization.gradient import (
    AdamW,
    CosineAnnealingWarmRestarts,
    CosineDecay,
    EMA,
)


class CalibrationResult:
    """
    Container for calibration results.

    Attributes:
        params: Calibrated parameter dictionary
        nse: Final NSE score
        kge: Final KGE score
        log_nse: Final log-NSE score
        rmse: Root mean square error
        volume_bias_pct: Volume bias as percentage
        history: Training history dictionary
        total_time: Total calibration time in seconds
        best_iter: Iteration with best NSE
        param_array: Parameters as numpy/JAX array
    """

    def __init__(
        self,
        params: Dict[str, float],
        nse: float,
        kge: float = None,
        log_nse: float = None,
        rmse: float = None,
        volume_bias_pct: float = None,
        history: Dict = None,
        total_time: float = None,
        best_iter: int = None,
        param_array: Any = None
    ):
        self.params = params
        self.nse = nse
        self.kge = kge
        self.log_nse = log_nse
        self.rmse = rmse
        self.volume_bias_pct = volume_bias_pct
        self.history = history or {}
        self.total_time = total_time
        self.best_iter = best_iter
        self.param_array = param_array

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(nse={self.nse:.4f}, "
            f"kge={self.kge:.4f if self.kge else 'N/A'}, "
            f"params={list(self.params.keys())})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'params': self.params,
            'nse': self.nse,
            'kge': self.kge,
            'log_nse': self.log_nse,
            'rmse': self.rmse,
            'volume_bias_pct': self.volume_bias_pct,
            'total_time': self.total_time,
            'best_iter': self.best_iter,
        }


# Extended parameter bounds for better exploration
EXTENDED_PARAM_BOUNDS = {
    'tt': (-3.0, 3.0),
    'cfmax': (0.5, 10.0),
    'sfcf': (0.5, 1.5),
    'cfr': (0.0, 0.1),
    'cwh': (0.0, 0.2),
    'fc': (50.0, 600.0),
    'lp': (0.3, 1.0),
    'beta': (1.0, 6.0),
    'perc': (0.0, 10.0),
    'k0': (0.005, 0.9),
    'k1': (0.0005, 0.3),
    'k2': (0.00005, 0.1),
    'uzl': (0.0, 100.0),
    'maxbas': (1.0, 7.0),
}


__all__ = [
    'AdamW',
    'CosineAnnealingWarmRestarts',
    'CosineDecay',
    'EMA',
    'CalibrationResult',
    'EXTENDED_PARAM_BOUNDS',
]
