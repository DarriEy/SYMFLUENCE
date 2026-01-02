"""
Model-Specific Optimizers

Optimizers that inherit from BaseModelOptimizer for each supported model.
These provide a unified interface while handling model-specific setup.

Available optimizers:
- SUMMAModelOptimizer: SUMMA hydrological model optimization
- FUSEModelOptimizer: FUSE model optimization
- NgenModelOptimizer: NextGen model optimization
"""

from .summa_optimizer import SUMMAModelOptimizer
from .fuse_model_optimizer import FUSEModelOptimizer
from .ngen_model_optimizer import NgenModelOptimizer

__all__ = [
    'SUMMAModelOptimizer',
    'FUSEModelOptimizer',
    'NgenModelOptimizer',
]
