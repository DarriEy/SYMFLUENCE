"""
Model-Specific Optimizers

Optimizers that inherit from BaseModelOptimizer for each supported model.
These provide a unified interface while handling model-specific setup.

Available optimizers:
- SUMMAModelOptimizer: SUMMA hydrological model optimization
- FUSEModelOptimizer: FUSE model optimization
- NgenModelOptimizer: NextGen model optimization
- HYPEModelOptimizer: HYPE model optimization
- MizuRouteModelOptimizer: MizuRoute routing model optimization
- TRouteModelOptimizer: T-Route routing model optimization
"""

from .summa_optimizer import SUMMAModelOptimizer
from .fuse_model_optimizer import FUSEModelOptimizer
from .ngen_model_optimizer import NgenModelOptimizer
from .hype_model_optimizer import HYPEModelOptimizer
from .mizuroute_model_optimizer import MizuRouteModelOptimizer
from .troute_model_optimizer import TRouteModelOptimizer

__all__ = [
    'SUMMAModelOptimizer',
    'FUSEModelOptimizer',
    'NgenModelOptimizer',
    'HYPEModelOptimizer',
    'MizuRouteModelOptimizer',
    'TRouteModelOptimizer',
]
