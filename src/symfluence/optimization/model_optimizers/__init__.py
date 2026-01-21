"""
Model-Specific Optimizers

Optimizers that inherit from BaseModelOptimizer for each supported model.
These provide a unified interface while handling model-specific setup.

Model-specific optimizers are available via:
1. Direct import: from symfluence.optimization.model_optimizers.{model}_model_optimizer import {Model}ModelOptimizer
2. Registry pattern: OptimizerRegistry.get_optimizer('{MODEL}')

Note: We import each optimizer to trigger @register_optimizer decorators.
Import errors are caught to handle missing dependencies gracefully.
"""

# Import optimizers to trigger registration decorators
# Errors are caught to handle optional dependencies
def _register_optimizers():
    """Import all model optimizers to trigger registry decorators."""
    import importlib
    import logging

    logger = logging.getLogger(__name__)

    models = [
        'ngen',
        'summa',
        'fuse',
        'gr',
        'hbv',
        'hype',
        'mesh',
        'gnn',
        'lstm',
        'rhessys',
        'mizuroute',
        'troute',
        'jfuse',
        'cfuse',
    ]

    for model in models:
        try:
            importlib.import_module(f'.{model}_model_optimizer', package='symfluence.optimization.model_optimizers')
        except Exception as e:
            # Silently skip models with missing dependencies
            # This is expected for optional models
            logger.debug(f"Could not import {model} optimizer: {e}")
            pass

# Trigger registration on import
_register_optimizers()

__all__ = []
