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

# Import optimizers from canonical locations to trigger registration decorators
# This avoids deprecation warnings while still enabling decorator-based registration
# Errors are caught to handle optional dependencies
def _register_optimizers():
    """Import all model optimizers from canonical locations to trigger registry decorators."""
    import importlib
    import logging

    logger = logging.getLogger(__name__)

    # Map of model names to their canonical module paths
    canonical_modules = [
        'symfluence.models.ngen.calibration.optimizer',
        'symfluence.models.summa.calibration.optimizer',
        'symfluence.models.fuse.calibration.optimizer',
        'symfluence.models.gr.calibration.optimizer',
        'symfluence.models.hbv.calibration.optimizer',
        'symfluence.models.hype.calibration.optimizer',
        'symfluence.models.mesh.calibration.optimizer',
        'symfluence.models.gnn.calibration.optimizer',
        'symfluence.models.lstm.calibration.optimizer',
        'symfluence.models.rhessys.calibration.optimizer',
        'symfluence.models.mizuroute.calibration.optimizer',
        'symfluence.models.troute.calibration.optimizer',
        'symfluence.models.jfuse.calibration.optimizer',
        'symfluence.models.cfuse.calibration.optimizer',
        'symfluence.models.sacsma.calibration.optimizer',
        'symfluence.models.xinanjiang.calibration.optimizer',
        'symfluence.models.parflow.calibration.optimizer',
        'symfluence.models.swat.calibration.optimizer',
        'symfluence.models.mhm.calibration.optimizer',
        'symfluence.models.crhm.calibration.optimizer',
        'symfluence.models.modflow.calibration.optimizer',
        'symfluence.models.pihm.calibration.optimizer',
        'symfluence.models.hydrogeosphere.calibration.optimizer',
    ]

    for module_path in canonical_modules:
        try:
            importlib.import_module(module_path)
        except (ImportError, AttributeError) as e:
            # Silently skip models with missing dependencies
            # This is expected for optional models
            model_name = module_path.split('.')[2]  # Extract model name from path
            logger.debug(f"Could not import {model_name} optimizer: {e}")
            pass

# Trigger registration on import
_register_optimizers()

__all__: list[str] = []
