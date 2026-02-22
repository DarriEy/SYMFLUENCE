"""
Model-Specific Optimizers

Optimizers that inherit from BaseModelOptimizer for each supported model.
These provide a unified interface while handling model-specific setup.

Model-specific optimizers are available via:
1. Direct import: from symfluence.models.{model}.calibration.optimizer import {Model}ModelOptimizer
2. Registry pattern: OptimizerRegistry.get_optimizer('{MODEL}')

Registration happens via ``@OptimizerRegistry.register_optimizer``
decorators.  This module auto-discovers all model packages at import time
so that every ``calibration/optimizer.py`` is imported and its decorator
fires.
"""


def _register_optimizers():
    """Auto-discover and import model optimizers from all model packages.

    Scans ``symfluence.models.*`` for sub-packages that contain a
    ``calibration.optimizer`` module and imports each one to trigger its
    ``@register_optimizer`` decorator.  Models whose dependencies are not
    installed are silently skipped.
    """
    import importlib
    import logging
    import pkgutil

    logger = logging.getLogger(__name__)

    try:
        import symfluence.models as models_pkg
    except ImportError:
        return

    for _importer, model_name, is_pkg in pkgutil.iter_modules(models_pkg.__path__):
        if not is_pkg:
            continue
        module_path = f'symfluence.models.{model_name}.calibration.optimizer'
        try:
            importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Expected for models without calibration support or missing deps
            logger.debug("Skipped optimizer for %s", model_name)


# Trigger registration on import
_register_optimizers()

__all__: list[str] = []
