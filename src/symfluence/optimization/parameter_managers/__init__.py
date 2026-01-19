"""
Parameter Managers

Parameter manager classes that handle parameter transformations, bounds, and
file modifications for each supported model during optimization.

Each parameter manager is responsible for:
- Defining parameter bounds and transformations
- Applying parameter values to model configuration files
- Managing parameter-specific preprocessing

Model-specific parameter managers are available via:
1. Direct import: from symfluence.optimization.parameter_managers.{model}_parameter_manager import {Model}ParameterManager
2. Registry pattern: OptimizerRegistry.get_parameter_manager('{MODEL}')

Note: We import each parameter manager to trigger @register_parameter_manager decorators.
Import errors are caught to handle missing dependencies gracefully.
"""

# Import parameter managers to trigger registration decorators
# Errors are caught to handle optional dependencies
def _register_parameter_managers():
    """Import all parameter managers to trigger registry decorators."""
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
        'rhessys',
        'ml'
    ]

    for model in models:
        try:
            logger.debug(f"Attempting to import parameter manager for {model}")
            importlib.import_module(f'.{model}_parameter_manager', package='symfluence.optimization.parameter_managers')
        except Exception as e:
            # Silently skip models with missing dependencies
            # This is expected for optional models
            logger.debug(f"Failed to import {model} parameter manager: {e}")
            pass

# Trigger registration on import
_register_parameter_managers()

# Re-export for backward compatibility
from .fuse_parameter_manager import FUSEParameterManager
from .gr_parameter_manager import GRParameterManager
from .hbv_parameter_manager import HBVParameterManager
from .hype_parameter_manager import HYPEParameterManager
from .mesh_parameter_manager import MESHParameterManager
from .ngen_parameter_manager import NgenParameterManager
from .rhessys_parameter_manager import RHESSysParameterManager
from .summa_parameter_manager import SUMMAParameterManager
from .ml_parameter_manager import MLParameterManager

__all__ = [
    'FUSEParameterManager',
    'GRParameterManager',
    'HBVParameterManager',
    'HYPEParameterManager',
    'MESHParameterManager',
    'NgenParameterManager',
    'RHESSysParameterManager',
    'SUMMAParameterManager',
    'MLParameterManager',
]
