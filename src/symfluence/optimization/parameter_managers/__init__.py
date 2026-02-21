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

# Import parameter managers from canonical locations to trigger registration decorators
# This avoids deprecation warnings while still enabling decorator-based registration
# Errors are caught to handle optional dependencies
def _register_parameter_managers():
    """Import all parameter managers from canonical locations to trigger registry decorators."""
    import importlib
    import logging

    logger = logging.getLogger(__name__)

    # Map of canonical module paths for parameter managers
    canonical_modules = [
        'symfluence.models.ngen.calibration.parameter_manager',
        'symfluence.models.summa.calibration.parameter_manager',
        'symfluence.models.fuse.calibration.parameter_manager',
        'symfluence.models.gr.calibration.parameter_manager',
        'symfluence.models.hbv.calibration.parameter_manager',
        'symfluence.models.hype.calibration.parameter_manager',
        'symfluence.models.mesh.calibration.parameter_manager',
        'symfluence.models.rhessys.calibration.parameter_manager',
        'symfluence.models.gnn.calibration.parameter_manager',  # ML parameter manager
        'symfluence.models.sacsma.calibration.parameter_manager',
        'symfluence.models.xinanjiang.calibration.parameter_manager',
        'symfluence.models.modflow.calibration.parameter_manager',
        'symfluence.models.pihm.calibration.parameter_manager',

        'symfluence.models.hechms.calibration.parameter_manager',
        'symfluence.models.topmodel.calibration.parameter_manager',
        'symfluence.models.gsflow.calibration.parameter_manager',
        'symfluence.models.watflood.calibration.parameter_manager',
        'symfluence.models.clmparflow.calibration.parameter_manager',
    ]

    for module_path in canonical_modules:
        try:
            logger.debug(f"Attempting to import parameter manager from {module_path}")
            importlib.import_module(module_path)
        except (ImportError, AttributeError) as e:
            # Silently skip models with missing dependencies
            # This is expected for optional models
            model_name = module_path.split('.')[2]  # Extract model name from path
            logger.debug(f"Failed to import {model_name} parameter manager: {e}")
            pass

# Trigger registration on import
_register_parameter_managers()

# Re-export from canonical locations (avoids deprecation warnings for internal use)
# Users who import directly from the stub modules will still see deprecation warnings
from symfluence.models.fuse.calibration.parameter_manager import FUSEParameterManager
from symfluence.models.gr.calibration.parameter_manager import GRParameterManager
from symfluence.models.hbv.calibration.parameter_manager import HBVParameterManager
from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.ngen.calibration.parameter_manager import NgenParameterManager
from symfluence.models.rhessys.calibration.parameter_manager import RHESSysParameterManager
from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
from symfluence.models.gnn.calibration.parameter_manager import MLParameterManager
from symfluence.models.sacsma.calibration.parameter_manager import SacSmaParameterManager
from symfluence.models.xinanjiang.calibration.parameter_manager import XinanjiangParameterManager
from symfluence.models.modflow.calibration.parameter_manager import CoupledGWParameterManager
from symfluence.models.pihm.calibration.parameter_manager import PIHMParameterManager

from symfluence.models.gsflow.calibration.parameter_manager import GSFLOWParameterManager
from symfluence.models.watflood.calibration.parameter_manager import WATFLOODParameterManager

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
    'SacSmaParameterManager',
    'XinanjiangParameterManager',
    'CoupledGWParameterManager',
    'PIHMParameterManager',

    'GSFLOWParameterManager',
    'WATFLOODParameterManager',
]
