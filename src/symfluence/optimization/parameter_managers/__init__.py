# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameter Managers

Parameter manager classes that handle parameter transformations, bounds, and
file modifications for each supported model during optimization.

Each parameter manager is responsible for:
- Defining parameter bounds and transformations
- Applying parameter values to model configuration files
- Managing parameter-specific preprocessing

Model-specific parameter managers are available via:
1. Direct import: from symfluence.models.{model}.calibration.parameter_manager import {Model}ParameterManager
2. Registry pattern: OptimizerRegistry.get_parameter_manager('{MODEL}')

Registration happens via ``@OptimizerRegistry.register_parameter_manager``
decorators.  This module auto-discovers all model packages at import time
so that every ``calibration/parameter_manager.py`` is imported and its
decorator fires.
"""


def _register_parameter_managers():
    """Auto-discover and import parameter managers from all model packages.

    Scans ``symfluence.models.*`` for sub-packages that contain a
    ``calibration.parameter_manager`` module and imports each one to
    trigger its ``@register_parameter_manager`` decorator.  Models whose
    dependencies are not installed are silently skipped.
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
        module_path = f'symfluence.models.{model_name}.calibration.parameter_manager'
        try:
            importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Expected for models without calibration support or missing deps
            logger.debug("Skipped parameter manager for %s", model_name)


# Trigger registration on import
_register_parameter_managers()

# Re-export from canonical locations (avoids deprecation warnings for internal use)
# Users who import directly from the stub modules will still see deprecation warnings
from symfluence.models.fuse.calibration.parameter_manager import FUSEParameterManager
from symfluence.models.gnn.calibration.parameter_manager import MLParameterManager
from symfluence.models.gr.calibration.parameter_manager import GRParameterManager
from symfluence.models.gsflow.calibration.parameter_manager import GSFLOWParameterManager
from symfluence.models.hbv.calibration.parameter_manager import HBVParameterManager
from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.modflow.calibration.parameter_manager import CoupledGWParameterManager
from symfluence.models.ngen.calibration.parameter_manager import NgenParameterManager
from symfluence.models.pihm.calibration.parameter_manager import PIHMParameterManager
from symfluence.models.rhessys.calibration.parameter_manager import RHESSysParameterManager
from symfluence.models.sacsma.calibration.parameter_manager import SacSmaParameterManager
from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
from symfluence.models.watflood.calibration.parameter_manager import WATFLOODParameterManager
from symfluence.models.xinanjiang.calibration.parameter_manager import XinanjiangParameterManager

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
