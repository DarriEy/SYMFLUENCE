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
    trigger its ``@register_parameter_manager`` decorator.  Models with no
    calibration support are skipped silently; models whose parameter-manager
    module *exists but fails to import* are surfaced at WARNING (see
    ``discover_calibration_components``) instead of vanishing silently.
    """
    import logging

    from symfluence.optimization._autodiscover import discover_calibration_components

    discover_calibration_components('parameter_manager', logging.getLogger(__name__))


# Trigger registration on import
_register_parameter_managers()

# Re-export in-tree parameter managers from their canonical locations.
# (HBV / SAC-SMA / Xinanjiang come from the optional JAX plugins and are
# re-exported lazily via __getattr__ below.)
from symfluence.models.fuse.calibration.parameter_manager import FUSEParameterManager
from symfluence.models.gnn.calibration.parameter_manager import MLParameterManager
from symfluence.models.gr.calibration.parameter_manager import GRParameterManager
from symfluence.models.gsflow.calibration.parameter_manager import GSFLOWParameterManager
from symfluence.models.hype.calibration.parameter_manager import HYPEParameterManager
from symfluence.models.mesh.calibration.parameter_manager import MESHParameterManager
from symfluence.models.modflow.calibration.parameter_manager import CoupledGWParameterManager
from symfluence.models.ngen.calibration.parameter_manager import NgenParameterManager
from symfluence.models.pihm.calibration.parameter_manager import PIHMParameterManager
from symfluence.models.rhessys.calibration.parameter_manager import RHESSysParameterManager
from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
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

# HBV / SAC-SMA / Xinanjiang parameter managers live in the optional JAX model
# plugins (the ``jax`` extra). Re-export them lazily (PEP 562) so importing this
# package never requires the plugins, while
# ``from ...parameter_managers import HBVParameterManager`` still works when they
# are installed. Accessing one without the plugin raises a clear ImportError.
_OPTIONAL_JAX_PARAM_MANAGERS = {
    'HBVParameterManager': 'jhbv.calibration.parameter_manager',
    'SacSmaParameterManager': 'jsacsma.calibration.parameter_manager',
    'XinanjiangParameterManager': 'jxaj.calibration.parameter_manager',
}


def __getattr__(name):
    """Lazily resolve the optional JAX-plugin parameter managers (PEP 562)."""
    module_path = _OPTIONAL_JAX_PARAM_MANAGERS.get(name)
    if module_path is not None:
        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"{name} is provided by the optional JAX model plugins. "
                'Install them with: pip install "symfluence[jax]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
