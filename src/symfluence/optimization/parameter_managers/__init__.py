# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-specific parameter managers (deprecated aggregate import surface).

Canonical homes are ``symfluence.models.<model>.calibration.parameter_manager``;
prefer the registry (``R.parameter_managers.get('<MODEL>')``). Names here are
resolved lazily (PEP 562) so importing this module does not import the models
layer at import time (optimization must not depend on models).

Importing this package still triggers parameter-manager auto-discovery so that
``R.parameter_managers`` is fully seeded, matching historical behaviour; the
discovery loop tolerates an absent models layer.
"""
from __future__ import annotations

import importlib
import logging


def _register_parameter_managers():
    """Auto-discover and import parameter managers from all model packages."""
    from symfluence.optimization._autodiscover import discover_calibration_components

    discover_calibration_components('parameter_manager', logging.getLogger(__name__))


# Trigger registration on import
_register_parameter_managers()

# In-tree managers, resolved lazily from their canonical model packages.
_MANAGERS = {
    'FUSEParameterManager': 'symfluence.models.fuse.calibration.parameter_manager',
    'MLParameterManager': 'symfluence.models.gnn.calibration.parameter_manager',
    'GRParameterManager': 'symfluence.models.gr.calibration.parameter_manager',
    'GSFLOWParameterManager': 'symfluence.models.gsflow.calibration.parameter_manager',
    'HYPEParameterManager': 'symfluence.models.hype.calibration.parameter_manager',
    'MESHParameterManager': 'symfluence.models.mesh.calibration.parameter_manager',
    'CoupledGWParameterManager': 'symfluence.models.modflow.calibration.parameter_manager',
    'NgenParameterManager': 'symfluence.models.ngen.calibration.parameter_manager',
    'PIHMParameterManager': 'symfluence.models.pihm.calibration.parameter_manager',
    'RHESSysParameterManager': 'symfluence.models.rhessys.calibration.parameter_manager',
    'SUMMAParameterManager': 'symfluence.models.summa.calibration.parameter_manager',
    'WATFLOODParameterManager': 'symfluence.models.watflood.calibration.parameter_manager',
}

# HBV / SAC-SMA / Xinanjiang parameter managers live in the optional JAX model
# plugins (the ``jax`` extra). Accessing one without the plugin installed
# raises a clear ImportError with the install hint.
_OPTIONAL_JAX_PARAM_MANAGERS = {
    'HBVParameterManager': 'jhbv.calibration.parameter_manager',
    'SacSmaParameterManager': 'jsacsma.calibration.parameter_manager',
    'XinanjiangParameterManager': 'jxaj.calibration.parameter_manager',
}

__all__ = list(_MANAGERS) + list(_OPTIONAL_JAX_PARAM_MANAGERS)


def __getattr__(name: str):
    module_path = _MANAGERS.get(name)
    if module_path is not None:
        value = getattr(importlib.import_module(module_path), name)
        globals()[name] = value
        return value
    module_path = _OPTIONAL_JAX_PARAM_MANAGERS.get(name)
    if module_path is not None:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"{name} is provided by the optional JAX model plugins. "
                'Install them with: pip install "symfluence[jax]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
