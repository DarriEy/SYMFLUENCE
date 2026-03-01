# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TOPMODEL (Beven & Kirkby 1979) for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

A native Python/JAX implementation of TOPMODEL, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- DDS and evolutionary calibration integration

Algorithms:
    - Degree-day snow module
    - Exponential transmissivity baseflow (Beven & Kirkby 1979)
    - Saturation-excess overland flow with topographic index distribution
    - Linear reservoir channel routing

Components:
    - TopmodelPreProcessor: Prepares forcing data (P, T, PET)
    - TopmodelRunner: Executes model simulations
    - TopmodelPostprocessor: Extracts streamflow results
    - TopmodelWorker: Handles calibration

References:
    Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable
    contributing area model of basin hydrology. Hydrological Sciences
    Bulletin, 24(1), 43-69.
"""

import warnings
from typing import TYPE_CHECKING

_warning_shown = False


def _show_experimental_warning():
    """Show the experimental warning once when TOPMODEL components are first accessed."""
    global _warning_shown
    if not _warning_shown:
        warnings.warn(
            "TOPMODEL is an EXPERIMENTAL module. The API may change without notice. "
            "For production use, consider using SUMMA or FUSE instead.",
            category=UserWarning,
            stacklevel=4
        )
        _warning_shown = True


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # Configuration
    'TOPMODELConfig': ('.config', 'TOPMODELConfig'),
    'TopmodelConfigAdapter': ('.config', 'TopmodelConfigAdapter'),

    # Main components
    'TopmodelPreProcessor': ('.preprocessor', 'TopmodelPreProcessor'),
    'TopmodelRunner': ('.runner', 'TopmodelRunner'),
    'TopmodelPostprocessor': ('.postprocessor', 'TopmodelPostprocessor'),
    'TopmodelResultExtractor': ('.extractor', 'TopmodelResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'TopmodelParameters': ('.parameters', 'TopmodelParameters'),
    'TopmodelState': ('.parameters', 'TopmodelState'),
    'create_params_from_dict': ('.parameters', 'create_params_from_dict'),
    'create_initial_state': ('.parameters', 'create_initial_state'),
    'generate_ti_distribution': ('.parameters', 'generate_ti_distribution'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'snow_step': ('.model', 'snow_step'),
    'topmodel_step': ('.model', 'topmodel_step'),
    'route_step': ('.model', 'route_step'),
    'step': ('.model', 'step'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Loss functions (for gradient-based calibration)
    'nse_loss': ('.losses', 'nse_loss'),
    'kge_loss': ('.losses', 'kge_loss'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),

    # Calibration
    'TopmodelWorker': ('.calibration', 'TopmodelWorker'),
    'TopmodelParameterManager': ('.calibration', 'TopmodelParameterManager'),
    'get_topmodel_calibration_bounds': ('.calibration', 'get_topmodel_calibration_bounds'),
}


def __getattr__(name: str):
    """Lazy import handler for TOPMODEL module components."""
    if name in _LAZY_IMPORTS:
        _show_experimental_warning()
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return available attributes for tab completion."""
    return list(_LAZY_IMPORTS.keys())


# Register all TOPMODEL components via unified registry
_show_experimental_warning()

from symfluence.core.registry import model_manifest

from .config import TopmodelConfigAdapter
from .extractor import TopmodelResultExtractor

model_manifest(
    "TOPMODEL",
    config_adapter=TopmodelConfigAdapter,
    result_extractor=TopmodelResultExtractor,
)


# Type hints for IDE support
if TYPE_CHECKING:
    from .calibration import TopmodelParameterManager, TopmodelWorker, get_topmodel_calibration_bounds
    from .config import TOPMODELConfig, TopmodelConfigAdapter
    from .extractor import TopmodelResultExtractor
    from .losses import (
        get_kge_gradient_fn,
        get_nse_gradient_fn,
        kge_loss,
        nse_loss,
    )
    from .model import (
        HAS_JAX,
        route_step,
        simulate,
        simulate_jax,
        simulate_numpy,
        snow_step,
        step,
        topmodel_step,
    )
    from .parameters import (
        DEFAULT_PARAMS,
        PARAM_BOUNDS,
        TopmodelParameters,
        TopmodelState,
        create_initial_state,
        create_params_from_dict,
        generate_ti_distribution,
    )
    from .postprocessor import TopmodelPostprocessor
    from .preprocessor import TopmodelPreProcessor
    from .runner import TopmodelRunner


__all__ = [
    # Main components
    'TopmodelPreProcessor',
    'TopmodelRunner',
    'TopmodelPostprocessor',
    'TopmodelResultExtractor',

    # Configuration
    'TOPMODELConfig',
    'TopmodelConfigAdapter',

    # Parameters
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'TopmodelParameters',
    'TopmodelState',
    'create_params_from_dict',
    'create_initial_state',
    'generate_ti_distribution',

    # Core model
    'simulate',
    'simulate_jax',
    'simulate_numpy',
    'snow_step',
    'topmodel_step',
    'route_step',
    'step',
    'HAS_JAX',

    # Loss functions
    'nse_loss',
    'kge_loss',
    'get_nse_gradient_fn',
    'get_kge_gradient_fn',

    # Calibration
    'TopmodelWorker',
    'TopmodelParameterManager',
    'get_topmodel_calibration_bounds',

]
