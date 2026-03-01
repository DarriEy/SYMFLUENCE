# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Xinanjiang (XAJ) Rainfall-Runoff Model for SYMFLUENCE.

A clean-room JAX implementation of the saturation-excess model (Zhao, 1992),
the foundational model for Chinese operational hydrology.

15 calibration parameters providing a 3-layer evapotranspiration scheme,
saturation-excess runoff generation with parabolic storage capacity
distribution, and 3-source separation (surface/interflow/groundwater).

Components:
    - XinanjiangPreProcessor: Prepares forcing data (P, PET)
    - XinanjiangRunner: Executes simulations (JAX or NumPy backend)
    - XinanjiangPostprocessor: Extracts streamflow results
    - XinanjiangWorker: Handles calibration with optional gradient support

Usage:
    from symfluence.models.xinanjiang import XinanjiangPreProcessor, XinanjiangRunner

    preprocessor = XinanjiangPreProcessor(config, logger)
    preprocessor.run_preprocessing()

    runner = XinanjiangRunner(config, logger)
    runner.run_xinanjiang()

References:
    Zhao, R.-J. (1992). The Xinanjiang model applied in China.
    Journal of Hydrology, 135(1-4), 371-381.
"""

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS = {
    # Configuration
    'XinanjiangConfig': ('.config', 'XinanjiangConfig'),
    'XinanjiangConfigAdapter': ('.config', 'XinanjiangConfigAdapter'),

    # Main components
    'XinanjiangPreProcessor': ('.preprocessor', 'XinanjiangPreProcessor'),
    'XinanjiangRunner': ('.runner', 'XinanjiangRunner'),
    'XinanjiangPostprocessor': ('.postprocessor', 'XinanjiangPostprocessor'),
    'XinanjiangResultExtractor': ('.extractor', 'XinanjiangResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'PARAM_NAMES': ('.parameters', 'PARAM_NAMES'),
    'XinanjiangParams': ('.parameters', 'XinanjiangParams'),
    'XinanjiangState': ('.parameters', 'XinanjiangState'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Loss functions
    'kge_loss': ('.losses', 'kge_loss'),
    'nse_loss': ('.losses', 'nse_loss'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),

    # Calibration
    'XinanjiangWorker': ('.calibration', 'XinanjiangWorker'),
    'XinanjiangParameterManager': ('.calibration', 'XinanjiangParameterManager'),
    'XinanjiangModelOptimizer': ('.calibration', 'XinanjiangModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for Xinanjiang module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())


# Register all Xinanjiang components via unified registry
from symfluence.core.registry import model_manifest

from .config import XinanjiangConfigAdapter
from .extractor import XinanjiangResultExtractor

model_manifest(
    "XINANJIANG",
    config_adapter=XinanjiangConfigAdapter,
    result_extractor=XinanjiangResultExtractor,
)


if TYPE_CHECKING:
    from .calibration import XinanjiangModelOptimizer, XinanjiangParameterManager, XinanjiangWorker
    from .config import XinanjiangConfig, XinanjiangConfigAdapter
    from .extractor import XinanjiangResultExtractor
    from .losses import get_kge_gradient_fn, get_nse_gradient_fn, kge_loss, nse_loss
    from .model import HAS_JAX, simulate, simulate_jax, simulate_numpy
    from .parameters import (
        DEFAULT_PARAMS,
        PARAM_BOUNDS,
        PARAM_NAMES,
        XinanjiangParams,
        XinanjiangState,
    )
    from .postprocessor import XinanjiangPostprocessor
    from .preprocessor import XinanjiangPreProcessor
    from .runner import XinanjiangRunner


__all__ = [
    'XinanjiangConfig', 'XinanjiangConfigAdapter',
    'XinanjiangPreProcessor', 'XinanjiangRunner', 'XinanjiangPostprocessor', 'XinanjiangResultExtractor',
    'PARAM_BOUNDS', 'DEFAULT_PARAMS', 'PARAM_NAMES', 'XinanjiangParams', 'XinanjiangState',
    'simulate', 'simulate_jax', 'simulate_numpy', 'HAS_JAX',
    'kge_loss', 'nse_loss', 'get_kge_gradient_fn', 'get_nse_gradient_fn',
    'XinanjiangWorker', 'XinanjiangParameterManager', 'XinanjiangModelOptimizer',
]
