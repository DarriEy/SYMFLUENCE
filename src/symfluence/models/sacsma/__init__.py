"""
SAC-SMA + Snow-17 Hydrological Model for SYMFLUENCE.

A native NumPy/JAX implementation of the NWS operational standard:
- Snow-17: Anderson (1973, 2006) temperature-index snow model
- SAC-SMA: Burnash (1995) Sacramento Soil Moisture Accounting model

26 calibration parameters providing a dual-layer tension/free water
conceptual model, bridging complexity between GR4J (4-6 params) and
distributed models.

Dual-backend: all branching uses xp.where() for JAX differentiability.
Snow-17 definitions imported from the shared ``symfluence.models.snow17``
module.

Components:
    - SacSmaPreProcessor: Prepares forcing data (P, T, PET)
    - SacSmaRunner: Executes coupled Snow-17 + SAC-SMA simulations
    - SacSmaPostprocessor: Extracts streamflow results
    - SacSmaWorker: Handles calibration

Usage:
    from symfluence.models.sacsma import SacSmaPreProcessor, SacSmaRunner

    preprocessor = SacSmaPreProcessor(config, logger)
    preprocessor.run_preprocessing()

    runner = SacSmaRunner(config, logger)
    runner.run_sacsma()

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
"""

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS = {
    # Configuration
    'SacSmaConfig': ('.config', 'SacSmaConfig'),
    'SacSmaConfigAdapter': ('.config', 'SacSmaConfigAdapter'),

    # Main components
    'SacSmaPreProcessor': ('.preprocessor', 'SacSmaPreProcessor'),
    'SacSmaRunner': ('.runner', 'SacSmaRunner'),
    'SacSmaPostprocessor': ('.postprocessor', 'SacSmaPostprocessor'),
    'SacSmaResultExtractor': ('.extractor', 'SacSmaResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'SACSMA_PARAM_NAMES': ('.parameters', 'SACSMA_PARAM_NAMES'),
    'Snow17Parameters': ('.parameters', 'Snow17Parameters'),
    'SacSmaParameters': ('.parameters', 'SacSmaParameters'),
    'split_params': ('.parameters', 'split_params'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'jit_simulate': ('.model', 'jit_simulate'),
    'SacSmaSnow17State': ('.model', 'SacSmaSnow17State'),

    # Snow-17 (from shared module)
    'Snow17State': ('symfluence.models.snow17.parameters', 'Snow17State'),
    'Snow17Params': ('symfluence.models.snow17.parameters', 'Snow17Params'),
    'snow17_step': ('symfluence.models.snow17.model', 'snow17_step'),
    'snow17_simulate': ('symfluence.models.snow17.model', 'snow17_simulate'),

    # SAC-SMA core
    'SacSmaState': ('.sacsma', 'SacSmaState'),
    'sacsma_step': ('.sacsma', 'sacsma_step'),
    'sacsma_simulate': ('.sacsma', 'sacsma_simulate'),
    'sacsma_simulate_jax': ('.sacsma', 'sacsma_simulate_jax'),
    'sacsma_simulate_numpy': ('.sacsma', 'sacsma_simulate_numpy'),
    'HAS_JAX': ('.sacsma', 'HAS_JAX'),

    # Calibration
    'SacSmaWorker': ('.calibration', 'SacSmaWorker'),
    'SacSmaParameterManager': ('.calibration', 'SacSmaParameterManager'),
    'SacSmaModelOptimizer': ('.calibration', 'SacSmaModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for SAC-SMA module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        if module_path.startswith('.'):
            module = import_module(module_path, package=__name__)
        else:
            module = import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + ['register_with_model_registry']


def register_with_model_registry():
    """Register SAC-SMA with the ModelRegistry."""
    from .config import SacSmaConfigAdapter
    from .extractor import SacSmaResultExtractor
    from .preprocessor import SacSmaPreProcessor  # noqa: F401 - triggers decorator
    from .runner import SacSmaRunner  # noqa: F401 - triggers decorator
    from symfluence.models.registry import ModelRegistry

    ModelRegistry.register_config_adapter('SACSMA')(SacSmaConfigAdapter)
    ModelRegistry.register_result_extractor('SACSMA')(SacSmaResultExtractor)


# Eagerly register when module is imported
register_with_model_registry()


if TYPE_CHECKING:
    from .config import SacSmaConfig, SacSmaConfigAdapter
    from .preprocessor import SacSmaPreProcessor
    from .runner import SacSmaRunner
    from .postprocessor import SacSmaPostprocessor
    from .extractor import SacSmaResultExtractor
    from .parameters import (
        PARAM_BOUNDS, DEFAULT_PARAMS, SACSMA_PARAM_NAMES,
        Snow17Parameters, SacSmaParameters, split_params,
    )
    from .model import simulate, jit_simulate, SacSmaSnow17State
    from symfluence.models.snow17.parameters import Snow17State, Snow17Params
    from symfluence.models.snow17.model import snow17_step, snow17_simulate
    from .sacsma import (
        SacSmaState, sacsma_step, sacsma_simulate,
        sacsma_simulate_jax, sacsma_simulate_numpy, HAS_JAX,
    )
    from .calibration import SacSmaWorker, SacSmaParameterManager, SacSmaModelOptimizer


__all__ = [
    'SacSmaConfig', 'SacSmaConfigAdapter',
    'SacSmaPreProcessor', 'SacSmaRunner', 'SacSmaPostprocessor', 'SacSmaResultExtractor',
    'PARAM_BOUNDS', 'DEFAULT_PARAMS', 'SACSMA_PARAM_NAMES',
    'Snow17Parameters', 'SacSmaParameters', 'split_params',
    'simulate', 'jit_simulate', 'SacSmaSnow17State',
    'Snow17State', 'Snow17Params', 'snow17_step', 'snow17_simulate',
    'SacSmaState', 'sacsma_step', 'sacsma_simulate',
    'sacsma_simulate_jax', 'sacsma_simulate_numpy', 'HAS_JAX',
    'SacSmaWorker', 'SacSmaParameterManager', 'SacSmaModelOptimizer',
    'register_with_model_registry',
]
