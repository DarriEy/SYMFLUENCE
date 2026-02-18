"""
HEC-HMS Hydrological Model for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

A native Python/JAX implementation of core HEC-HMS algorithms, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- DDS and evolutionary calibration integration

Algorithms:
    - Temperature-Index (ATI) Snow Model
    - SCS Curve Number Continuous Loss
    - Clark Unit Hydrograph Transform
    - Linear Reservoir Baseflow

Components:
    - HecHmsPreProcessor: Prepares forcing data (P, T, PET)
    - HecHmsRunner: Executes model simulations
    - HecHmsPostprocessor: Extracts streamflow results
    - HecHmsWorker: Handles calibration

References:
    US Army Corps of Engineers (2000). Hydrologic Modeling System HEC-HMS
    Technical Reference Manual. Hydrologic Engineering Center, Davis, CA.
"""

import warnings
from typing import TYPE_CHECKING

_warning_shown = False


def _show_experimental_warning():
    """Show the experimental warning once when HEC-HMS components are first accessed."""
    global _warning_shown
    if not _warning_shown:
        warnings.warn(
            "HEC-HMS is an EXPERIMENTAL module. The API may change without notice. "
            "For production use, consider using SUMMA or FUSE instead.",
            category=UserWarning,
            stacklevel=4
        )
        _warning_shown = True


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # Configuration
    'HECHMSConfig': ('.config', 'HECHMSConfig'),
    'HecHmsConfigAdapter': ('.config', 'HecHmsConfigAdapter'),

    # Main components
    'HecHmsPreProcessor': ('.preprocessor', 'HecHmsPreProcessor'),
    'HecHmsRunner': ('.runner', 'HecHmsRunner'),
    'HecHmsPostprocessor': ('.postprocessor', 'HecHmsPostprocessor'),
    'HecHmsResultExtractor': ('.extractor', 'HecHmsResultExtractor'),

    # Parameters
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'HecHmsParameters': ('.parameters', 'HecHmsParameters'),
    'HecHmsState': ('.parameters', 'HecHmsState'),
    'create_params_from_dict': ('.parameters', 'create_params_from_dict'),
    'create_initial_state': ('.parameters', 'create_initial_state'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'snow_step': ('.model', 'snow_step'),
    'loss_step': ('.model', 'loss_step'),
    'transform_step': ('.model', 'transform_step'),
    'baseflow_step': ('.model', 'baseflow_step'),
    'step': ('.model', 'step'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Calibration
    'HecHmsWorker': ('.calibration', 'HecHmsWorker'),
    'HecHmsParameterManager': ('.calibration', 'HecHmsParameterManager'),
    'get_hechms_calibration_bounds': ('.calibration', 'get_hechms_calibration_bounds'),
}


def __getattr__(name: str):
    """Lazy import handler for HEC-HMS module components."""
    if name in _LAZY_IMPORTS:
        _show_experimental_warning()
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return available attributes for tab completion."""
    return list(_LAZY_IMPORTS.keys()) + ['register_with_model_registry']


def register_with_model_registry():
    """Explicitly register HEC-HMS with the ModelRegistry.

    Call this function to register the HEC-HMS runner, preprocessor, config adapter,
    and result extractor with the central ModelRegistry.
    """
    _show_experimental_warning()

    from .config import HecHmsConfigAdapter
    from .extractor import HecHmsResultExtractor
    from .preprocessor import HecHmsPreProcessor  # Trigger @register_preprocessor decorator
    from .runner import HecHmsRunner  # Trigger @register_runner decorator
    from symfluence.models.registry import ModelRegistry

    ModelRegistry.register_config_adapter('HECHMS')(HecHmsConfigAdapter)
    ModelRegistry.register_result_extractor('HECHMS')(HecHmsResultExtractor)


# Eagerly register HEC-HMS components when module is imported
register_with_model_registry()


# Type hints for IDE support
if TYPE_CHECKING:
    from .config import HECHMSConfig, HecHmsConfigAdapter
    from .preprocessor import HecHmsPreProcessor
    from .runner import HecHmsRunner
    from .postprocessor import HecHmsPostprocessor
    from .extractor import HecHmsResultExtractor
    from .parameters import (
        PARAM_BOUNDS,
        DEFAULT_PARAMS,
        HecHmsParameters,
        HecHmsState,
        create_params_from_dict,
        create_initial_state,
    )
    from .model import (
        simulate,
        simulate_jax,
        simulate_numpy,
        snow_step,
        loss_step,
        transform_step,
        baseflow_step,
        step,
        HAS_JAX,
    )
    from .calibration import HecHmsWorker, HecHmsParameterManager, get_hechms_calibration_bounds


__all__ = [
    # Main components
    'HecHmsPreProcessor',
    'HecHmsRunner',
    'HecHmsPostprocessor',
    'HecHmsResultExtractor',

    # Configuration
    'HECHMSConfig',
    'HecHmsConfigAdapter',

    # Parameters
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'HecHmsParameters',
    'HecHmsState',
    'create_params_from_dict',
    'create_initial_state',

    # Core model
    'simulate',
    'simulate_jax',
    'simulate_numpy',
    'snow_step',
    'loss_step',
    'transform_step',
    'baseflow_step',
    'step',
    'HAS_JAX',

    # Calibration
    'HecHmsWorker',
    'HecHmsParameterManager',
    'get_hechms_calibration_bounds',

    # Registration helper
    'register_with_model_registry',
]
