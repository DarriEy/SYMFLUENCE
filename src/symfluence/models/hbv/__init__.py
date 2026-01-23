"""
HBV-96 Hydrological Model for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

A native JAX-based implementation of the HBV-96 hydrological model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- GPU acceleration when available
- Distributed modeling with graph-based Muskingum-Cunge routing

Components:
    - HBVPreProcessor: Prepares forcing data (P, T, PET)
    - HBVRunner: Executes model simulations
    - HBVPostprocessor: Extracts streamflow results
    - HBVWorker: Handles calibration with gradient support
    - DistributedHBV: Semi-distributed HBV with river network routing

Usage:
    # Standard workflow
    from symfluence.models.hbv import HBVPreProcessor, HBVRunner, HBVPostprocessor

    preprocessor = HBVPreProcessor(config, logger)
    preprocessor.run_preprocessing()

    runner = HBVRunner(config, logger)
    output_path = runner.run_hbv()

    # Distributed HBV with routing
    from symfluence.models.hbv import DistributedHBV, create_synthetic_network

    network = create_synthetic_network(n_nodes=5, topology='fishbone')
    model = DistributedHBV(network)
    outlet_flow, state = model.simulate(precip, temp, pet)

    # Gradient-based calibration
    grad_fn = model.get_gradient_function(precip, temp, pet, obs)
    gradients = grad_fn(params_array)

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

import warnings
from typing import TYPE_CHECKING

# Flag to track if the experimental warning has been shown
_warning_shown = False


def _show_experimental_warning():
    """Show the experimental warning once when HBV components are first accessed."""
    global _warning_shown
    if not _warning_shown:
        warnings.warn(
            "HBV is an EXPERIMENTAL module. The API may change without notice. "
            "For production use, consider using SUMMA or FUSE instead.",
            category=UserWarning,
            stacklevel=4
        )
        _warning_shown = True


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # Configuration
    'HBVConfig': ('.config', 'HBVConfig'),
    'HBVConfigAdapter': ('.config', 'HBVConfigAdapter'),

    # Main components
    'HBVPreProcessor': ('.preprocessor', 'HBVPreProcessor'),
    'HBVRunner': ('.runner', 'HBVRunner'),
    'HBVPostprocessor': ('.postprocessor', 'HBVPostprocessor'),
    'HBVRoutedPostprocessor': ('.postprocessor', 'HBVRoutedPostprocessor'),
    'HBVResultExtractor': ('.extractor', 'HBVResultExtractor'),

    # Parameters (from parameters module)
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'RATE_PARAMS': ('.parameters', 'RATE_PARAMS'),
    'DURATION_PARAMS': ('.parameters', 'DURATION_PARAMS'),
    'HBVParameters': ('.parameters', 'HBVParameters'),
    'create_params_from_dict': ('.parameters', 'create_params_from_dict'),
    'scale_params_for_timestep': ('.parameters', 'scale_params_for_timestep'),
    'get_routing_buffer_length': ('.parameters', 'get_routing_buffer_length'),

    # Loss functions (from losses module)
    'nse_loss': ('.losses', 'nse_loss'),
    'kge_loss': ('.losses', 'kge_loss'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'simulate_ensemble': ('.model', 'simulate_ensemble'),
    'HBVState': ('.model', 'HBVState'),
    'create_initial_state': ('.model', 'create_initial_state'),
    'step_jax': ('.model', 'step_jax'),
    'snow_routine_jax': ('.model', 'snow_routine_jax'),
    'soil_routine_jax': ('.model', 'soil_routine_jax'),
    'response_routine_jax': ('.model', 'response_routine_jax'),
    'routing_routine_jax': ('.model', 'routing_routine_jax'),
    'triangular_weights': ('.model', 'triangular_weights'),
    'jit_simulate': ('.model', 'jit_simulate'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Calibration
    'HBVWorker': ('.calibration', 'HBVWorker'),
    'HBVParameterManager': ('.calibration', 'HBVParameterManager'),
    'get_hbv_calibration_bounds': ('.calibration', 'get_hbv_calibration_bounds'),

    # Distributed HBV with routing
    'DistributedHBV': ('.distributed', 'DistributedHBV'),
    'DistributedHBVState': ('.distributed', 'DistributedHBVState'),
    'DistributedHBVParams': ('.distributed', 'DistributedHBVParams'),
    'calibrate_distributed_hbv': ('.distributed', 'calibrate_distributed_hbv'),
    'calibrate_distributed_hbv_adam': ('.distributed', 'calibrate_distributed_hbv_adam'),
    'load_distributed_hbv_from_config': ('.distributed', 'load_distributed_hbv_from_config'),
    'RiverNetwork': ('.network', 'RiverNetwork'),
    'NetworkBuilder': ('.network', 'NetworkBuilder'),
    'create_synthetic_network': ('.network', 'create_synthetic_network'),
    'RoutingParams': ('.routing', 'RoutingParams'),
    'RoutingState': ('.routing', 'RoutingState'),
    'compute_muskingum_params': ('.routing', 'compute_muskingum_params'),
    'route_reach_step': ('.routing', 'route_reach_step'),
    'runoff_mm_to_cms': ('.routing', 'runoff_mm_to_cms'),

    # Regionalization
    'forward_transfer_function': ('.regionalization', 'forward_transfer_function'),
    'initialize_weights': ('.regionalization', 'initialize_weights'),
    'TransferFunctionConfig': ('.regionalization', 'TransferFunctionConfig'),
    'TransferLayer': ('.regionalization', 'TransferLayer'),

    # Optimizers
    'AdamW': ('.optimizers', 'AdamW'),
    'CosineAnnealingWarmRestarts': ('.optimizers', 'CosineAnnealingWarmRestarts'),
    'CosineDecay': ('.optimizers', 'CosineDecay'),
    'EMA': ('.optimizers', 'EMA'),
    'CalibrationResult': ('.optimizers', 'CalibrationResult'),
    'EXTENDED_PARAM_BOUNDS': ('.optimizers', 'EXTENDED_PARAM_BOUNDS'),
}


def __getattr__(name: str):
    """Lazy import handler for HBV module components.

    This allows importing from the hbv module without loading all submodules
    until they are actually accessed.
    """
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
    """Explicitly register HBV with the ModelRegistry.

    Call this function to register the HBV config adapter and result extractor
    with the central ModelRegistry. This is automatically done when HBV
    components are imported, but can be called explicitly if needed.
    """
    _show_experimental_warning()

    from .config import HBVConfigAdapter
    from .extractor import HBVResultExtractor
    from symfluence.models.registry import ModelRegistry

    ModelRegistry.register_config_adapter('HBV')(HBVConfigAdapter)
    ModelRegistry.register_result_extractor('HBV')(HBVResultExtractor)


# Type hints for IDE support
if TYPE_CHECKING:
    from .config import HBVConfig, HBVConfigAdapter
    from .preprocessor import HBVPreProcessor
    from .runner import HBVRunner
    from .postprocessor import HBVPostprocessor, HBVRoutedPostprocessor
    from .extractor import HBVResultExtractor
    from .parameters import (
        PARAM_BOUNDS,
        DEFAULT_PARAMS,
        RATE_PARAMS,
        DURATION_PARAMS,
        HBVParameters,
        create_params_from_dict,
        scale_params_for_timestep,
        get_routing_buffer_length,
    )
    from .losses import (
        nse_loss,
        kge_loss,
        get_nse_gradient_fn,
        get_kge_gradient_fn,
    )
    from .model import (
        simulate,
        simulate_jax,
        simulate_numpy,
        simulate_ensemble,
        HBVState,
        create_initial_state,
        step_jax,
        snow_routine_jax,
        soil_routine_jax,
        response_routine_jax,
        routing_routine_jax,
        triangular_weights,
        jit_simulate,
        HAS_JAX,
    )
    from .calibration import HBVWorker, HBVParameterManager, get_hbv_calibration_bounds
    from .network import RiverNetwork, NetworkBuilder, create_synthetic_network
    from .routing import (
        RoutingParams,
        RoutingState,
        compute_muskingum_params,
        route_reach_step,
        runoff_mm_to_cms,
    )
    from .distributed import (
        DistributedHBV,
        DistributedHBVState,
        DistributedHBVParams,
        calibrate_distributed_hbv,
        calibrate_distributed_hbv_adam,
        load_distributed_hbv_from_config,
    )
    from .regionalization import (
        forward_transfer_function,
        initialize_weights,
        TransferFunctionConfig,
        TransferLayer,
    )
    from .optimizers import (
        AdamW,
        CosineAnnealingWarmRestarts,
        CosineDecay,
        EMA,
        CalibrationResult,
        EXTENDED_PARAM_BOUNDS,
    )


__all__ = [
    # Main components
    'HBVPreProcessor',
    'HBVRunner',
    'HBVPostprocessor',
    'HBVRoutedPostprocessor',
    'HBVResultExtractor',

    # Configuration
    'HBVConfig',
    'HBVConfigAdapter',

    # Parameters (from parameters module)
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'RATE_PARAMS',
    'DURATION_PARAMS',
    'HBVParameters',
    'create_params_from_dict',
    'scale_params_for_timestep',
    'get_routing_buffer_length',

    # Loss functions (from losses module)
    'nse_loss',
    'kge_loss',
    'get_nse_gradient_fn',
    'get_kge_gradient_fn',

    # Core model
    'simulate',
    'simulate_jax',
    'simulate_numpy',
    'simulate_ensemble',
    'HBVState',
    'create_initial_state',
    'step_jax',
    'snow_routine_jax',
    'soil_routine_jax',
    'response_routine_jax',
    'routing_routine_jax',
    'triangular_weights',
    'jit_simulate',
    'HAS_JAX',

    # Calibration
    'HBVWorker',
    'HBVParameterManager',
    'get_hbv_calibration_bounds',

    # Distributed HBV with routing
    'DistributedHBV',
    'DistributedHBVState',
    'DistributedHBVParams',
    'calibrate_distributed_hbv',
    'calibrate_distributed_hbv_adam',
    'load_distributed_hbv_from_config',
    'RiverNetwork',
    'NetworkBuilder',
    'create_synthetic_network',
    'RoutingParams',
    'RoutingState',
    'compute_muskingum_params',
    'route_reach_step',
    'runoff_mm_to_cms',

    # Regionalization
    'forward_transfer_function',
    'initialize_weights',
    'TransferFunctionConfig',
    'TransferLayer',

    # Optimizers
    'AdamW',
    'CosineAnnealingWarmRestarts',
    'CosineDecay',
    'EMA',
    'CalibrationResult',
    'EXTENDED_PARAM_BOUNDS',

    # Registration helper
    'register_with_model_registry',
]
