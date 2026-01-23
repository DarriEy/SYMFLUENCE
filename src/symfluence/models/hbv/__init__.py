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

# Emit experimental warning on import
warnings.warn(
    "HBV is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider using SUMMA or FUSE instead.",
    category=UserWarning,
    stacklevel=2
)

# Register model components with ModelRegistry via imports
from .config import HBVConfig, HBVConfigAdapter
from .preprocessor import HBVPreProcessor
from .runner import HBVRunner
from .postprocessor import HBVPostprocessor, HBVRoutedPostprocessor
from .extractor import HBVResultExtractor

# Core model functions (for direct use)
from .model import (
    simulate,
    simulate_ensemble,
    HBVParameters,
    HBVState,
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    create_params_from_dict,
    create_initial_state,
    nse_loss,
    kge_loss,
    HAS_JAX,
)

# Calibration support
from .calibration import HBVWorker, HBVParameterManager, get_hbv_calibration_bounds

# Distributed HBV with routing
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

# Optimizers for gradient-based calibration
from .optimizers import (
    AdamW,
    CosineAnnealingWarmRestarts,
    CosineDecay,
    EMA,
    CalibrationResult,
    EXTENDED_PARAM_BOUNDS,
)

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('HBV')(HBVConfigAdapter)

# Register result extractor with ModelRegistry
ModelRegistry.register_result_extractor('HBV')(HBVResultExtractor)

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

    # Core model
    'simulate',
    'simulate_ensemble',
    'HBVParameters',
    'HBVState',
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'create_params_from_dict',
    'create_initial_state',
    'nse_loss',
    'kge_loss',
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
]
