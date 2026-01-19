"""
HBV-96 Hydrological Model for SYMFLUENCE.

A native JAX-based implementation of the HBV-96 hydrological model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- GPU acceleration when available

Components:
    - HBVPreprocessor: Prepares forcing data (P, T, PET)
    - HBVRunner: Executes model simulations
    - HBVPostprocessor: Extracts streamflow results
    - HBVWorker: Handles calibration with gradient support

Usage:
    from symfluence.models.hbv import HBVPreprocessor, HBVRunner, HBVPostprocessor

    # Preprocessing
    preprocessor = HBVPreprocessor(config, logger)
    preprocessor.run_preprocessing()

    # Simulation
    runner = HBVRunner(config, logger)
    output_path = runner.run_hbv()

    # Post-processing
    postprocessor = HBVPostprocessor(config, logger, sim_dir=output_path)
    results_path = postprocessor.extract_streamflow()

Core Model Functions (for direct use):
    from symfluence.models.hbv.model import simulate, PARAM_BOUNDS, DEFAULT_PARAMS

    # Run simulation
    runoff, final_state = simulate(precip, temp, pet, params=my_params)

References:
    Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

# Register model components with ModelRegistry via imports
from .config import HBVConfig, HBVConfigAdapter
from .preprocessor import HBVPreprocessor
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

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('HBV')(HBVConfigAdapter)

__all__ = [
    # Main components
    'HBVPreprocessor',
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
]
