"""
cFUSE Model Integration for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

.. note::
    **API Stability**: This module is not yet covered by semantic versioning guarantees.
    Breaking changes may occur in minor releases until this module reaches stable status.

    **Known Limitations**:
    - Routing integration is not yet implemented for distributed mode
    - Some parameter combinations may produce numerical instabilities
    - Performance optimization is ongoing

    To disable this experimental module at import time, set the environment variable:
    ``SYMFLUENCE_DISABLE_EXPERIMENTAL=1``

This module provides integration for cFUSE (differentiable FUSE), a PyTorch/Enzyme AD
implementation of the FUSE hydrological model that supports gradient-based calibration.

cFUSE implements the Framework for Understanding Structural Errors (FUSE) model
with support for:
- Multiple model structures (PRMS, Sacramento, TOPMODEL, VIC, ARNO)
- Both lumped and distributed spatial modes
- Native gradient computation via Enzyme AD (with PyTorch autograd fallback)
- Efficient batch processing for multi-HRU simulations

Components:
    Preprocessor: Prepares forcing data (precip, temp, PET)
    Runner: Executes cFUSE simulations
    Postprocessor: Extracts streamflow results
    Extractor: Advanced result analysis utilities
    Worker: Calibration worker with native gradient support
    ParameterManager: Parameter bounds and transformations

Example Usage:

    Basic simulation:
    >>> from symfluence.models.cfuse import CFUSERunner
    >>> runner = CFUSERunner(config, logger)
    >>> output = runner.run_cfuse()

    Gradient-based calibration:
    >>> from symfluence.models.cfuse import CFUSEWorker
    >>> worker = CFUSEWorker(config, logger)
    >>> if worker.supports_native_gradients():
    ...     loss, grads = worker.evaluate_with_gradient(params, 'kge')

Requirements:
    - cfuse: Clone from https://github.com/DarriEy/cFUSE.git
    - PyTorch: pip install torch (for gradient computation)
    - Enzyme AD: Optional, for native gradients (falls back to numerical)
"""

import os
import warnings

# Check if experimental modules are disabled
_DISABLE_EXPERIMENTAL = os.environ.get('SYMFLUENCE_DISABLE_EXPERIMENTAL', '').lower() in ('1', 'true', 'yes')

if _DISABLE_EXPERIMENTAL:
    raise ImportError(
        "cFUSE module is disabled via SYMFLUENCE_DISABLE_EXPERIMENTAL environment variable. "
        "This experimental module is not yet stable. Remove the environment variable to enable."
    )

# Deferred warning - only shown when module is actually used
_EXPERIMENTAL_WARNING_SHOWN = False


def _warn_experimental():
    """Emit experimental warning on first actual use."""
    global _EXPERIMENTAL_WARNING_SHOWN
    if not _EXPERIMENTAL_WARNING_SHOWN:
        warnings.warn(
            "cFUSE is an EXPERIMENTAL module. The API may change without notice. "
            "For production use, consider the stable FUSE module instead.",
            category=UserWarning,
            stacklevel=3
        )
        _EXPERIMENTAL_WARNING_SHOWN = True

# Import components to trigger registration with registries
from .config import CFUSEConfig, CFUSEConfigAdapter
from .preprocessor import CFUSEPreprocessor
from .runner import CFUSERunner
from .postprocessor import CFUSEPostprocessor, CFUSERoutedPostprocessor
from .extractor import CFUSEResultExtractor

# Import calibration components
from .calibration import CFUSEWorker, CFUSEParameterManager, get_cfuse_calibration_bounds

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('CFUSE')(CFUSEConfigAdapter)

# Check for cFUSE availability
try:
    import cfuse
    HAS_CFUSE = True
    CFUSE_VERSION = getattr(cfuse, '__version__', 'unknown')
except ImportError:
    HAS_CFUSE = False
    CFUSE_VERSION = None

# Check for cFUSE core (C++ module) availability
try:
    import cfuse_core
    HAS_CFUSE_CORE = True
except ImportError:
    HAS_CFUSE_CORE = False

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
except ImportError:
    HAS_TORCH = False
    TORCH_VERSION = None

# Check for Enzyme AD availability
try:
    import cfuse_core
    HAS_ENZYME = getattr(cfuse_core, 'HAS_ENZYME', False)
except (ImportError, AttributeError):
    HAS_ENZYME = False


def check_cfuse_installation() -> dict:
    """
    Check cFUSE and dependency installation status.

    Returns:
        Dictionary with installation status and version info.
    """
    return {
        'cfuse_installed': HAS_CFUSE,
        'cfuse_version': CFUSE_VERSION,
        'cfuse_core_installed': HAS_CFUSE_CORE,
        'torch_installed': HAS_TORCH,
        'torch_version': TORCH_VERSION,
        'enzyme_available': HAS_ENZYME,
        'native_gradients_available': HAS_TORCH and HAS_CFUSE_CORE,
        'enzyme_gradients_available': HAS_ENZYME,
    }


def get_available_model_structures() -> list:
    """
    Get list of available cFUSE model structures.

    Returns:
        List of model structure names.
    """
    return ['vic', 'topmodel', 'prms', 'sacramento', 'arno']


def get_model_config(structure: str) -> dict:
    """
    Get model configuration for a given structure.

    Args:
        structure: Model structure name (vic, topmodel, prms, sacramento, arno)

    Returns:
        Dictionary with model configuration
    """
    if not HAS_CFUSE:
        raise ImportError("cFUSE not installed. Cannot get model configuration.")

    from cfuse import (
        VIC_CONFIG, TOPMODEL_CONFIG, PRMS_CONFIG,
        SACRAMENTO_CONFIG, ARNO_CONFIG
    )

    configs = {
        'vic': VIC_CONFIG,
        'topmodel': TOPMODEL_CONFIG,
        'prms': PRMS_CONFIG,
        'sacramento': SACRAMENTO_CONFIG,
        'arno': ARNO_CONFIG,
    }

    structure_lower = structure.lower()
    if structure_lower not in configs:
        raise ValueError(f"Unknown model structure: {structure}. "
                        f"Available: {list(configs.keys())}")

    return configs[structure_lower].to_dict()


__all__ = [
    # Configuration
    'CFUSEConfig',
    'CFUSEConfigAdapter',
    # Model components
    'CFUSEPreprocessor',
    'CFUSERunner',
    'CFUSEPostprocessor',
    'CFUSERoutedPostprocessor',
    'CFUSEResultExtractor',
    # Calibration components
    'CFUSEWorker',
    'CFUSEParameterManager',
    'get_cfuse_calibration_bounds',
    # Utilities
    'check_cfuse_installation',
    'get_available_model_structures',
    'get_model_config',
    # Availability flags
    'HAS_CFUSE',
    'HAS_CFUSE_CORE',
    'HAS_TORCH',
    'HAS_ENZYME',
]
