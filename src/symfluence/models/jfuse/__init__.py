"""
jFUSE Model Integration for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

This module provides integration for jFUSE (JAX-based FUSE), a differentiable
hydrological model that supports gradient-based calibration via JAX autodiff.

jFUSE implements the Framework for Understanding Structural Errors (FUSE) model
with support for:
- Multiple model structures (PRMS, Sacramento, TOPMODEL, VIC)
- Both lumped and distributed spatial modes
- Native gradient computation via JAX autodiff
- Efficient JIT compilation for fast model execution

Components:
    Preprocessor: Prepares forcing data (precip, temp, PET)
    Runner: Executes jFUSE simulations
    Postprocessor: Extracts streamflow results
    Extractor: Advanced result analysis utilities
    Worker: Calibration worker with native gradient support
    ParameterManager: Parameter bounds and transformations

Example Usage:

    Basic simulation:
    >>> from symfluence.models.jfuse import JFUSERunner
    >>> runner = JFUSERunner(config, logger)
    >>> output = runner.run_jfuse()

    Gradient-based calibration:
    >>> from symfluence.models.jfuse import JFUSEWorker
    >>> worker = JFUSEWorker(config, logger)
    >>> if worker.supports_native_gradients():
    ...     loss, grads = worker.evaluate_with_gradient(params, 'kge')

Requirements:
    - jfuse: pip install jfuse (or from local development install)
    - JAX: pip install jax jaxlib (for gradient computation)
"""

import warnings

# Emit experimental warning on import
warnings.warn(
    "jFUSE is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider the stable FUSE module instead.",
    category=UserWarning,
    stacklevel=2
)

# Import components to trigger registration with registries
from .config import JFUSEConfig, JFUSEConfigAdapter
from .preprocessor import JFUSEPreProcessor
from .runner import JFUSERunner
from .postprocessor import JFUSEPostprocessor, JFUSERoutedPostprocessor
from .extractor import JFUSEResultExtractor

# Import calibration components
from .calibration import JFUSEWorker, JFUSEParameterManager, get_jfuse_calibration_bounds

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('JFUSE')(JFUSEConfigAdapter)

# Register result extractor with ModelRegistry
ModelRegistry.register_result_extractor('JFUSE')(JFUSEResultExtractor)

# Check for jFUSE availability
try:
    import jfuse
    HAS_JFUSE = True
    JFUSE_VERSION = getattr(jfuse, '__version__', 'unknown')
except ImportError:
    HAS_JFUSE = False
    JFUSE_VERSION = None

# Check for JAX availability
try:
    import jax
    HAS_JAX = True
    JAX_VERSION = jax.__version__
except ImportError:
    HAS_JAX = False
    JAX_VERSION = None


def check_jfuse_installation() -> dict:
    """
    Check jFUSE and JAX installation status.

    Returns:
        Dictionary with installation status and version info.
    """
    return {
        'jfuse_installed': HAS_JFUSE,
        'jfuse_version': JFUSE_VERSION,
        'jax_installed': HAS_JAX,
        'jax_version': JAX_VERSION,
        'native_gradients_available': HAS_JFUSE and HAS_JAX,
    }


__all__ = [
    # Configuration
    'JFUSEConfig',
    'JFUSEConfigAdapter',
    # Model components
    'JFUSEPreProcessor',
    'JFUSERunner',
    'JFUSEPostprocessor',
    'JFUSERoutedPostprocessor',
    'JFUSEResultExtractor',
    # Calibration components
    'JFUSEWorker',
    'JFUSEParameterManager',
    'get_jfuse_calibration_bounds',
    # Utilities
    'check_jfuse_installation',
    'HAS_JFUSE',
    'HAS_JAX',
]
