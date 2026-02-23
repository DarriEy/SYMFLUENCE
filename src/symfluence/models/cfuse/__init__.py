"""cFUSE (differentiable FUSE) PyTorch/Enzyme AD hydrological model.

.. warning::
    **EXPERIMENTAL** — API may change without notice.

Supports multiple model structures, native gradient computation via
Enzyme AD (with PyTorch autograd fallback), and batch processing.
"""

import warnings
from typing import TYPE_CHECKING

# Emit experimental warning on import
warnings.warn(
    "cFUSE is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider the stable FUSE module instead.",
    category=UserWarning,
    stacklevel=2
)

# Lightweight availability probes (cheap — no heavy imports)
try:
    import cfuse  # noqa: F401
    HAS_CFUSE = True
    CFUSE_VERSION = getattr(cfuse, '__version__', 'unknown')
except ImportError:
    HAS_CFUSE = False
    CFUSE_VERSION = None

try:
    import cfuse_core  # noqa: F401
    HAS_CFUSE_CORE = True
except ImportError:
    HAS_CFUSE_CORE = False

try:
    import torch  # noqa: F401
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
except ImportError:
    HAS_TORCH = False
    TORCH_VERSION = None

try:
    import cfuse_core as _cc
    HAS_ENZYME = getattr(_cc, 'HAS_ENZYME', False)
except (ImportError, AttributeError):
    HAS_ENZYME = False


def check_cfuse_installation() -> dict:
    """Check cFUSE and dependency installation status."""
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
    """Get list of available cFUSE model structures."""
    return ['vic', 'topmodel', 'prms', 'sacramento', 'arno']


def get_model_config(structure: str) -> dict:
    """Get model configuration for a given structure.

    Args:
        structure: Model structure name (vic, topmodel, prms, sacramento, arno)
    """
    if not HAS_CFUSE:
        raise ImportError("cFUSE not installed. Cannot get model configuration.")

    from cfuse import ARNO_CONFIG, PRMS_CONFIG, SACRAMENTO_CONFIG, TOPMODEL_CONFIG, VIC_CONFIG

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


# Lazy import mapping — avoids importing PyTorch/cfuse internals at module level
_LAZY_IMPORTS = {
    'CFUSEConfig': ('.config', 'CFUSEConfig'),
    'CFUSEConfigAdapter': ('.config', 'CFUSEConfigAdapter'),
    'CFUSEPreProcessor': ('.preprocessor', 'CFUSEPreProcessor'),
    'CFUSERunner': ('.runner', 'CFUSERunner'),
    'CFUSEPostprocessor': ('.postprocessor', 'CFUSEPostprocessor'),
    'CFUSERoutedPostprocessor': ('.postprocessor', 'CFUSERoutedPostprocessor'),
    'CFUSEResultExtractor': ('.extractor', 'CFUSEResultExtractor'),
    'CFUSEWorker': ('.calibration', 'CFUSEWorker'),
    'CFUSEParameterManager': ('.calibration', 'CFUSEParameterManager'),
    'get_cfuse_calibration_bounds': ('.calibration', 'get_cfuse_calibration_bounds'),
}


def __getattr__(name: str):
    """Lazy import handler for cFUSE module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + [
        'HAS_CFUSE', 'HAS_CFUSE_CORE', 'HAS_TORCH', 'HAS_ENZYME',
        'CFUSE_VERSION', 'TORCH_VERSION',
        'check_cfuse_installation', 'get_available_model_structures',
        'get_model_config', 'register_with_model_registry',
    ]


def register_with_model_registry():
    """Register cFUSE components with the ModelRegistry."""
    from symfluence.models.registry import ModelRegistry

    from .config import CFUSEConfigAdapter
    from .extractor import CFUSEResultExtractor

    ModelRegistry.register_config_adapter('CFUSE')(CFUSEConfigAdapter)
    ModelRegistry.register_result_extractor('CFUSE')(CFUSEResultExtractor)


# Eagerly register when module is imported
register_with_model_registry()


if TYPE_CHECKING:
    from .calibration import CFUSEParameterManager, CFUSEWorker, get_cfuse_calibration_bounds
    from .config import CFUSEConfig, CFUSEConfigAdapter
    from .extractor import CFUSEResultExtractor
    from .postprocessor import CFUSEPostprocessor, CFUSERoutedPostprocessor
    from .preprocessor import CFUSEPreProcessor
    from .runner import CFUSERunner


__all__ = [
    'CFUSEConfig', 'CFUSEConfigAdapter',
    'CFUSEPreProcessor', 'CFUSERunner', 'CFUSEPostprocessor',
    'CFUSERoutedPostprocessor', 'CFUSEResultExtractor',
    'CFUSEWorker', 'CFUSEParameterManager', 'get_cfuse_calibration_bounds',
    'check_cfuse_installation', 'get_available_model_structures', 'get_model_config',
    'HAS_CFUSE', 'HAS_CFUSE_CORE', 'HAS_TORCH', 'HAS_ENZYME',
    'register_with_model_registry',
]
