"""jFUSE (JAX-based FUSE) differentiable hydrological model.

.. warning::
    **EXPERIMENTAL** — API may change without notice.

Supports multiple model structures, gradient-based calibration via JAX
autodiff, and both lumped and distributed spatial modes.
"""

import warnings
from typing import TYPE_CHECKING

# Emit experimental warning on import
warnings.warn(
    "jFUSE is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider the stable FUSE module instead.",
    category=UserWarning,
    stacklevel=2
)

# Lightweight availability probes (cheap — no heavy imports)
try:
    import jfuse  # noqa: F401
    HAS_JFUSE = True
    JFUSE_VERSION = getattr(jfuse, '__version__', 'unknown')
except ImportError:
    HAS_JFUSE = False
    JFUSE_VERSION = None

try:
    import jax  # noqa: F401
    HAS_JAX = True
    JAX_VERSION = jax.__version__
except ImportError:
    HAS_JAX = False
    JAX_VERSION = None


def check_jfuse_installation() -> dict:
    """Check jFUSE and JAX installation status."""
    return {
        'jfuse_installed': HAS_JFUSE,
        'jfuse_version': JFUSE_VERSION,
        'jax_installed': HAS_JAX,
        'jax_version': JAX_VERSION,
        'native_gradients_available': HAS_JFUSE and HAS_JAX,
    }


# Lazy import mapping — avoids importing JAX/jfuse internals at module level
_LAZY_IMPORTS = {
    'JFUSEConfig': ('.config', 'JFUSEConfig'),
    'JFUSEConfigAdapter': ('.config', 'JFUSEConfigAdapter'),
    'JFUSEPreProcessor': ('.preprocessor', 'JFUSEPreProcessor'),
    'JFUSERunner': ('.runner', 'JFUSERunner'),
    'JFUSEPostprocessor': ('.postprocessor', 'JFUSEPostprocessor'),
    'JFUSERoutedPostprocessor': ('.postprocessor', 'JFUSERoutedPostprocessor'),
    'JFUSEResultExtractor': ('.extractor', 'JFUSEResultExtractor'),
    'JFUSEWorker': ('.calibration', 'JFUSEWorker'),
    'JFUSEParameterManager': ('.calibration', 'JFUSEParameterManager'),
    'get_jfuse_calibration_bounds': ('.calibration', 'get_jfuse_calibration_bounds'),
}


def __getattr__(name: str):
    """Lazy import handler for jFUSE module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + [
        'HAS_JFUSE', 'HAS_JAX', 'JFUSE_VERSION', 'JAX_VERSION',
        'check_jfuse_installation', 'register_with_model_registry',
    ]


def register_with_model_registry():
    """Register jFUSE components with the ModelRegistry."""
    from symfluence.models.registry import ModelRegistry

    from .config import JFUSEConfigAdapter
    from .extractor import JFUSEResultExtractor

    ModelRegistry.register_config_adapter('JFUSE')(JFUSEConfigAdapter)
    ModelRegistry.register_result_extractor('JFUSE')(JFUSEResultExtractor)

    # Import component modules to trigger their @ModelRegistry.register_* decorators
    from . import (
        postprocessor,  # noqa: F401 — registers JFUSEPostprocessor
        preprocessor,  # noqa: F401 — registers JFUSEPreProcessor
        runner,  # noqa: F401 — registers JFUSERunner
    )


# Eagerly register when module is imported
register_with_model_registry()


if TYPE_CHECKING:
    from .calibration import JFUSEParameterManager, JFUSEWorker, get_jfuse_calibration_bounds
    from .config import JFUSEConfig, JFUSEConfigAdapter
    from .extractor import JFUSEResultExtractor
    from .postprocessor import JFUSEPostprocessor, JFUSERoutedPostprocessor
    from .preprocessor import JFUSEPreProcessor
    from .runner import JFUSERunner


__all__ = [
    'JFUSEConfig', 'JFUSEConfigAdapter',
    'JFUSEPreProcessor', 'JFUSERunner', 'JFUSEPostprocessor',
    'JFUSERoutedPostprocessor', 'JFUSEResultExtractor',
    'JFUSEWorker', 'JFUSEParameterManager', 'get_jfuse_calibration_bounds',
    'check_jfuse_installation',
    'HAS_JFUSE', 'HAS_JAX',
    'register_with_model_registry',
]
