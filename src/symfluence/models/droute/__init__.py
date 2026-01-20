"""
dRoute River Routing Model for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

.. note::
    **API Stability**: This module is not yet covered by semantic versioning guarantees.
    Breaking changes may occur in minor releases until this module reaches stable status.

    **Known Limitations**:
    - dRoute library must be separately installed with Python bindings
    - Gradient support requires dRoute compiled with AD enabled (CoDiPack or Enzyme)
    - Some routing methods may not be available in all dRoute builds
    - Performance optimization is ongoing

    To disable this experimental module at import time, set the environment variable:
    ``SYMFLUENCE_DISABLE_EXPERIMENTAL=1``

dRoute is a C++ river routing library with Python bindings that enables:
- Automatic differentiation for gradient-based calibration
- Multiple routing methods (Muskingum-Cunge, IRF, Lag, Diffusive Wave, KWT)
- Native Python API for fast in-memory routing (no subprocess overhead)
- Compatible network topology format with mizuRoute

Components:
    - DRoutePreProcessor: Prepares network topology and configuration
    - DRouteRunner: Executes routing simulations via Python API or subprocess
    - DRouteResultExtractor: Extracts routed streamflow results
    - DRouteWorker: Handles calibration with gradient support
    - DRouteNetworkAdapter: Converts mizuRoute topology to dRoute format

Usage:
    # Standard workflow
    from symfluence.models.droute import DRoutePreProcessor, DRouteRunner

    preprocessor = DRoutePreProcessor(config, logger)
    preprocessor.run_preprocessing()

    runner = DRouteRunner(config, logger)
    output_path = runner.run_droute()

    # Gradient-based calibration (requires AD-enabled dRoute)
    from symfluence.models.droute.calibration import DRouteWorker

    worker = DRouteWorker(config, logger)
    gradients = worker.compute_gradient(params)

References:
    dRoute Library: https://github.com/your-org/droute
    Muskingum-Cunge routing: Cunge, J.A. (1969). On the Subject of a Flood
    Propagation Method (Muskingum Method). Journal of Hydraulic Research.
"""

import os
import warnings

# Check if experimental modules are disabled
_DISABLE_EXPERIMENTAL = os.environ.get('SYMFLUENCE_DISABLE_EXPERIMENTAL', '').lower() in ('1', 'true', 'yes')

if _DISABLE_EXPERIMENTAL:
    raise ImportError(
        "dRoute module is disabled via SYMFLUENCE_DISABLE_EXPERIMENTAL environment variable. "
        "This experimental module is not yet stable. Remove the environment variable to enable."
    )

# Emit runtime warning about experimental status
warnings.warn(
    "dRoute is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider using mizuRoute instead.",
    category=UserWarning,
    stacklevel=2
)

# Import core components
from .config import DRouteConfigAdapter
from .preprocessor import DRoutePreProcessor
from .runner import DRouteRunner
from .extractor import DRouteResultExtractor
from .mixins import DRouteConfigMixin
from .network_adapter import DRouteNetworkAdapter

# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('DROUTE')(DRouteConfigAdapter)

# Register preprocessor with ModelRegistry
ModelRegistry.register_preprocessor('DROUTE')(DRoutePreProcessor)

# Register runner with ModelRegistry
ModelRegistry.register_runner('DROUTE', method_name='run_droute')(DRouteRunner)

# Register result extractor with ModelRegistry
ModelRegistry.register_result_extractor('DROUTE')(DRouteResultExtractor)

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

__all__ = [
    # Main components
    'DRoutePreProcessor',
    'DRouteRunner',
    'DRouteResultExtractor',
    'DRouteNetworkAdapter',

    # Configuration
    'DRouteConfigAdapter',
    'DRouteConfigMixin',
]
