"""
dRoute River Routing Model for SYMFLUENCE.

.. warning::
    **EXPERIMENTAL MODULE** - This module is in active development and should be
    used at your own risk. The API may change without notice in future releases.
    Please report any issues at https://github.com/DarriEy/SYMFLUENCE/issues

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

import warnings

# Emit experimental warning on import
warnings.warn(
    "dRoute is an EXPERIMENTAL module. The API may change without notice. "
    "For production use, consider using mizuRoute instead.",
    category=UserWarning,
    stacklevel=2
)

# Import core components
# Register all dRoute components via unified registry
from symfluence.core.registry import model_manifest

from .config import DRouteConfigAdapter
from .extractor import DRouteResultExtractor
from .mixins import DRouteConfigMixin
from .network_adapter import DRouteNetworkAdapter
from .postprocessor import DRoutePostProcessor
from .preprocessor import DRoutePreProcessor
from .runner import DRouteRunner

model_manifest(
    "DROUTE",
    config_adapter=DRouteConfigAdapter,
    preprocessor=DRoutePreProcessor,
    runner=DRouteRunner,
    runner_method="run_droute",
    result_extractor=DRouteResultExtractor,
    build_instructions_module="symfluence.models.droute.build_instructions",
)

__all__ = [
    # Main components
    'DRoutePreProcessor',
    'DRouteRunner',
    'DRouteResultExtractor',
    'DRoutePostProcessor',
    'DRouteNetworkAdapter',

    # Configuration
    'DRouteConfigAdapter',
    'DRouteConfigMixin',
]

# Register calibration components with OptimizerRegistry
try:
    from .calibration import DRouteModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
