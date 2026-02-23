"""
IGNACIO Fire Model Integration for SYMFLUENCE

This module provides SYMFLUENCE integration for the IGNACIO fire spread model,
which implements the Canadian Forest Fire Behavior Prediction (FBP) System
with Richards' elliptical wave propagation.

IGNACIO is an external Python package that must be installed separately:
    symfluence binary install ignacio

This module provides:
- IGNACIOConfig: Configuration model for fire simulation parameters
- IGNACIORunner: Model runner registered with ModelRegistry
- IGNACIOPreProcessor: Terrain and fuel data preparation
- IGNACIOPostProcessor: Result extraction and comparison with WMFire

The actual fire simulation logic is in the ignacio package:
- ignacio.simulation: Fire spread simulation
- ignacio.fbp: FBP fuel types and rate of spread
- ignacio.fwi: Fire Weather Index calculations
- ignacio.spread: Richards' elliptical propagation

References:
    IGNACIO: https://github.com/KatherineHopeReece/Fire-Engine-Framework
"""

import logging

logger = logging.getLogger(__name__)

# Import build instructions to register with BuildInstructionsRegistry
try:
    from . import build_instructions
except ImportError as e:
    logger.debug(f"Could not import build_instructions: {e}")

# Import SYMFLUENCE integration components
try:
    from .config import IGNACIOConfig
except ImportError as e:
    logger.debug(f"Could not import IGNACIOConfig: {e}")

try:
    from .runner import IGNACIORunner
except ImportError as e:
    logger.debug(f"Could not import IGNACIORunner: {e}")

try:
    from .preprocessor import IGNACIOPreProcessor
except ImportError as e:
    logger.debug(f"Could not import IGNACIOPreProcessor: {e}")

try:
    from .postprocessor import IGNACIOPostProcessor
except ImportError as e:
    logger.debug(f"Could not import IGNACIOPostProcessor: {e}")

try:
    from .extractor import IGNACIOResultExtractor
except ImportError as e:
    logger.debug(f"Could not import IGNACIOResultExtractor: {e}")

__all__ = [
    "IGNACIOConfig",
    "IGNACIORunner",
    "IGNACIOPreProcessor",
    "IGNACIOPostProcessor",
    "IGNACIOResultExtractor",
]

# Register result extractor with ModelRegistry
try:
    from symfluence.models.registry import ModelRegistry
    ModelRegistry.register_result_extractor('IGNACIO')(IGNACIOResultExtractor)
except Exception:  # noqa: BLE001 — optional dependency
    pass

# Register calibration components with OptimizerRegistry
try:
    from .calibration import IGNACIOModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
