# src/symfluence.models/__init__.py
"""Hydrological model utilities.

This module provides:
- ModelRegistry: Central registry for model runners/preprocessors/postprocessors
- Execution Framework: Unified subprocess/SLURM execution (execution submodule)
- Config Schemas: Type-safe configuration contracts (config submodule)
- Templates: Base classes for new model implementations (templates submodule)
"""

from .registry import ModelRegistry

# Import execution framework components
try:
    from .execution import (
        ModelExecutor,
        ExecutionResult,
        SlurmJobConfig,
        SpatialOrchestrator,
        SpatialMode,
        RoutingConfig,
    )
except ImportError:
    pass  # Optional - may not be needed by all users

# Import config schema components
try:
    from .config import (
        ModelConfigSchema,
        get_model_schema,
        validate_model_config,
    )
except ImportError:
    pass  # Optional

# Import template components
try:
    from .templates import (
        UnifiedModelRunner,
        ModelRunResult,
    )
except ImportError:
    pass  # Optional

# Import all models to register them
import logging
logger = logging.getLogger(__name__)

# Import from modular packages (preferred)
try:
    from . import summa
except ImportError as e:
    logger.warning(f"Could not import summa: {e}")

try:
    from . import fuse
except ImportError as e:
    logger.warning(f"Could not import fuse: {e}")

try:
    from . import ngen
except ImportError as e:
    logger.warning(f"Could not import ngen: {e}")

try:
    from . import mizuroute
except ImportError as e:
    logger.warning(f"Could not import mizuroute: {e}")

try:
    from . import troute
except ImportError as e:
    logger.warning(f"Could not import troute: {e}")

try:
    from . import droute
except ImportError as e:
    logger.warning(f"Could not import droute: {e}")

try:
    from . import hype
except ImportError as e:
    logger.warning(f"Could not import hype: {e}")

try:
    from . import mesh
except ImportError as e:
    logger.warning(f"Could not import mesh: {e}")

try:
    from . import lstm
except ImportError as e:
    logger.warning(f"Could not import lstm: {e}")

try:
    from . import gr
except Exception as e:
    # Catch Exception broadly because GR depends on rpy2 which can raise
    # RuntimeError or RRuntimeError when R is installed but broken
    logger.warning(f"Could not import gr: {e}")

try:
    from . import gnn
except ImportError as e:
    logger.warning(f"Could not import gnn: {e}")

try:
    from . import rhessys
except ImportError as e:
    logger.warning(f"Could not import rhessys: {e}")

try:
    from . import hbv
except ImportError as e:
    logger.warning(f"Could not import hbv: {e}")

try:
    from . import jfuse
except ImportError as e:
    logger.warning(f"Could not import jfuse: {e}")

try:
    from . import cfuse
except ImportError as e:
    logger.warning(f"Could not import cfuse: {e}")


__all__ = [
    # Core
    "ModelRegistry",
    # Execution Framework
    "ModelExecutor",
    "ExecutionResult",
    "SlurmJobConfig",
    "SpatialOrchestrator",
    "SpatialMode",
    "RoutingConfig",
    # Config Schemas
    "ModelConfigSchema",
    "get_model_schema",
    "validate_model_config",
    # Templates
    "UnifiedModelRunner",
    "ModelRunResult",
]
