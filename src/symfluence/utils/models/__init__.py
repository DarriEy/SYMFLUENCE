# src/symfluence/utils/models/__init__.py
"""Hydrological model utilities."""

from .registry import ModelRegistry

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
    from . import hype
except ImportError as e:
    logger.warning(f"Could not import hype: {e}")

try:
    from . import mesh
except ImportError as e:
    logger.warning(f"Could not import mesh: {e}")

try:
    from . import flash
except ImportError as e:
    logger.warning(f"Could not import flash: {e}")

try:
    from . import gr
except ImportError as e:
    logger.warning(f"Could not import gr: {e}")


__all__ = ["ModelRegistry"]
