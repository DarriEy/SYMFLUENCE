# src/symfluence/utils/models/__init__.py
"""Hydrological model utilities."""

from .registry import ModelRegistry

# Import all models to register them
try:
    from . import summa_utils
    from . import fuse_utils
    from . import gr_utils
    from . import hype_utils
    from . import flash_utils
    from . import mizuroute_utils
    from . import ngen_utils
    # from . import mesh_utils
except ImportError:
    # Handle cases where some dependencies might be missing
    pass

__all__ = ["ModelRegistry"]
