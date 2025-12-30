from .objective_registry import ObjectiveRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import multivariate
except ImportError:
    pass

__all__ = ["ObjectiveRegistry"]
