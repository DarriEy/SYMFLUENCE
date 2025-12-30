from .registry import EvaluationRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import streamflow, tws, snow, et
except ImportError:
    pass

__all__ = ["EvaluationRegistry"]
