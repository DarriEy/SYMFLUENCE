from .registry import AcquisitionRegistry
from . import handlers

# The above 'from . import handlers' is sufficient to trigger registration
# of all handlers within the handlers directory.

__all__ = ["AcquisitionRegistry"]
