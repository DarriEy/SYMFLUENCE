"""
Delft-FEWS General Adapter for SYMFLUENCE.

Provides pre- and post-adapters that translate between FEWS's PI-XML/NetCDF-CF
data exchange protocol and SYMFLUENCE's native formats, enabling any registered
SYMFLUENCE model to be driven by a FEWS forecasting system.
"""

from .config import FEWSConfig, IDMapEntry
from .exceptions import (
    FEWSAdapterError,
    IDMappingError,
    PIXMLError,
    RunInfoParseError,
    StateExchangeError,
)
from .post_adapter import FEWSPostAdapter
from .pre_adapter import FEWSPreAdapter

__all__ = [
    "FEWSConfig",
    "IDMapEntry",
    "FEWSPreAdapter",
    "FEWSPostAdapter",
    "FEWSAdapterError",
    "IDMappingError",
    "PIXMLError",
    "RunInfoParseError",
    "StateExchangeError",
]
