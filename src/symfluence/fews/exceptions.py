"""
FEWS adapter exceptions.

Provides specific exception types for the Delft-FEWS General Adapter integration.
All exceptions inherit from SYMFLUENCEError for consistent error handling.
"""

from symfluence.core.exceptions import SYMFLUENCEError


class FEWSAdapterError(SYMFLUENCEError):
    """Base exception for all FEWS adapter errors."""
    pass


class RunInfoParseError(FEWSAdapterError):
    """Raised when run_info.xml cannot be parsed or is malformed."""
    pass


class PIXMLError(FEWSAdapterError):
    """Raised when PI-XML timeseries read/write fails."""
    pass


class IDMappingError(FEWSAdapterError):
    """Raised when variable ID mapping fails (unknown variable, ambiguous match)."""
    pass


class StateExchangeError(FEWSAdapterError):
    """Raised when state file import/export fails."""
    pass
