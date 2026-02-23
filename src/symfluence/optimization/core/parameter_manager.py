"""
SUMMA Parameter Manager - Deprecated Location

.. deprecated::
    This module has been moved to symfluence.optimization.parameter_managers.summa_parameter_manager

    For new code, import from the new location:

    >>> from symfluence.optimization.parameter_managers import SUMMAParameterManager

    Or use the registry:

    >>> from symfluence.optimization.registry import OptimizerRegistry
    >>> ParameterManager = OptimizerRegistry.get_parameter_manager('SUMMA')

This file provides backward compatibility by re-exporting from the new location.
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing ParameterManager from symfluence.optimization.core.parameter_manager is deprecated. "
    "Use 'from symfluence.optimization.parameter_managers import SUMMAParameterManager' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
# Import directly from new location to avoid circular import
from symfluence.models.summa.calibration.parameter_manager import (
    SUMMAParameterManager,
)
from symfluence.models.summa.calibration.parameter_manager import (
    SUMMAParameterManager as ParameterManager,
)

__all__ = ['ParameterManager', 'SUMMAParameterManager']
