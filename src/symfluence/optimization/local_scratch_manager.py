"""
Deprecated: LocalScratchManager has moved to mixins.parallel

This module is maintained for backward compatibility only.
Please update imports to use the new location:

    from symfluence.optimization.mixins.parallel import LocalScratchManager

This redirect will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing LocalScratchManager from symfluence.optimization.local_scratch_manager "
    "is deprecated. Use 'from symfluence.optimization.mixins.parallel import "
    "LocalScratchManager' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from symfluence.optimization.mixins.parallel.local_scratch_manager import LocalScratchManager

__all__ = ['LocalScratchManager']
