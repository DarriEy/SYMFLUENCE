"""
FUSE model utilities - Backward compatibility wrapper.

This file maintains backward compatibility with existing code that imports from fuse_utils.py.
All FUSE functionality has been refactored into the `fuse/` module for better organization.

DEPRECATED: This file will be removed in a future version.
Please update imports to use:
    from symfluence.utils.models.fuse import FUSEPreProcessor, FUSERunner, etc.

Or for individual components:
    from symfluence.utils.models.fuse.preprocessor import FUSEPreProcessor
    from symfluence.utils.models.fuse.runner import FUSERunner
    from symfluence.utils.models.fuse.postprocessor import FUSEPostprocessor
    from symfluence.utils.models.fuse.decision_analyzer import FuseDecisionAnalyzer

Refactored: 2025-12-31
Original file: 3,094 lines → Split into 4 modules (fuse/ package)
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "fuse_utils.py is deprecated. Please import from symfluence.utils.models.fuse instead. "
    "Example: from symfluence.utils.models.fuse import FUSEPreProcessor",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from the new modular structure for backward compatibility
from .fuse import (
    FUSEPreProcessor,
    FUSERunner,
    FUSEPostprocessor,
    FuseDecisionAnalyzer,
)

__all__ = [
    'FUSEPreProcessor',
    'FUSERunner',
    'FUSEPostprocessor',
    'FuseDecisionAnalyzer',
]
