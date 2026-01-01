"""
SUMMA model utilities - Backward compatibility wrapper.

This file maintains backward compatibility with existing code that imports from summa_utils.py.
All SUMMA functionality has been refactored into the `summa/` module for better organization.

DEPRECATED: This file will be removed in a future version.
Please update imports to use:
    from symfluence.utils.models.summa import SummaPreProcessor, SummaRunner, SUMMAPostprocessor

Or for individual components:
    from symfluence.utils.models.summa.preprocessor import SummaPreProcessor
    from symfluence.utils.models.summa.runner import SummaRunner
    from symfluence.utils.models.summa.postprocessor import SUMMAPostprocessor
    from symfluence.utils.models.summa.forcing_processor import SummaForcingProcessor
    from symfluence.utils.models.summa.config_manager import SummaConfigManager
    from symfluence.utils.models.summa.attributes_manager import SummaAttributesManager

Refactored: 2025-12-31
Original file: 2,524 lines → Split into 6 modules (summa/ package)
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "summa_utils.py is deprecated. Please import from symfluence.utils.models.summa instead. "
    "Example: from symfluence.utils.models.summa import SummaPreProcessor",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from the new modular structure for backward compatibility
from .summa import (
    SummaPreProcessor,
    SummaRunner,
    SUMMAPostprocessor,
    SummaForcingProcessor,
    SummaConfigManager,
    SummaAttributesManager,
)

__all__ = [
    'SummaPreProcessor',
    'SummaRunner',
    'SUMMAPostprocessor',
    'SummaForcingProcessor',
    'SummaConfigManager',
    'SummaAttributesManager',
]
