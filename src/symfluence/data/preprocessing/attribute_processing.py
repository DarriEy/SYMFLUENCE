# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatible attribute processing module.

This module re-exports the refactored attribute processor for backward compatibility
with code that imports from `attribute_processing` instead of `attribute_processor`.

The original monolithic attributeProcessor class has been split into specialized
processors (ElevationProcessor, SoilProcessor, etc.) in the attribute_processors
subpackage, but this compatibility wrapper maintains the original interface.
"""

import warnings

warnings.warn(
    "The 'attribute_processing' module is deprecated and will be removed in a future version. "
    "Please use 'attribute_processor' or the specialized processors in "
    "'symfluence.data.preprocessing.attribute_processors' instead. "
    "See 'attribute_processing_refactored' documentation.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the backward-compatible wrapper
from .attribute_processor import attributeProcessor

__all__ = ['attributeProcessor']
