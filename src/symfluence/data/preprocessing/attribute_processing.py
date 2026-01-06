"""
Backward-compatible attribute processing module.

This module re-exports the refactored attribute processor for backward compatibility
with code that imports from `attribute_processing` instead of `attribute_processor`.

The original monolithic attributeProcessor class has been split into specialized
processors (ElevationProcessor, SoilProcessor, etc.) in the attribute_processors
subpackage, but this compatibility wrapper maintains the original interface.
"""

# Re-export the backward-compatible wrapper
from .attribute_processor import attributeProcessor

__all__ = ['attributeProcessor']
