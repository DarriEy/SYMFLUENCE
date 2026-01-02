"""
Backward-compatible attribute processing module.

This module re-exports the refactored attributeProcessor for backward compatibility.
New code should import directly from attribute_processing_refactored.
"""

from .attribute_processing_refactored import attributeProcessor

__all__ = ['attributeProcessor']

