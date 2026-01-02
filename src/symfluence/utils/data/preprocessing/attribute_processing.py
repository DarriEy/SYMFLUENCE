"""
Backward-compatible attribute processing module.

This module re-exports the attributeProcessor for backward compatibility.
New code should import directly from attribute_processor.
"""

from .attribute_processor import attributeProcessor

__all__ = ['attributeProcessor']

