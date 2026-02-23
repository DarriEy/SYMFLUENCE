"""
Data Acquisition Mixins.

Provides reusable mixin classes for data acquisition handlers:
- RetryMixin: Exponential backoff retry logic
- ChunkedDownloadMixin: Temporal chunking and parallel downloads
- SpatialSubsetMixin: Spatial subsetting operations
"""

from .chunked import ChunkedDownloadMixin
from .retry import RetryMixin
from .spatial import SpatialSubsetMixin

__all__ = [
    'RetryMixin',
    'ChunkedDownloadMixin',
    'SpatialSubsetMixin',
]
