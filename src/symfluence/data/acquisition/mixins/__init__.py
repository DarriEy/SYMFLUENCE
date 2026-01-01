"""
Data Acquisition Mixins.

Provides reusable mixin classes for data acquisition handlers:
- RetryMixin: Exponential backoff retry logic
- ChunkedDownloadMixin: Temporal chunking and parallel downloads
- SpatialSubsetMixin: Spatial subsetting operations
"""

from .retry import RetryMixin
from .chunked import ChunkedDownloadMixin
from .spatial import SpatialSubsetMixin

__all__ = [
    'RetryMixin',
    'ChunkedDownloadMixin',
    'SpatialSubsetMixin',
]
