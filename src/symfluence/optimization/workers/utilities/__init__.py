"""
Worker utilities module.

Provides shared functionality for optimization workers including:
- RoutingDecider: Unified routing decision logic (moved to symfluence.models.utilities)
- StreamflowMetrics: Shared metric calculation utilities
"""

# RoutingDecider has moved to symfluence.models.utilities
# Import from there for backward compatibility
from symfluence.models.utilities.routing_decider import RoutingDecider

from .streamflow_metrics import StreamflowMetrics

__all__ = ['RoutingDecider', 'StreamflowMetrics']
