"""
UnifiedModelExecutor - Combined execution framework for model runners.

.. deprecated::
    Execution capabilities are now built into ``BaseModelRunner``.
    ``UnifiedModelExecutor`` is kept as a thin shim that re-exports
    ``SpatialOrchestrator`` so existing ``class MyRunner(BaseModelRunner,
    UnifiedModelExecutor)`` declarations continue to work.

    Migration path:
        - Replace ``UnifiedModelExecutor`` with ``SpatialOrchestrator`` in
          inheritance lists (or remove it entirely if routing is not needed).
"""

from .model_executor import (
    ModelExecutor,
    ExecutionMode,
    ExecutionResult,
    SlurmJobConfig
)
from .spatial_orchestrator import (
    SpatialOrchestrator,
    SpatialMode,
    SpatialConfig,
    RoutingModel,
    RoutingConfig
)


class UnifiedModelExecutor(SpatialOrchestrator):
    """
    Deprecated combined mixin — now equivalent to ``SpatialOrchestrator``.

    Execution methods (``execute_subprocess``, SLURM helpers, etc.) have been
    absorbed into ``BaseModelRunner``.  This class inherits only from
    ``SpatialOrchestrator`` to provide routing / spatial-mode capabilities.

    Existing runner declarations like
    ``class FUSERunner(BaseModelRunner, UnifiedModelExecutor)``
    continue to work unchanged.
    """
    pass


# Re-export types for convenience
__all__ = [
    'UnifiedModelExecutor',
    # From model_executor
    'ModelExecutor',
    'ExecutionMode',
    'ExecutionResult',
    'SlurmJobConfig',
    # From spatial_orchestrator
    'SpatialOrchestrator',
    'SpatialMode',
    'SpatialConfig',
    'RoutingModel',
    'RoutingConfig',
]
