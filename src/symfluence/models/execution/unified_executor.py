"""Deprecated shim — UnifiedModelExecutor is now equivalent to SpatialOrchestrator."""

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
    """Deprecated shim — replace with ``SpatialOrchestrator``.

    Execution methods now live on ``BaseModelRunner`` via mixins.
    This class is equivalent to ``SpatialOrchestrator``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        import warnings
        warnings.warn(
            f"{cls.__name__} inherits from UnifiedModelExecutor which is deprecated. "
            "Replace with SpatialOrchestrator in the inheritance list.",
            DeprecationWarning,
            stacklevel=2,
        )

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
