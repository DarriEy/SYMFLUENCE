"""
Unified Model Execution Framework.

This module provides a standardized execution infrastructure for all hydrological models,
consolidating subprocess management, SLURM job handling, and spatial orchestration.

Components:
    - UnifiedModelExecutor: Combined mixin for execution + spatial capabilities (preferred)
    - ModelExecutor: Mixin for unified subprocess/SLURM execution
    - SpatialOrchestrator: Centralized spatial mode handling and routing integration
    - ExecutionResult: Dataclass for standardized execution results

Usage:
    # Preferred: Use UnifiedModelExecutor for both capabilities
    class MyRunner(BaseModelRunner, UnifiedModelExecutor):
        ...

    # Legacy: Individual mixins still available for backward compatibility
    class OldRunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator):
        ...
"""

from .model_executor import (
    ModelExecutor,
    ExecutionResult,
    SlurmJobConfig,
    ExecutionMode
)
from .spatial_orchestrator import (
    SpatialOrchestrator,
    SpatialMode,
    RoutingConfig,
    SpatialConfig,
    RoutingModel
)
from .unified_executor import UnifiedModelExecutor

__all__ = [
    # Preferred unified class
    'UnifiedModelExecutor',
    # Individual components (backward compatibility)
    'ModelExecutor',
    'ExecutionResult',
    'SlurmJobConfig',
    'ExecutionMode',
    'SpatialOrchestrator',
    'SpatialMode',
    'RoutingConfig',
    'SpatialConfig',
    'RoutingModel',
]
