"""
Unified Model Execution Framework.

This module provides a standardized execution infrastructure for all hydrological models,
consolidating subprocess management, SLURM job handling, and spatial orchestration.

Components:
    - ModelExecutor: Mixin for unified subprocess/SLURM execution
    - SpatialOrchestrator: Centralized spatial mode handling and routing integration
    - ExecutionResult: Dataclass for standardized execution results
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

__all__ = [
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
