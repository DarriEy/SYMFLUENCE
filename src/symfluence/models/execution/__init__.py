# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model Execution Framework — types, spatial orchestration, and legacy shims.

Execution capabilities (``execute_subprocess``, SLURM helpers, ``run_with_retry``,
``execute_in_mode``) now live directly on ``BaseModelRunner``.  This package
exports the supporting data types and the ``SpatialOrchestrator`` mixin for
routing / spatial-mode handling.

Components:
    - SpatialOrchestrator: Spatial mode handling and routing integration
    - ExecutionResult / SlurmJobConfig / ExecutionMode: Execution data types
    - augment_conda_library_paths: Conda library path helper
    - ModelExecutor: **Deprecated** empty shim (kept for MRO compatibility)
    - UnifiedModelExecutor: **Deprecated** thin wrapper around SpatialOrchestrator
"""

from .model_executor import (
    ExecutionMode,
    ExecutionResult,
    ModelExecutor,
    SlurmJobConfig,
    augment_conda_library_paths,
)
from .spatial_orchestrator import RoutingConfig, RoutingModel, SpatialConfig, SpatialMode, SpatialOrchestrator
from .unified_executor import UnifiedModelExecutor

__all__ = [
    # Preferred unified class
    'UnifiedModelExecutor',
    # Individual components (backward compatibility)
    'ModelExecutor',
    'ExecutionResult',
    'SlurmJobConfig',
    'ExecutionMode',
    'augment_conda_library_paths',
    'SpatialOrchestrator',
    'SpatialMode',
    'RoutingConfig',
    'SpatialConfig',
    'RoutingModel',
]
