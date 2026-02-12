"""
UnifiedModelExecutor - Combined execution framework for model runners.

This module provides a single mixin class that combines the capabilities of:
- ModelExecutor: subprocess execution, SLURM job management, retry logic
- SpatialOrchestrator: spatial mode handling, routing integration

Usage:
    class MyRunner(BaseModelRunner, UnifiedModelExecutor):
        def run_model(self):
            # Execution capabilities from ModelExecutor
            result = self.execute_subprocess(
                command=['./model.exe', '-c', 'config.txt'],
                log_file=self.get_log_path() / 'run.log'
            )

            # Spatial capabilities from SpatialOrchestrator
            if self.requires_routing():
                output = self.route_model_output(output)

This consolidates the common inheritance pattern:
    class OldRunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator, ...):

Into a cleaner form:
    class NewRunner(BaseModelRunner, UnifiedModelExecutor, ...):
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


class UnifiedModelExecutor(ModelExecutor, SpatialOrchestrator):
    """
    Combined mixin providing both execution and spatial/routing capabilities.

    This class inherits from both ModelExecutor and SpatialOrchestrator,
    providing a single mixin for model runners that need both capabilities.

    Inherited from ModelExecutor:
        - execute_subprocess: Run commands locally with logging
        - submit_slurm_job: Submit SLURM jobs
        - submit_slurm_array: Submit SLURM job arrays
        - monitor_slurm_job: Wait for SLURM job completion
        - create_slurm_script: Generate SLURM batch scripts
        - run_with_retry: Execute with automatic retry
        - execute_in_mode: Execute in LOCAL or SLURM mode

    Inherited from SpatialOrchestrator:
        - get_spatial_config: Build spatial configuration
        - requires_routing: Check if routing is needed
        - convert_to_routing_format: Transform output for routing
        - route_model_output: Execute routing model

    Note:
        Neither ModelExecutor nor SpatialOrchestrator define abstract methods,
        so this class also defines none. Both parent classes use ABC for type
        hinting purposes but don't require subclass implementations.
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
