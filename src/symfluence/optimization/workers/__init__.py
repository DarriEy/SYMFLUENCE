"""
Optimization Workers

Worker classes that handle the evaluation of parameter sets during optimization.
Each worker is responsible for:
- Applying parameters to model configuration files
- Running model simulations
- Calculating objective metrics from outputs

Available workers:
- BaseWorker: Abstract base class for all workers
- WorkerTask: Data structure for worker inputs
- WorkerResult: Data structure for worker outputs

Model-specific workers are available via:
1. Direct import: from symfluence.optimization.workers.{model}_worker import {Model}Worker
2. Registry pattern: OptimizerRegistry.get_worker('{MODEL}')

Note: Model-specific workers are NOT eagerly imported here to avoid circular dependencies.
As models migrate to symfluence.models/{model}/calibration/, the old paths become facades
that re-export from the new location. Eager imports would create circular dependencies:
  optimization.workers.__init__ → workers.fuse_worker → models.fuse.calibration →
  optimization.optimizers.base_model_optimizer → optimization.workers.base_worker →
  optimization.workers.__init__ (CYCLE!)
"""

from .base_worker import BaseWorker, WorkerTask, WorkerResult

# Re-export for backward compatibility
# Import only when explicitly accessed to avoid circular dependencies
def __getattr__(name):
    """Lazy import of worker classes to avoid circular dependencies."""
    worker_mapping = {
        'FUSEWorker': '.fuse_worker',
        'GRWorker': '.gr_worker',
        'HBVWorker': '.hbv_worker',
        'HYPEWorker': '.hype_worker',
        'MESHWorker': '.mesh_worker',
        'NgenWorker': '.ngen_worker',
        'RHESSysWorker': '.rhessys_worker',
        'SUMMAWorker': '.summa_worker',
        'GNNWorker': '.gnn_worker',
        'LSTMWorker': '.lstm_worker',
    }

    if name in worker_mapping:
        from importlib import import_module
        module = import_module(worker_mapping[name], package='symfluence.optimization.workers')
        return getattr(module, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseWorker',
    'WorkerTask',
    'WorkerResult',
    'FUSEWorker',
    'GRWorker',
    'HBVWorker',
    'HYPEWorker',
    'MESHWorker',
    'NgenWorker',
    'RHESSysWorker',
    'SUMMAWorker',
    'GNNWorker',
    'LSTMWorker',
]
