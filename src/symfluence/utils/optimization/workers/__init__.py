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
- SUMMAWorker: Worker for SUMMA model calibration
- FUSEWorker: Worker for FUSE model calibration
- NgenWorker: Worker for NextGen model calibration
- HYPEWorker: Worker for HYPE model calibration
"""

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from .summa_worker import SUMMAWorker
from .fuse_worker import FUSEWorker
from .ngen_worker import NgenWorker
from .hype_worker import HYPEWorker

__all__ = [
    'BaseWorker',
    'WorkerTask',
    'WorkerResult',
    'SUMMAWorker',
    'FUSEWorker',
    'NgenWorker',
    'HYPEWorker',
]
