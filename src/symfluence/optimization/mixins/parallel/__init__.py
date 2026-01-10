"""
Parallel Processing Module

Provides infrastructure for parallel execution of optimization tasks.
"""

from .directory_manager import DirectoryManager
from .config_updater import ConfigurationUpdater
from .task_distributor import TaskDistributor
from .worker_environment import WorkerEnvironmentConfig
from .execution_strategies import (
    ExecutionStrategy,
    SequentialExecutionStrategy,
    ProcessPoolExecutionStrategy,
    MPIExecutionStrategy,
)

__all__ = [
    'DirectoryManager',
    'ConfigurationUpdater',
    'TaskDistributor',
    'WorkerEnvironmentConfig',
    'ExecutionStrategy',
    'SequentialExecutionStrategy',
    'ProcessPoolExecutionStrategy',
    'MPIExecutionStrategy',
]
