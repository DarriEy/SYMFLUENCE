"""
Execution Strategies

Different strategies for parallel task execution.
"""

from .base import ExecutionStrategy
from .sequential import SequentialExecutionStrategy
from .process_pool import ProcessPoolExecutionStrategy
from .mpi import MPIExecutionStrategy

__all__ = [
    'ExecutionStrategy',
    'SequentialExecutionStrategy',
    'ProcessPoolExecutionStrategy',
    'MPIExecutionStrategy',
]
