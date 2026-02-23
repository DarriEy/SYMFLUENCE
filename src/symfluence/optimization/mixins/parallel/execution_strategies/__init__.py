"""
Execution Strategies

Different strategies for parallel task execution.
"""

from .base import ExecutionStrategy
from .mpi import MPIExecutionStrategy
from .process_pool import ProcessPoolExecutionStrategy
from .sequential import SequentialExecutionStrategy

__all__ = [
    'ExecutionStrategy',
    'SequentialExecutionStrategy',
    'ProcessPoolExecutionStrategy',
    'MPIExecutionStrategy',
]
