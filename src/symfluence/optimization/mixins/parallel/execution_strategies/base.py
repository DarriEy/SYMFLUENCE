"""
Base Execution Strategy

Abstract base class for parallel execution strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable


class ExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.

    Defines the interface for different parallel execution approaches
    (sequential, process pool, MPI, etc.).
    """

    @abstractmethod
    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tasks.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results from task execution
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for logging."""
        pass
