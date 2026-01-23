"""
Sequential Execution Strategy

Executes tasks one at a time in the current process.
"""

import logging
from typing import List, Dict, Any, Callable

from .base import ExecutionStrategy


class SequentialExecutionStrategy(ExecutionStrategy):
    """
    Sequential execution strategy.

    Executes tasks one at a time in the current process.
    Used when parallel execution is disabled or for single tasks.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize sequential execution strategy.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Strategy identifier for logging and selection."""
        return "sequential"

    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks sequentially.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Ignored for sequential execution

        Returns:
            List of results from task execution
        """
        results = []

        for task in tasks:
            try:
                result = worker_func(task)
                results.append(result)
            except (ValueError, RuntimeError, IOError) as e:
                self.logger.error(f"Task failed in sequential execution: {e}")
                results.append({'error': str(e), 'task': task})

        return results
