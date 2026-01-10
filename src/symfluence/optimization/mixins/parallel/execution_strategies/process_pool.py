"""
Process Pool Execution Strategy

Executes tasks using Python's ProcessPoolExecutor.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Callable

from .base import ExecutionStrategy


class ProcessPoolExecutionStrategy(ExecutionStrategy):
    """
    Process pool execution strategy.

    Uses ProcessPoolExecutor to run tasks in parallel while
    preserving result order.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize process pool execution strategy.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "process_pool"

    def execute(
        self,
        tasks: List[Dict[str, Any]],
        worker_func: Callable,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks using ProcessPoolExecutor.

        Args:
            tasks: List of task dictionaries
            worker_func: Function to call for each task
            max_workers: Maximum number of worker processes

        Returns:
            List of results in the same order as input tasks
        """
        if max_workers == 1 or len(tasks) == 1:
            # Fall back to sequential for single worker/task
            return [worker_func(task) for task in tasks]

        # Use ProcessPoolExecutor.map to preserve order
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker_func, tasks))

        return results
