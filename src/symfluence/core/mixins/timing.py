"""
Timing mixin for SYMFLUENCE modules.

Provides timing and profiling utilities.
"""

import time
import logging
from typing import Iterator
from contextlib import contextmanager


class TimingMixin:
    """
    Mixin providing timing and profiling utilities.

    Requires self.logger to be available.
    """

    @contextmanager
    def time_limit(self, task_name: str) -> Iterator[None]:
        """
        Context manager to limit the execution time of a task.
        """
        start_time = time.time()
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        logger.debug(f"Starting task: {task_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed task: {task_name} in {duration:.2f} seconds")
