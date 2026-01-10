"""
Timing mixin for SYMFLUENCE modules.

Provides timing and profiling utilities.
"""

import logging
import time
from contextlib import contextmanager
from typing import ContextManager


class TimingMixin:
    """
    Mixin providing timing and profiling utilities.

    Requires self.logger to be available.
    """

    @contextmanager
    def time_limit(self, task_name: str) -> ContextManager[None]:
        """
        Context manager to time a task and log the duration.
        """
        start_time = time.time()
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        logger.debug(f"Starting task: {task_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed task: {task_name} in {duration:.2f} seconds")
