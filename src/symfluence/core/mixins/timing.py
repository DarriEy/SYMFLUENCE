"""
Timing mixin for SYMFLUENCE modules.

Provides timing and profiling utilities for measuring code execution time.
"""

import time
import logging
from typing import Iterator
from contextlib import contextmanager


class TimingMixin:
    """
    Mixin providing timing and profiling utilities.

    Provides
    --------
    - time_limit(task_name) : ContextManager
        Context manager that logs the execution time of a code block

    Optional Attributes
    -------------------
    self.logger : logging.Logger
        If available, timing messages are logged via this logger.
        Falls back to module-level logger if not present.

    Example
    -------
    >>> class MyProcessor(TimingMixin):
    ...     def process(self):
    ...         with self.time_limit("data loading"):
    ...             data = load_data()
    ...         with self.time_limit("computation"):
    ...             result = compute(data)
    ...         return result

    Notes
    -----
    This mixin is included in ConfigurableMixin, so you typically don't
    need to inherit it directly if using ConfigurableMixin.
    """

    @contextmanager
    def time_limit(self, task_name: str, log_level: int = logging.INFO) -> Iterator[None]:
        """
        Context manager to measure and log the execution time of a code block.

        Parameters
        ----------
        task_name : str
            Descriptive name for the task being timed
        log_level : int, optional
            Logging level for the completion message (default: INFO).
            Start message is always DEBUG.

        Yields
        ------
        None
            Control is yielded to the with-block

        Example
        -------
        >>> with self.time_limit("model execution"):
        ...     model.run()
        # Logs: "Starting task: model execution"
        # Logs: "Completed task: model execution in 12.34 seconds"
        """
        start_time = time.time()
        logger: logging.Logger = getattr(self, 'logger', logging.getLogger(__name__))
        logger.debug(f"Starting task: {task_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.log(log_level, f"Completed task: {task_name} in {duration:.2f} seconds")
