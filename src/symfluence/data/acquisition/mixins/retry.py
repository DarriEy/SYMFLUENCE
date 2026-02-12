"""
Retry Mixin for Data Acquisition Handlers.

Provides exponential backoff retry logic for network operations.
"""

import time
import logging
from typing import Any, Callable, Tuple, Type, Optional


class RetryMixin:
    """
    Mixin providing retry logic with exponential backoff.

    Can be mixed into any class that needs retry capabilities.
    Expects the class to have a `logger` attribute.
    """

    def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        max_retries: int = 3,
        base_delay: float = 60.0,
        backoff_factor: float = 2.0,
        max_delay: float = 3600.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        retry_condition: Optional[Callable[[Exception], bool]] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to func
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Initial delay in seconds (default: 60)
            backoff_factor: Multiplier for delay on each retry (default: 2.0)
            max_delay: Maximum delay cap in seconds (default: 3600)
            retryable_exceptions: Tuple of exception types to retry on
            retry_condition: Optional function that takes an exception and returns
                           True if the error should be retried. If None, all
                           exceptions in retryable_exceptions are retried.
            on_retry: Optional callback(attempt, exception, delay) called before each retry
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            The last exception if all retries are exhausted

        Example:
            >>> def download_data():
            ...     # might raise network errors
            ...     pass
            >>> result = self.execute_with_retry(
            ...     download_data,
            ...     max_retries=3,
            ...     retry_condition=lambda e: "timeout" in str(e).lower()
            ... )
        """
        logger = getattr(self, 'logger', logging.getLogger(__name__))
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                # Check if this specific error should be retried
                should_retry = True
                if retry_condition is not None:
                    should_retry = retry_condition(e)

                if attempt < max_retries and should_retry:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                    # Call optional callback
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.0f}s..."
                        )

                    time.sleep(delay)
                else:
                    # Final attempt failed or error not retryable
                    if not should_retry:
                        logger.debug(f"Error not retryable: {e}")
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    def is_retryable_http_error(self, error: Exception) -> bool:
        """
        Check if an HTTP error is worth retrying.

        Common pattern for CDS, Earthdata, and similar APIs.
        Override in subclasses for custom logic.

        Args:
            error: The exception to check

        Returns:
            True if the error is likely transient and worth retrying
        """
        error_msg = str(error).lower()

        # Transient errors worth retrying
        transient_indicators = [
            "timeout",
            "timed out",
            "temporarily",
            "temporary",
            "503",
            "502",
            "500",
            "429",  # Rate limit
            "connection reset",
            "connection refused",
            "connection aborted",
            "broken pipe",
            "network unreachable",
            "maintenance",
        ]

        # Permanent errors not worth retrying
        permanent_indicators = [
            "401",  # Unauthorized
            "404",  # Not found
            "invalid credentials",
            "authentication failed",
            "request too large",
            "cost limits exceeded",
            "quota exceeded",
        ]

        # Check for permanent errors first
        for indicator in permanent_indicators:
            if indicator in error_msg:
                return False

        # Check for transient errors
        for indicator in transient_indicators:
            if indicator in error_msg:
                return True

        # 403 is special - could be temporary (rate limit) or permanent (auth)
        if "403" in error_msg or "forbidden" in error_msg:
            # Retry if it seems temporary
            return "temporarily" in error_msg or "rate" in error_msg

        # Default: don't retry unknown errors
        return False

    def is_retryable_cds_error(self, error: Exception) -> bool:
        """
        Check if a CDS API error is worth retrying.

        Specific logic for Copernicus Climate Data Store.

        Args:
            error: The exception to check

        Returns:
            True if the error should be retried
        """
        error_msg = str(error).lower()

        # Don't retry if request is too large
        if "too large" in error_msg or "cost limits exceeded" in error_msg:
            return False

        # Retry on 403 (often temporary rate limits) or maintenance
        if "403" in error_msg or "forbidden" in error_msg:
            return True

        if "temporarily" in error_msg or "maintenance" in error_msg:
            return True

        # Use general HTTP retry logic for other cases
        return self.is_retryable_http_error(error)


__all__ = ['RetryMixin']
