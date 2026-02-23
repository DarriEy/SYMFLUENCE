"""
Unit Tests for RetryMixin.

Tests the retry logic with exponential backoff:
- Successful execution (no retry)
- Retry on transient failure
- Max retries exhausted
- Exponential backoff calculation
- Max delay cap
- Retry condition filtering
- HTTP error classification
- CDS error classification
"""

import time
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def retry_mixin():
    """Create an instance of RetryMixin for testing."""
    from symfluence.data.acquisition.mixins.retry import RetryMixin

    class TestableRetryMixin(RetryMixin):
        def __init__(self):
            self.logger = MagicMock()

    return TestableRetryMixin()


# =============================================================================
# Execute With Retry Tests
# =============================================================================

@pytest.mark.mixin_retry
@pytest.mark.acquisition
class TestExecuteWithRetry:
    """Tests for execute_with_retry method."""

    def test_successful_execution_no_retry(self, retry_mixin):
        """Successful function should execute once without retry."""
        mock_func = Mock(return_value="success")

        result = retry_mixin.execute_with_retry(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_passes_args_to_function(self, retry_mixin):
        """Arguments should be passed to the function."""
        mock_func = Mock(return_value="success")

        retry_mixin.execute_with_retry(mock_func, "arg1", "arg2", kwarg1="value1")

        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_retry_on_failure(self, retry_mixin):
        """Should retry on failure until success."""
        # Fail twice, then succeed
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        with patch('time.sleep'):  # Skip actual delay
            result = retry_mixin.execute_with_retry(
                mock_func,
                max_retries=3,
                base_delay=1.0
            )

        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_retries_exhausted(self, retry_mixin):
        """Should raise after max retries exhausted."""
        mock_func = Mock(side_effect=Exception("persistent failure"))

        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=2,
                    base_delay=1.0
                )

        assert "persistent failure" in str(exc_info.value)
        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_exponential_backoff_calculation(self, retry_mixin):
        """Delay should increase exponentially."""
        mock_func = Mock(side_effect=[Exception("fail")] * 4 + ["success"])
        delays = []

        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = lambda d: delays.append(d)
            retry_mixin.execute_with_retry(
                mock_func,
                max_retries=4,
                base_delay=10.0,
                backoff_factor=2.0
            )

        # Expected delays: 10, 20, 40, 80
        assert len(delays) == 4
        assert delays[0] == 10.0  # 10 * 2^0
        assert delays[1] == 20.0  # 10 * 2^1
        assert delays[2] == 40.0  # 10 * 2^2
        assert delays[3] == 80.0  # 10 * 2^3

    def test_max_delay_cap(self, retry_mixin):
        """Delay should not exceed max_delay."""
        mock_func = Mock(side_effect=[Exception("fail")] * 5 + ["success"])
        delays = []

        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = lambda d: delays.append(d)
            retry_mixin.execute_with_retry(
                mock_func,
                max_retries=5,
                base_delay=100.0,
                backoff_factor=2.0,
                max_delay=200.0  # Cap at 200
            )

        # Delays should be capped at 200
        assert all(d <= 200.0 for d in delays)
        # Later delays should hit the cap
        assert delays[-1] == 200.0

    def test_retry_condition_filters_errors(self, retry_mixin):
        """retry_condition should filter which errors to retry."""
        # Only retry errors containing "transient"
        retry_condition = lambda e: "transient" in str(e)

        mock_func = Mock(side_effect=Exception("permanent error"))

        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=3,
                    retry_condition=retry_condition
                )

        # Should fail immediately without retry
        assert mock_func.call_count == 1

    def test_retry_condition_allows_retry(self, retry_mixin):
        """retry_condition returning True should allow retry."""
        retry_condition = lambda e: "transient" in str(e)

        mock_func = Mock(side_effect=[
            Exception("transient error"),
            Exception("transient error"),
            "success"
        ])

        with patch('time.sleep'):
            result = retry_mixin.execute_with_retry(
                mock_func,
                max_retries=3,
                retry_condition=retry_condition
            )

        assert result == "success"
        assert mock_func.call_count == 3

    def test_on_retry_callback_called(self, retry_mixin):
        """on_retry callback should be called before each retry."""
        on_retry_calls = []
        on_retry = lambda attempt, exc, delay: on_retry_calls.append((attempt, str(exc), delay))

        mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])

        with patch('time.sleep'):
            retry_mixin.execute_with_retry(
                mock_func,
                max_retries=3,
                base_delay=10.0,
                on_retry=on_retry
            )

        assert len(on_retry_calls) == 2
        assert on_retry_calls[0][0] == 1  # First retry attempt
        assert on_retry_calls[1][0] == 2  # Second retry attempt

    def test_retryable_exceptions_filter(self, retry_mixin):
        """Only specified exception types should be retried."""
        mock_func = Mock(side_effect=ValueError("not retryable"))

        with patch('time.sleep'):
            with pytest.raises(ValueError):
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=3,
                    retryable_exceptions=(IOError,)  # Not ValueError
                )

        # Should fail immediately
        assert mock_func.call_count == 1

    def test_logs_warning_on_retry(self, retry_mixin):
        """Should log warning message on each retry."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])

        with patch('time.sleep'):
            retry_mixin.execute_with_retry(
                mock_func,
                max_retries=2,
                base_delay=10.0
            )

        # Logger should have been called
        retry_mixin.logger.warning.assert_called()


# =============================================================================
# HTTP Error Classification Tests
# =============================================================================

@pytest.mark.mixin_retry
@pytest.mark.acquisition
class TestIsRetryableHttpError:
    """Tests for is_retryable_http_error method."""

    @pytest.mark.parametrize("error_msg", [
        "Connection timed out",
        "Request timeout",
        "503 Service Unavailable",
        "502 Bad Gateway",
        "500 Internal Server Error",
        "429 Too Many Requests",
        "Connection reset by peer",
        "Connection refused",
        "Connection aborted",
        "Broken pipe",
        "Network unreachable",
        "Server temporarily unavailable",
        "Temporary maintenance",
    ])
    def test_transient_errors_are_retryable(self, retry_mixin, error_msg):
        """Transient errors should be retryable."""
        error = Exception(error_msg)

        result = retry_mixin.is_retryable_http_error(error)

        assert result is True, f"'{error_msg}' should be retryable"

    @pytest.mark.parametrize("error_msg", [
        "401 Unauthorized",
        "404 Not Found",
        "Invalid credentials",
        "Authentication failed",
        "Request too large",
        "Cost limits exceeded",
        "Quota exceeded",
    ])
    def test_permanent_errors_not_retryable(self, retry_mixin, error_msg):
        """Permanent errors should not be retryable."""
        error = Exception(error_msg)

        result = retry_mixin.is_retryable_http_error(error)

        assert result is False, f"'{error_msg}' should not be retryable"

    def test_403_temporary_is_retryable(self, retry_mixin):
        """403 with 'temporarily' should be retryable."""
        error = Exception("403 Forbidden: temporarily unavailable")

        result = retry_mixin.is_retryable_http_error(error)

        assert result is True

    def test_403_rate_limit_is_retryable(self, retry_mixin):
        """403 with 'rate' should be retryable."""
        error = Exception("403 Forbidden: rate limit exceeded")

        result = retry_mixin.is_retryable_http_error(error)

        assert result is True

    def test_403_permanent_not_retryable(self, retry_mixin):
        """403 without temporary indicators should not be retryable."""
        error = Exception("403 Forbidden: access denied")

        result = retry_mixin.is_retryable_http_error(error)

        assert result is False

    def test_unknown_error_not_retryable(self, retry_mixin):
        """Unknown errors should not be retryable by default."""
        error = Exception("Some unknown error occurred")

        result = retry_mixin.is_retryable_http_error(error)

        assert result is False


# =============================================================================
# CDS Error Classification Tests
# =============================================================================

@pytest.mark.mixin_retry
@pytest.mark.acquisition
class TestIsRetryableCdsError:
    """Tests for is_retryable_cds_error method."""

    def test_too_large_not_retryable(self, retry_mixin):
        """'Too large' errors should not be retryable."""
        error = Exception("Request too large, please reduce the area")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is False

    def test_cost_limits_not_retryable(self, retry_mixin):
        """'Cost limits exceeded' should not be retryable."""
        error = Exception("Cost limits exceeded")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is False

    def test_403_is_retryable(self, retry_mixin):
        """403 errors should be retryable for CDS (often rate limits)."""
        error = Exception("403 Forbidden")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is True

    def test_maintenance_is_retryable(self, retry_mixin):
        """Maintenance messages should be retryable."""
        error = Exception("CDS is under maintenance")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is True

    def test_temporarily_is_retryable(self, retry_mixin):
        """'Temporarily' messages should be retryable."""
        error = Exception("Service temporarily unavailable")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is True

    def test_general_http_errors_delegated(self, retry_mixin):
        """General HTTP errors should delegate to is_retryable_http_error."""
        # 500 errors are retryable via general HTTP logic
        error = Exception("500 Internal Server Error")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is True

        # 404 errors are not retryable
        error = Exception("404 Not Found")

        result = retry_mixin.is_retryable_cds_error(error)

        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.mixin_retry
@pytest.mark.acquisition
class TestRetryMixinIntegration:
    """Integration tests combining retry methods."""

    def test_retry_with_is_retryable_http_error(self, retry_mixin):
        """execute_with_retry using is_retryable_http_error as condition."""
        # First call fails with retryable error, second succeeds
        mock_func = Mock(side_effect=[
            Exception("503 Service Unavailable"),
            "success"
        ])

        with patch('time.sleep'):
            result = retry_mixin.execute_with_retry(
                mock_func,
                max_retries=3,
                retry_condition=retry_mixin.is_retryable_http_error
            )

        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_with_non_retryable_stops_immediately(self, retry_mixin):
        """Non-retryable errors should stop immediately with condition."""
        mock_func = Mock(side_effect=Exception("404 Not Found"))

        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=3,
                    retry_condition=retry_mixin.is_retryable_http_error
                )

        assert "404" in str(exc_info.value)
        assert mock_func.call_count == 1  # No retries

    def test_realistic_cds_retry_scenario(self, retry_mixin):
        """Simulate realistic CDS API retry scenario."""
        # Sequence: rate limit, maintenance, success
        mock_func = Mock(side_effect=[
            Exception("403 Forbidden: rate limit"),
            Exception("CDS under maintenance"),
            "data_downloaded"
        ])

        with patch('time.sleep'):
            result = retry_mixin.execute_with_retry(
                mock_func,
                max_retries=5,
                base_delay=60.0,
                retry_condition=retry_mixin.is_retryable_cds_error
            )

        assert result == "data_downloaded"
        assert mock_func.call_count == 3

    def test_realistic_cds_failure_scenario(self, retry_mixin):
        """Simulate CDS API failure that shouldn't be retried."""
        mock_func = Mock(side_effect=Exception("Request too large"))

        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=5,
                    retry_condition=retry_mixin.is_retryable_cds_error
                )

        assert "too large" in str(exc_info.value).lower()
        assert mock_func.call_count == 1  # No retries


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.mixin_retry
@pytest.mark.acquisition
class TestRetryEdgeCases:
    """Edge case tests for retry mixin."""

    def test_zero_max_retries(self, retry_mixin):
        """With max_retries=0, should still execute once."""
        mock_func = Mock(side_effect=Exception("fail"))

        with patch('time.sleep'):
            with pytest.raises(Exception):
                retry_mixin.execute_with_retry(
                    mock_func,
                    max_retries=0
                )

        assert mock_func.call_count == 1

    def test_large_backoff_factor(self, retry_mixin):
        """Large backoff factor should still respect max_delay."""
        mock_func = Mock(side_effect=[Exception("fail")] * 3 + ["success"])
        delays = []

        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = lambda d: delays.append(d)
            retry_mixin.execute_with_retry(
                mock_func,
                max_retries=3,
                base_delay=10.0,
                backoff_factor=100.0,  # Very aggressive
                max_delay=100.0
            )

        # All delays should be capped at max_delay
        assert all(d <= 100.0 for d in delays)

    def test_function_with_no_return_value(self, retry_mixin):
        """Functions returning None should work correctly."""
        mock_func = Mock(return_value=None)

        result = retry_mixin.execute_with_retry(mock_func)

        assert result is None
        assert mock_func.call_count == 1
