"""
Mock Factories for Acquisition Handler Tests.

Provides reusable mock factories for testing data acquisition handlers:
- MockConfigFactory: Generate test configs with customizable settings
- MockResponseFactory: Create HTTP response mocks
- MockSessionFactory: Create mock requests.Session with response mapping
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest


# =============================================================================
# Mock Configuration Factory
# =============================================================================

@dataclass
class MockConfigFactory:
    """Factory for generating test configurations."""

    # Default bounding box (small area in Switzerland for fast tests)
    # Format: north/west/south/east (lat_max/lon_min/lat_min/lon_max)
    default_bbox: str = "47.0/8.0/46.0/9.0"  # north/west/south/east

    # Default time range (short period for fast tests)
    default_start: str = "2020-01-01"
    default_end: str = "2020-01-31"

    # Default data directory
    default_data_dir: str = "/tmp/symfluence_test"

    @classmethod
    def create(
        cls,
        bbox: str = None,
        start_date: str = None,
        end_date: str = None,
        data_dir: str = None,
        domain_name: str = "test_domain",
        forcing_dataset: str = "ERA5",
        force_download: bool = False,
        **extra_config
    ) -> Dict[str, Any]:
        """
        Create a test configuration dictionary.

        Args:
            bbox: Bounding box string (lat_min/lon_min/lat_max/lon_max)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            data_dir: Base data directory path
            domain_name: Domain name for the project
            forcing_dataset: Forcing dataset name (e.g., ERA5, CARRA)
            force_download: Whether to force re-download existing files
            **extra_config: Additional configuration keys to include

        Returns:
            Configuration dictionary suitable for handlers
        """
        factory = cls()
        base_data_dir = data_dir or factory.default_data_dir

        config = {
            # Core settings
            "DOMAIN_NAME": domain_name,
            "DATA_DIR": base_data_dir,

            # Bounding box
            "BOUNDING_BOX_COORDS": bbox or factory.default_bbox,

            # Time range
            "EXPERIMENT_TIME_START": start_date or factory.default_start,
            "EXPERIMENT_TIME_END": end_date or factory.default_end,

            # Data acquisition settings
            "FORCING_DATASET": forcing_dataset,
            "FORCE_DOWNLOAD": force_download,

            # Paths (derived)
            "PROJECT_DIR": f"{base_data_dir}/domain_{domain_name}",

            # Required SymfluenceConfig fields
            "SYMFLUENCE_DATA_DIR": base_data_dir,
            "SYMFLUENCE_CODE_DIR": "/tmp/symfluence_code",
            "EXPERIMENT_ID": "test_run",
            "DOMAIN_DEFINITION_METHOD": "lumped",
            "SUB_GRID_DISCRETIZATION": "lumped",
            "HYDROLOGICAL_MODEL": "SUMMA",
        }

        # Add extra config
        config.update(extra_config)

        return config

    @classmethod
    def create_minimal(cls) -> Dict[str, Any]:
        """Create a minimal config with only required fields."""
        return cls.create()

    @classmethod
    def create_with_credentials(
        cls,
        earthdata_user: str = "test_user",
        earthdata_pass: str = "test_pass",
        cds_url: str = "https://cds.climate.copernicus.eu/api",
        cds_key: str = "test_uid:test_key",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a config with credential settings."""
        config = cls.create(**kwargs)
        config.update({
            "EARTHDATA_USERNAME": earthdata_user,
            "EARTHDATA_PASSWORD": earthdata_pass,
            "CDSAPI_URL": cds_url,
            "CDSAPI_KEY": cds_key,
        })
        return config


# =============================================================================
# Mock HTTP Response Factory
# =============================================================================

@dataclass
class MockResponse:
    """Mock HTTP response object."""

    status_code: int = 200
    content: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)
    url: str = "https://example.com/data"
    text: str = ""
    json_data: Optional[Dict] = None
    raise_for_status_error: Optional[Exception] = None

    def __post_init__(self):
        if not self.headers:
            self.headers = {"content-length": str(len(self.content))}
        if not self.text and self.content:
            try:
                self.text = self.content.decode("utf-8")
            except UnicodeDecodeError:
                self.text = ""

    def raise_for_status(self):
        """Raise an exception if status code indicates error."""
        if self.raise_for_status_error:
            raise self.raise_for_status_error
        if self.status_code >= 400:
            from requests.exceptions import HTTPError
            raise HTTPError(f"{self.status_code} Error")

    def json(self):
        """Return JSON data."""
        if self.json_data is not None:
            return self.json_data
        import json
        return json.loads(self.text) if self.text else {}

    def iter_content(self, chunk_size=1024):
        """Iterate over response content in chunks."""
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockResponseFactory:
    """Factory for creating mock HTTP responses."""

    @staticmethod
    def success(content: bytes = b"test content", **kwargs) -> MockResponse:
        """Create a successful response."""
        return MockResponse(
            status_code=200,
            content=content,
            headers={"content-length": str(len(content))},
            **kwargs
        )

    @staticmethod
    def json_success(data: Dict, **kwargs) -> MockResponse:
        """Create a successful JSON response."""
        import json
        content = json.dumps(data).encode("utf-8")
        return MockResponse(
            status_code=200,
            content=content,
            json_data=data,
            headers={
                "content-type": "application/json",
                "content-length": str(len(content))
            },
            **kwargs
        )

    @staticmethod
    def error(status_code: int = 500, message: str = "Server Error") -> MockResponse:
        """Create an error response."""
        from requests.exceptions import HTTPError
        return MockResponse(
            status_code=status_code,
            content=message.encode("utf-8"),
            raise_for_status_error=HTTPError(f"{status_code}: {message}")
        )

    @staticmethod
    def not_found(url: str = "https://example.com/missing") -> MockResponse:
        """Create a 404 Not Found response."""
        from requests.exceptions import HTTPError
        return MockResponse(
            status_code=404,
            content=b"Not Found",
            url=url,
            raise_for_status_error=HTTPError(f"404 Client Error: Not Found for url: {url}")
        )

    @staticmethod
    def timeout() -> MockResponse:
        """Create a timeout response (raises exception on access)."""
        from requests.exceptions import Timeout
        response = MockResponse(status_code=0)
        response.raise_for_status_error = Timeout("Connection timed out")
        return response

    @staticmethod
    def rate_limited(retry_after: int = 60) -> MockResponse:
        """Create a 429 rate limit response."""
        from requests.exceptions import HTTPError
        return MockResponse(
            status_code=429,
            content=b"Too Many Requests",
            headers={"retry-after": str(retry_after)},
            raise_for_status_error=HTTPError("429 Too Many Requests")
        )

    @staticmethod
    def geotiff_response(width: int = 100, height: int = 100) -> MockResponse:
        """Create a mock GeoTIFF response with minimal valid header."""
        # Minimal GeoTIFF header
        header = b"II*\x00"  # Little-endian TIFF marker
        # Add enough dummy data for the raster
        data = header + b"\x00" * (width * height)
        return MockResponse(
            status_code=200,
            content=data,
            headers={
                "content-type": "image/tiff",
                "content-length": str(len(data))
            }
        )


# =============================================================================
# Mock Session Factory
# =============================================================================

class MockSessionFactory:
    """Factory for creating mock requests.Session objects."""

    @staticmethod
    def create(
        responses: Dict[str, MockResponse] = None,
        default_response: MockResponse = None
    ) -> Mock:
        """
        Create a mock session with URL-based response mapping.

        Args:
            responses: Dict mapping URL patterns to MockResponse objects
            default_response: Response to return for unmapped URLs

        Returns:
            Mock Session object
        """
        if responses is None:
            responses = {}
        if default_response is None:
            default_response = MockResponseFactory.success()

        session = Mock()

        def get_side_effect(url, *args, **kwargs):
            for pattern, response in responses.items():
                if pattern in url:
                    return response
            return default_response

        session.get = Mock(side_effect=get_side_effect)
        session.post = Mock(side_effect=get_side_effect)
        session.head = Mock(side_effect=get_side_effect)

        # Add context manager support
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)

        return session

    @staticmethod
    def create_failing(
        error: Exception = None,
        fail_count: int = 1,
        then_succeed: MockResponse = None
    ) -> Mock:
        """
        Create a session that fails a specified number of times, then succeeds.

        Useful for testing retry logic.

        Args:
            error: Exception to raise on failure
            fail_count: Number of times to fail before succeeding
            then_succeed: Response to return after failures

        Returns:
            Mock Session object
        """
        from requests.exceptions import ConnectionError

        if error is None:
            error = ConnectionError("Connection refused")
        if then_succeed is None:
            then_succeed = MockResponseFactory.success()

        session = Mock()
        call_count = {"count": 0}

        def get_side_effect(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] <= fail_count:
                raise error
            return then_succeed

        session.get = Mock(side_effect=get_side_effect)
        session.post = Mock(side_effect=get_side_effect)

        return session


# =============================================================================
# Mock Logger Factory
# =============================================================================

def create_mock_logger(name: str = "test") -> logging.Logger:
    """
    Create a mock logger for testing.

    Returns a real logger with a NullHandler to avoid output during tests.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers
    logger.handlers.clear()
    # Add null handler to suppress output
    logger.addHandler(logging.NullHandler())
    return logger


def create_capturing_logger(name: str = "test") -> tuple:
    """
    Create a logger that captures messages for assertion.

    Returns:
        Tuple of (logger, captured_records_list)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    captured = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    logger.addHandler(CapturingHandler())
    return logger, captured


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_config_factory():
    """Fixture providing MockConfigFactory."""
    return MockConfigFactory


@pytest.fixture
def mock_response_factory():
    """Fixture providing MockResponseFactory."""
    return MockResponseFactory


@pytest.fixture
def mock_session_factory():
    """Fixture providing MockSessionFactory."""
    return MockSessionFactory


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger."""
    return create_mock_logger("test_acquisition")


@pytest.fixture
def capturing_logger():
    """Fixture providing a logger that captures messages."""
    return create_capturing_logger("test_acquisition_capture")


@pytest.fixture
def mock_config():
    """Fixture providing a default mock configuration."""
    return MockConfigFactory.create()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing a temporary output directory."""
    output_dir = tmp_path / "acquisition_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_session():
    """Fixture providing a basic mock session."""
    return MockSessionFactory.create()


# =============================================================================
# Context Managers for Patching
# =============================================================================

def patch_requests_session(responses: Dict[str, MockResponse] = None):
    """
    Context manager to patch requests.Session with mock responses.

    Args:
        responses: Dict mapping URL patterns to MockResponse objects

    Usage:
        with patch_requests_session({"example.com": MockResponseFactory.success()}):
            result = handler.download(output_dir)
    """
    mock_session = MockSessionFactory.create(responses=responses)
    return patch("requests.Session", return_value=mock_session)


def patch_robust_session(responses: Dict[str, MockResponse] = None):
    """
    Context manager to patch create_robust_session.

    Args:
        responses: Dict mapping URL patterns to MockResponse objects
    """
    mock_session = MockSessionFactory.create(responses=responses)
    return patch(
        "symfluence.data.acquisition.utils.create_robust_session",
        return_value=mock_session
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MockConfigFactory",
    "MockResponse",
    "MockResponseFactory",
    "MockSessionFactory",
    "create_mock_logger",
    "create_capturing_logger",
    "patch_requests_session",
    "patch_robust_session",
]
