"""
Unit test fixtures and configuration.

Fixtures specific to unit tests (fast, isolated tests).
"""

from unittest.mock import MagicMock

import pytest

# ============================================================================
# Common Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a basic mock configuration for unit tests."""
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/test',
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp'
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for unit tests."""
    return MagicMock()
