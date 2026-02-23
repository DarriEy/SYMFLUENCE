"""
Unit Tests for AcquisitionRegistry.

Tests the registry pattern implementation:
- Handler registration
- Handler retrieval
- Case-insensitive lookups
- Error handling
"""

import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fixtures.acquisition_fixtures import MockConfigFactory

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def isolated_registry():
    """
    Create an isolated registry for testing without affecting global state.

    This fixture saves and restores R.acquisition_handlers entries so that
    tests run in isolation from globally-registered handlers.
    """
    from symfluence.core.registries import R
    from symfluence.data.acquisition.base import BaseAcquisitionHandler
    from symfluence.data.acquisition.registry import AcquisitionRegistry

    # Save original R.acquisition_handlers state
    original_entries = dict(R.acquisition_handlers._entries)
    original_meta = dict(R.acquisition_handlers._meta)
    original_aliases = dict(R.acquisition_handlers._aliases)

    # Clear for isolated testing
    R.acquisition_handlers.clear()

    yield AcquisitionRegistry

    # Restore original state
    R.acquisition_handlers._entries.clear()
    R.acquisition_handlers._entries.update(original_entries)
    R.acquisition_handlers._meta.clear()
    R.acquisition_handlers._meta.update(original_meta)
    R.acquisition_handlers._aliases.clear()
    R.acquisition_handlers._aliases.update(original_aliases)


@pytest.fixture
def mock_handler_class():
    """Create a mock handler class for registration testing."""
    from symfluence.data.acquisition.base import BaseAcquisitionHandler

    class MockHandler(BaseAcquisitionHandler):
        """Mock handler for registry testing."""

        def download(self, output_dir: Path) -> Path:
            return output_dir / "mock_output.nc"

    return MockHandler


# =============================================================================
# Registration Tests
# =============================================================================

@pytest.mark.acquisition
class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_register_decorator(self, isolated_registry, mock_handler_class):
        """Register decorator should add handler to registry."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        assert isolated_registry.is_registered('test_handler')

    def test_register_multiple_handlers(self, isolated_registry, mock_handler_class):
        """Multiple handlers can be registered."""
        @isolated_registry.register('handler_a')
        class HandlerA(mock_handler_class):
            pass

        @isolated_registry.register('handler_b')
        class HandlerB(mock_handler_class):
            pass

        assert isolated_registry.is_registered('handler_a')
        assert isolated_registry.is_registered('handler_b')

    def test_register_normalizes_to_lowercase(self, isolated_registry, mock_handler_class):
        """Registration should normalize names to lowercase."""
        @isolated_registry.register('TEST_HANDLER')
        class TestHandler(mock_handler_class):
            pass

        # Should be retrievable via lowercase key
        assert isolated_registry.is_registered('test_handler')

    def test_register_returns_class(self, isolated_registry, mock_handler_class):
        """Register decorator should return the class unchanged."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        # Class should still be usable directly
        assert TestHandler is not None
        assert issubclass(TestHandler, mock_handler_class)


# =============================================================================
# Retrieval Tests
# =============================================================================

@pytest.mark.acquisition
class TestHandlerRetrieval:
    """Tests for handler retrieval."""

    def test_get_handler_returns_instance(
        self, isolated_registry, mock_handler_class, mock_config, mock_logger
    ):
        """get_handler should return an instance of the registered handler."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        handler = isolated_registry.get_handler('test_handler', mock_config, mock_logger)

        assert handler is not None
        assert isinstance(handler, TestHandler)

    def test_get_handler_case_insensitive(
        self, isolated_registry, mock_handler_class, mock_config, mock_logger
    ):
        """get_handler should be case-insensitive."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        # All these should work
        handler1 = isolated_registry.get_handler('test_handler', mock_config, mock_logger)
        handler2 = isolated_registry.get_handler('TEST_HANDLER', mock_config, mock_logger)
        handler3 = isolated_registry.get_handler('Test_Handler', mock_config, mock_logger)

        assert all(h is not None for h in [handler1, handler2, handler3])

    def test_get_handler_passes_config(
        self, isolated_registry, mock_handler_class, mock_config, mock_logger
    ):
        """get_handler should pass config to handler constructor."""
        received_config = None

        class CapturingHandler(mock_handler_class):
            def __init__(self, config, logger):
                nonlocal received_config
                received_config = config
                super().__init__(config, logger)

        from symfluence.core.registries import R
        R.acquisition_handlers.add('test_handler', CapturingHandler)

        isolated_registry.get_handler('test_handler', mock_config, mock_logger)

        assert received_config is mock_config

    def test_get_handler_passes_logger(
        self, isolated_registry, mock_handler_class, mock_config, mock_logger
    ):
        """get_handler should pass logger to handler constructor."""
        received_logger = None

        class CapturingHandler(mock_handler_class):
            def __init__(self, config, logger):
                nonlocal received_logger
                received_logger = logger
                super().__init__(config, logger)

        from symfluence.core.registries import R
        R.acquisition_handlers.add('test_handler', CapturingHandler)

        isolated_registry.get_handler('test_handler', mock_config, mock_logger)

        assert received_logger is mock_logger

    def test_get_handler_unknown_raises(self, isolated_registry, mock_config, mock_logger):
        """get_handler should raise DataAcquisitionError for unknown handlers."""
        from symfluence.core.exceptions import DataAcquisitionError

        with pytest.raises(DataAcquisitionError):
            isolated_registry.get_handler('nonexistent_handler', mock_config, mock_logger)

    def test_get_handler_error_message_includes_available(
        self, isolated_registry, mock_handler_class, mock_config, mock_logger
    ):
        """Error message should list available handlers."""
        @isolated_registry.register('available_handler')
        class AvailableHandler(mock_handler_class):
            pass

        try:
            isolated_registry.get_handler('missing_handler', mock_config, mock_logger)
            pytest.fail("Should have raised DataAcquisitionError")
        except Exception as e:  # noqa: BLE001
            assert 'available_handler' in str(e).lower()


# =============================================================================
# Handler Class Retrieval Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetHandlerClass:
    """Tests for _get_handler_class method."""

    def test_get_handler_class_returns_class(self, isolated_registry, mock_handler_class):
        """_get_handler_class should return the class, not instance."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        handler_class = isolated_registry._get_handler_class('test_handler')

        assert handler_class is TestHandler
        assert isinstance(handler_class, type)

    def test_get_handler_class_case_insensitive(self, isolated_registry, mock_handler_class):
        """_get_handler_class should be case-insensitive."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        handler_class = isolated_registry._get_handler_class('TEST_HANDLER')

        assert handler_class is TestHandler

    def test_get_handler_class_unknown_raises(self, isolated_registry):
        """_get_handler_class should raise ValueError for unknown handlers."""
        with pytest.raises(ValueError):
            isolated_registry._get_handler_class('nonexistent')


# =============================================================================
# Listing Tests
# =============================================================================

@pytest.mark.acquisition
class TestListHandlers:
    """Tests for list_handlers and related methods."""

    def test_list_handlers_empty_registry(self, isolated_registry):
        """list_handlers should return empty list for empty registry."""
        result = isolated_registry.list_handlers()

        assert result == []

    def test_list_handlers_returns_all(self, isolated_registry, mock_handler_class):
        """list_handlers should return all registered handlers."""
        @isolated_registry.register('handler_a')
        class HandlerA(mock_handler_class):
            pass

        @isolated_registry.register('handler_b')
        class HandlerB(mock_handler_class):
            pass

        result = isolated_registry.list_handlers()

        assert 'handler_a' in result
        assert 'handler_b' in result

    def test_list_handlers_sorted(self, isolated_registry, mock_handler_class):
        """list_handlers should return sorted list."""
        @isolated_registry.register('zebra')
        class Zebra(mock_handler_class):
            pass

        @isolated_registry.register('alpha')
        class Alpha(mock_handler_class):
            pass

        @isolated_registry.register('beta')
        class Beta(mock_handler_class):
            pass

        result = isolated_registry.list_handlers()

        assert result == sorted(result)
        assert result == ['alpha', 'beta', 'zebra']

    def test_list_datasets_alias(self, isolated_registry, mock_handler_class):
        """list_datasets should be alias for list_handlers."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        handlers = isolated_registry.list_handlers()
        datasets = isolated_registry.list_datasets()

        assert handlers == datasets


# =============================================================================
# Is Registered Tests
# =============================================================================

@pytest.mark.acquisition
class TestIsRegistered:
    """Tests for is_registered method."""

    def test_is_registered_true(self, isolated_registry, mock_handler_class):
        """is_registered should return True for registered handlers."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        assert isolated_registry.is_registered('test_handler') is True

    def test_is_registered_false(self, isolated_registry):
        """is_registered should return False for unregistered handlers."""
        assert isolated_registry.is_registered('nonexistent') is False

    def test_is_registered_case_insensitive(self, isolated_registry, mock_handler_class):
        """is_registered should be case-insensitive."""
        @isolated_registry.register('test_handler')
        class TestHandler(mock_handler_class):
            pass

        assert isolated_registry.is_registered('TEST_HANDLER') is True
        assert isolated_registry.is_registered('Test_Handler') is True


# =============================================================================
# Clear Tests
# =============================================================================

@pytest.mark.acquisition
class TestClearRegistry:
    """Tests for clear method."""

    def test_clear_removes_all_handlers(self, isolated_registry, mock_handler_class):
        """clear should remove all registered handlers."""
        @isolated_registry.register('handler_a')
        class HandlerA(mock_handler_class):
            pass

        @isolated_registry.register('handler_b')
        class HandlerB(mock_handler_class):
            pass

        isolated_registry.clear()

        assert isolated_registry.list_handlers() == []


# =============================================================================
# Integration with Global Registry
# =============================================================================

@pytest.mark.acquisition
class TestGlobalRegistry:
    """Integration tests with the global AcquisitionRegistry."""

    def test_global_registry_has_handlers(self):
        """Global registry should have handlers registered."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        # Import handlers to ensure registration
        try:
            from symfluence.data.acquisition import handlers
        except ImportError:
            pass

        handlers_list = AcquisitionRegistry.list_handlers()

        assert len(handlers_list) > 0, "Global registry should have handlers"

    def test_era5_is_registered(self):
        """ERA5 handler should always be registered."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        try:
            from symfluence.data.acquisition import handlers
        except ImportError:
            pass

        assert AcquisitionRegistry.is_registered('era5'), (
            "ERA5 handler should be registered"
        )

    def test_get_handler_with_typed_config(self, mock_logger):
        """get_handler should work with SymfluenceConfig object."""
        from symfluence.core.config.models import SymfluenceConfig
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        try:
            from symfluence.data.acquisition import handlers
        except ImportError:
            pass

        config_dict = MockConfigFactory.create()
        typed_config = SymfluenceConfig(**config_dict)

        handlers_list = AcquisitionRegistry.list_handlers()
        if handlers_list:
            handler_name = handlers_list[0]
            try:
                handler = AcquisitionRegistry.get_handler(
                    handler_name, typed_config, mock_logger
                )
                assert handler is not None
            except Exception as e:
                # Some handlers may need credentials, etc.
                if "credential" not in str(e).lower():
                    raise
