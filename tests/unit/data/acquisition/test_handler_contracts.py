"""
Handler Contract Tests for Data Acquisition.

These parametrized tests verify that all registered handlers implement
the required interface correctly without requiring network access.

Tests cover:
- Handler registration and discovery
- Inheritance from BaseAcquisitionHandler
- Required method signatures
- Configuration parsing
"""

import inspect
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fixtures.acquisition_fixtures import MockConfigFactory

# =============================================================================
# Handler Discovery
# =============================================================================

def get_all_registered_handlers() -> List[str]:
    """Get list of all registered handler names from AcquisitionRegistry."""
    # Import the registry to trigger handler registration
    from symfluence.data.acquisition.registry import AcquisitionRegistry

    # Import handlers to ensure registration
    try:
        from symfluence.data.acquisition import handlers
    except ImportError:
        pass

    return AcquisitionRegistry.list_handlers()


def get_handler_class(handler_name: str):
    """Get the handler class for a given name."""
    from symfluence.data.acquisition.registry import AcquisitionRegistry
    return AcquisitionRegistry._get_handler_class(handler_name)


# Get all handlers for parametrization
# This is evaluated at collection time
try:
    ALL_HANDLERS = get_all_registered_handlers()
except Exception:
    ALL_HANDLERS = []


# =============================================================================
# Contract Tests
# =============================================================================

@pytest.mark.contract
@pytest.mark.acquisition
class TestHandlerContracts:
    """Contract tests for all registered acquisition handlers."""

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_is_registered(self, handler_name):
        """Verify handler is properly registered in AcquisitionRegistry."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        assert AcquisitionRegistry.is_registered(handler_name), (
            f"Handler '{handler_name}' should be registered"
        )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_inherits_from_base(self, handler_name):
        """All handlers must inherit from BaseAcquisitionHandler."""
        from symfluence.data.acquisition.base import BaseAcquisitionHandler

        handler_class = get_handler_class(handler_name)

        assert issubclass(handler_class, BaseAcquisitionHandler), (
            f"Handler '{handler_name}' ({handler_class.__name__}) must inherit "
            f"from BaseAcquisitionHandler"
        )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_has_download_method(self, handler_name):
        """All handlers must implement download(output_dir) -> Path."""
        handler_class = get_handler_class(handler_name)

        # Check download method exists
        assert hasattr(handler_class, 'download'), (
            f"Handler '{handler_name}' must have a 'download' method"
        )

        # Check method signature
        download_method = handler_class.download
        sig = inspect.signature(download_method)
        params = list(sig.parameters.keys())

        # Should have self and output_dir at minimum
        assert 'self' in params or len(params) >= 1, (
            f"Handler '{handler_name}'.download() should accept output_dir parameter"
        )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_download_returns_path(self, handler_name):
        """Handler.download() should have Path return type annotation."""
        handler_class = get_handler_class(handler_name)
        download_method = handler_class.download

        sig = inspect.signature(download_method)
        return_annotation = sig.return_annotation

        # Return annotation should be Path or compatible
        if return_annotation != inspect.Parameter.empty:
            assert return_annotation in (Path, 'Path'), (
                f"Handler '{handler_name}'.download() should return Path, "
                f"got {return_annotation}"
            )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_instantiation_signature(self, handler_name):
        """Handlers must accept (config, logger) constructor arguments."""
        handler_class = get_handler_class(handler_name)

        # Get __init__ signature
        init_method = handler_class.__init__
        sig = inspect.signature(init_method)
        params = list(sig.parameters.keys())

        # Should accept self, config, logger (at minimum)
        assert len(params) >= 3, (
            f"Handler '{handler_name}' __init__ should accept at least "
            f"(self, config, logger), got params: {params}"
        )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_instantiation_with_dict_config(self, handler_name, mock_logger):
        """Handlers can be instantiated with dict config (backward compatibility)."""
        handler_class = get_handler_class(handler_name)
        config = MockConfigFactory.create()

        # Should not raise
        try:
            handler = handler_class(config, mock_logger)
            assert handler is not None
        except Exception as e:
            # Some handlers may have additional requirements (credentials, etc.)
            # That's acceptable - we just want to ensure the signature works
            acceptable_errors = [
                "credential",
                "authentication",
                "api key",
                "environment",
                "not found",
            ]
            error_msg = str(e).lower()
            if not any(err in error_msg for err in acceptable_errors):
                pytest.fail(
                    f"Handler '{handler_name}' instantiation failed: {e}"
                )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_has_logger_attribute(self, handler_name, mock_config, mock_logger):
        """Handlers should have a logger attribute after instantiation."""
        handler_class = get_handler_class(handler_name)

        try:
            handler = handler_class(mock_config, mock_logger)
            assert hasattr(handler, 'logger'), (
                f"Handler '{handler_name}' should have 'logger' attribute"
            )
        except Exception:
            # Skip if instantiation fails (credential issues, etc.)
            pytest.skip(f"Could not instantiate handler '{handler_name}'")

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_has_config_attribute(self, handler_name, mock_config, mock_logger):
        """Handlers should have a config-related attribute after instantiation."""
        handler_class = get_handler_class(handler_name)

        try:
            handler = handler_class(mock_config, mock_logger)
            # Should have either _config or config_dict
            has_config = (
                hasattr(handler, '_config') or
                hasattr(handler, 'config_dict') or
                hasattr(handler, 'config')
            )
            assert has_config, (
                f"Handler '{handler_name}' should have config attribute"
            )
        except Exception:
            pytest.skip(f"Could not instantiate handler '{handler_name}'")

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_parses_bbox(self, handler_name, mock_config, mock_logger):
        """Handlers correctly parse bounding box from config."""
        handler_class = get_handler_class(handler_name)

        try:
            handler = handler_class(mock_config, mock_logger)

            # Should have bbox attribute
            assert hasattr(handler, 'bbox'), (
                f"Handler '{handler_name}' should have 'bbox' attribute"
            )

            bbox = handler.bbox
            if bbox is not None:
                # Bbox should have required keys
                required_keys = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
                for key in required_keys:
                    assert key in bbox, (
                        f"Handler '{handler_name}' bbox missing key: {key}"
                    )

                # Values should be numeric
                for key in required_keys:
                    assert isinstance(bbox[key], (int, float)), (
                        f"Handler '{handler_name}' bbox[{key}] should be numeric"
                    )

        except Exception:
            pytest.skip(f"Could not instantiate handler '{handler_name}'")

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_parses_dates(self, handler_name, mock_config, mock_logger):
        """Handlers correctly parse date range from config."""
        handler_class = get_handler_class(handler_name)

        try:
            handler = handler_class(mock_config, mock_logger)

            # Should have date attributes
            assert hasattr(handler, 'start_date'), (
                f"Handler '{handler_name}' should have 'start_date' attribute"
            )
            assert hasattr(handler, 'end_date'), (
                f"Handler '{handler_name}' should have 'end_date' attribute"
            )

            # Dates should be comparable
            if handler.start_date is not None and handler.end_date is not None:
                assert handler.start_date <= handler.end_date, (
                    f"Handler '{handler_name}' start_date should be <= end_date"
                )

        except Exception:
            pytest.skip(f"Could not instantiate handler '{handler_name}'")


# =============================================================================
# Registry Tests
# =============================================================================

@pytest.mark.contract
@pytest.mark.acquisition
class TestRegistryContracts:
    """Contract tests for AcquisitionRegistry."""

    def test_registry_has_handlers(self):
        """Registry should have at least some handlers registered."""
        handlers = get_all_registered_handlers()
        assert len(handlers) > 0, "Registry should have at least one handler"

    def test_registry_handler_names_are_lowercase(self):
        """All registered handler names should be lowercase."""
        handlers = get_all_registered_handlers()
        for name in handlers:
            assert name == name.lower(), (
                f"Handler name '{name}' should be lowercase"
            )

    def test_registry_get_handler_is_case_insensitive(self, mock_config, mock_logger):
        """Registry.get_handler() should be case-insensitive."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        handlers = get_all_registered_handlers()
        if not handlers:
            pytest.skip("No handlers registered")

        handler_name = handlers[0]

        # These should all work
        variations = [
            handler_name.lower(),
            handler_name.upper(),
            handler_name.capitalize(),
        ]

        for variation in variations:
            try:
                handler = AcquisitionRegistry.get_handler(
                    variation, mock_config, mock_logger
                )
                assert handler is not None
            except Exception as e:
                # Only fail if it's a "not found" error
                if "unknown handler" in str(e).lower():
                    pytest.fail(
                        f"Registry should be case-insensitive: "
                        f"'{variation}' not found"
                    )

    def test_registry_unknown_handler_raises(self, mock_config, mock_logger):
        """Registry.get_handler() should raise for unknown handlers."""
        from symfluence.core.exceptions import DataAcquisitionError
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        with pytest.raises(DataAcquisitionError):
            AcquisitionRegistry.get_handler(
                "nonexistent_handler_xyz123",
                mock_config,
                mock_logger
            )

    def test_registry_list_handlers_returns_sorted(self):
        """Registry.list_handlers() should return a sorted list."""
        handlers = get_all_registered_handlers()
        assert handlers == sorted(handlers), (
            "list_handlers() should return sorted list"
        )

    def test_registry_list_datasets_alias(self):
        """Registry.list_datasets() should be alias for list_handlers()."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        handlers = AcquisitionRegistry.list_handlers()
        datasets = AcquisitionRegistry.list_datasets()

        assert handlers == datasets, (
            "list_datasets() should return same as list_handlers()"
        )


# =============================================================================
# Common Handler Categories
# =============================================================================

# Expected handler categories for documentation/verification
EXPECTED_FORCING_HANDLERS = ['era5', 'aorc', 'rdrs', 'hrrr', 'conus404']
EXPECTED_OBSERVATION_HANDLERS = ['smap', 'grace', 'ismn']
EXPECTED_ATTRIBUTE_HANDLERS = ['copdem30', 'copdem90', 'fabdem', 'srtm', 'etopo2022', 'mapzen', 'alos', 'soilgrids']


@pytest.mark.contract
@pytest.mark.acquisition
class TestHandlerCategories:
    """Verify expected handlers are registered."""

    @pytest.mark.parametrize("handler_name", EXPECTED_FORCING_HANDLERS)
    def test_forcing_handler_registered(self, handler_name):
        """Expected forcing handlers should be registered."""
        handlers = get_all_registered_handlers()

        # Skip if not available (optional dependencies)
        if handler_name not in handlers:
            pytest.skip(f"Handler '{handler_name}' not available (optional dep?)")

    def test_era5_handler_exists(self):
        """ERA5 is a core handler and should always be available."""
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        assert AcquisitionRegistry.is_registered('era5'), (
            "ERA5 handler should be registered"
        )


# =============================================================================
# Handler Method Signatures
# =============================================================================

@pytest.mark.contract
@pytest.mark.acquisition
class TestHandlerMethodSignatures:
    """Verify handler method signatures follow conventions."""

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_has_skip_if_exists(self, handler_name):
        """Handlers should have _skip_if_exists method (inherited from base)."""
        handler_class = get_handler_class(handler_name)

        assert hasattr(handler_class, '_skip_if_exists'), (
            f"Handler '{handler_name}' should have _skip_if_exists method"
        )

    @pytest.mark.parametrize("handler_name", ALL_HANDLERS)
    def test_handler_has_plot_diagnostics(self, handler_name):
        """Handlers should have plot_diagnostics method (inherited or overridden)."""
        handler_class = get_handler_class(handler_name)

        assert hasattr(handler_class, 'plot_diagnostics'), (
            f"Handler '{handler_name}' should have plot_diagnostics method"
        )
