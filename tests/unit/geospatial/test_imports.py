"""
Import validation tests for geospatial module.

Tests that all public classes and functions can be imported without errors
or circular import issues.

This test file serves as a canary for import problems - if imports fail here,
they will fail for users of the module.
"""

import pytest
import time
import importlib
import sys


class TestTopLevelImports:
    """Tests for top-level module imports."""

    def test_import_delineation_module(self):
        """Test that delineation module can be imported."""
        from symfluence.geospatial import delineation
        assert delineation is not None

    def test_import_domain_delineator(self):
        """Test that DomainDelineator can be imported."""
        from symfluence.geospatial.delineation import DomainDelineator
        assert DomainDelineator is not None

    def test_import_delineation_artifacts(self):
        """Test that DelineationArtifacts can be imported."""
        from symfluence.geospatial.delineation import DelineationArtifacts
        assert DelineationArtifacts is not None


class TestDelineatorImports:
    """Tests for individual delineator class imports."""

    def test_import_grid_delineator(self):
        """Test that GridDelineator can be imported."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator
        assert GridDelineator is not None

    def test_import_lumped_delineator(self):
        """Test that LumpedWatershedDelineator can be imported."""
        from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
        assert LumpedWatershedDelineator is not None

    def test_import_point_delineator(self):
        """Test that PointDelineator can be imported."""
        from symfluence.geospatial.geofabric.delineators.point_delineator import PointDelineator
        assert PointDelineator is not None

    def test_import_distributed_delineator(self):
        """Test that GeofabricDelineator can be imported."""
        from symfluence.geospatial.geofabric.delineators.distributed_delineator import GeofabricDelineator
        assert GeofabricDelineator is not None


class TestRegistryImports:
    """Tests for registry and protocol imports."""

    def test_import_delineation_registry(self):
        """Test that DelineationRegistry can be imported."""
        from symfluence.geospatial.delineation_registry import DelineationRegistry
        assert DelineationRegistry is not None

    def test_import_delineation_protocol(self):
        """Test that DelineationResult can be imported."""
        from symfluence.geospatial.delineation_protocol import DelineationResult
        assert DelineationResult is not None

    def test_import_delineation_strategy(self):
        """Test that DelineationStrategy protocol can be imported."""
        from symfluence.geospatial.delineation_protocol import DelineationStrategy
        assert DelineationStrategy is not None


class TestExceptionImports:
    """Tests for exception class imports."""

    def test_import_geospatial_error(self):
        """Test that GeospatialError can be imported."""
        from symfluence.geospatial.exceptions import GeospatialError
        assert GeospatialError is not None

    def test_import_delineation_error(self):
        """Test that DelineationError can be imported."""
        from symfluence.geospatial.exceptions import DelineationError
        assert DelineationError is not None

    def test_import_taudem_error(self):
        """Test that TauDEMError can be imported."""
        from symfluence.geospatial.exceptions import TauDEMError
        assert TauDEMError is not None

    def test_import_grid_creation_error(self):
        """Test that GridCreationError can be imported."""
        from symfluence.geospatial.exceptions import GridCreationError
        assert GridCreationError is not None

    def test_import_error_handler(self):
        """Test that geospatial_error_handler can be imported."""
        from symfluence.geospatial.exceptions import geospatial_error_handler
        assert geospatial_error_handler is not None


class TestUtilityImports:
    """Tests for utility module imports."""

    def test_import_raster_utils(self):
        """Test that raster_utils can be imported."""
        from symfluence.geospatial import raster_utils
        assert raster_utils is not None

    def test_import_scipy_mode_compat(self):
        """Test that _scipy_mode_compat can be imported."""
        from symfluence.geospatial.raster_utils import _scipy_mode_compat
        assert _scipy_mode_compat is not None


class TestCircularImportPrevention:
    """Tests to ensure no circular imports occur."""

    def test_import_time_reasonable(self):
        """Test that import time is reasonable (no hang from circular imports)."""
        # Clear any cached imports
        modules_to_clear = [
            key for key in sys.modules.keys()
            if key.startswith('symfluence.geospatial')
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start_time = time.time()
        from symfluence.geospatial import delineation
        import_time = time.time() - start_time

        # Import should complete in under 5 seconds (generous for CI)
        assert import_time < 5.0, f"Import took {import_time:.2f}s, possible circular import"

    def test_registry_before_delineators(self):
        """Test that registry can be imported before delineators."""
        # Clear cached imports
        modules_to_clear = [
            key for key in sys.modules.keys()
            if key.startswith('symfluence.geospatial')
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import registry first
        from symfluence.geospatial.delineation_registry import DelineationRegistry
        assert DelineationRegistry is not None

        # Then import delineators (which should register themselves)
        from symfluence.geospatial.geofabric.delineators.point_delineator import PointDelineator
        assert PointDelineator is not None

    def test_exceptions_standalone(self):
        """Test that exceptions module can be imported standalone."""
        # Clear cached imports
        modules_to_clear = [
            key for key in sys.modules.keys()
            if key.startswith('symfluence.geospatial')
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import exceptions module first
        from symfluence.geospatial.exceptions import GeospatialError
        assert GeospatialError is not None


class TestRegistryRegistration:
    """Tests that delineators are properly registered."""

    def test_point_delineator_registered(self):
        """Test that PointDelineator is registered as 'point'."""
        # Import to trigger registration
        from symfluence.geospatial.geofabric.delineators.point_delineator import PointDelineator
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        strategy = DelineationRegistry.get_strategy('point')
        assert strategy is PointDelineator

    def test_lumped_delineator_registered(self):
        """Test that LumpedWatershedDelineator is registered as 'lumped'."""
        from symfluence.geospatial.geofabric.delineators.lumped_delineator import LumpedWatershedDelineator
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        strategy = DelineationRegistry.get_strategy('lumped')
        assert strategy is LumpedWatershedDelineator

    def test_distributed_delineator_registered(self):
        """Test that GridDelineator is registered as 'distributed'."""
        from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        strategy = DelineationRegistry.get_strategy('distributed')
        assert strategy is GridDelineator

    def test_semidistributed_delineator_registered(self):
        """Test that GeofabricDelineator is registered as 'semidistributed'."""
        from symfluence.geospatial.geofabric.delineators.distributed_delineator import GeofabricDelineator
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        strategy = DelineationRegistry.get_strategy('semidistributed')
        assert strategy is GeofabricDelineator


class TestRegistryAliases:
    """Tests for registry alias functionality."""

    def test_delineate_alias_resolves(self):
        """Test that 'delineate' alias resolves to 'semidistributed'."""
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        canonical = DelineationRegistry.get_canonical_name('delineate')
        assert canonical == 'semidistributed'

    def test_distribute_alias_resolves(self):
        """Test that 'distribute' alias resolves to 'distributed'."""
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        canonical = DelineationRegistry.get_canonical_name('distribute')
        assert canonical == 'distributed'

    def test_unknown_method_returns_none(self):
        """Test that unknown method name returns None."""
        from symfluence.geospatial.delineation_registry import DelineationRegistry

        strategy = DelineationRegistry.get_strategy('unknown_method')
        assert strategy is None


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_delineation_error_is_geospatial_error(self):
        """Test that DelineationError inherits from GeospatialError."""
        from symfluence.geospatial.exceptions import DelineationError, GeospatialError

        assert issubclass(DelineationError, GeospatialError)

    def test_taudem_error_is_delineation_error(self):
        """Test that TauDEMError inherits from DelineationError."""
        from symfluence.geospatial.exceptions import TauDEMError, DelineationError

        assert issubclass(TauDEMError, DelineationError)

    def test_grid_creation_error_is_delineation_error(self):
        """Test that GridCreationError inherits from DelineationError."""
        from symfluence.geospatial.exceptions import GridCreationError, DelineationError

        assert issubclass(GridCreationError, DelineationError)

    def test_exception_can_be_raised_and_caught(self):
        """Test that custom exceptions can be raised and caught."""
        from symfluence.geospatial.exceptions import TauDEMError, DelineationError, GeospatialError

        with pytest.raises(TauDEMError):
            raise TauDEMError("TauDEM failed")

        # Should also be catchable as parent types
        with pytest.raises(DelineationError):
            raise TauDEMError("TauDEM failed")

        with pytest.raises(GeospatialError):
            raise TauDEMError("TauDEM failed")
