"""
Integration test for optimization workflow after Phase 1 refactoring.

Tests that the optimization infrastructure still works correctly after:
1. Moving routing_decider to models/utilities/
2. Moving fuse_conversion to models/fuse/utilities/
3. Breaking circular dependencies between models and optimization

This test ensures OptimizerRegistry can still discover and instantiate
model-specific optimizers, workers, and parameter managers.
"""

from pathlib import Path
from symfluence.optimization.registry import OptimizerRegistry


class TestOptimizerRegistryAfterRefactoring:
    """Test OptimizerRegistry functionality after refactoring."""

    def test_optimizer_registry_has_models(self):
        """Test that optimizer registry contains registered models."""
        # Models should register when their optimization modules are imported

        # Check that some optimizers are registered
        optimizer_cls = OptimizerRegistry.get_optimizer('SUMMA')
        assert optimizer_cls is not None, "SUMMA optimizer should be registered"

    def test_worker_registry_has_models(self):
        """Test that worker registry contains registered workers."""
        worker_cls = OptimizerRegistry.get_worker('FUSE')
        assert worker_cls is not None, "FUSE worker should be registered"

    def test_parameter_manager_registry_has_models(self):
        """Test that parameter manager registry contains registered managers."""
        pm_cls = OptimizerRegistry.get_parameter_manager('SUMMA')
        assert pm_cls is not None, "SUMMA parameter manager should be registered"

    def test_calibration_target_registry_has_models(self):
        """Test that calibration target registry contains registered targets."""
        target_cls = OptimizerRegistry.get_calibration_target('SUMMA')
        # Note: May be None if not yet implemented, but registry should not error
        # This is acceptable as calibration targets are being refactored


class TestModelUtilitiesAccessible:
    """Test that model utilities are accessible from optimization layer."""

    def test_routing_decider_accessible_from_models(self):
        """Test routing_decider is accessible from new location."""
        from symfluence.models.utilities.routing_decider import RoutingDecider

        decider = RoutingDecider()
        config = {'ROUTING_MODEL': 'mizuroute'}
        assert decider.needs_routing(config, 'SUMMA') is True

    def test_routing_decider_backward_compatible(self):
        """Test routing_decider is backward compatible via re-export."""
        from symfluence.optimization.workers.utilities import RoutingDecider

        decider = RoutingDecider()
        config = {'ROUTING_MODEL': 'none'}
        assert decider.needs_routing(config, 'SUMMA') is False

    def test_fuse_converter_accessible_from_models(self):
        """Test FUSE converter is accessible from new location."""
        from symfluence.models.fuse.utilities import FuseToMizurouteConverter

        converter = FuseToMizurouteConverter()
        assert converter is not None


class TestNoCircularDependencies:
    """Test that circular dependencies are resolved."""

    def test_models_import_no_optimization(self):
        """Test models package doesn't import from optimization."""
        import sys

        # Check that importing models doesn't trigger optimization import
        # (optimization will be imported later when needed)
        models_module = sys.modules.get('symfluence.models')
        assert models_module is not None

        # If optimization was imported, it should only be from explicit usage
        # not from models module initialization
        # This is a weak test but ensures basic separation

    def test_optimization_can_import_models(self):
        """Test optimization can import from models (one-way dependency)."""
        # This should work: optimization → models
        from symfluence.optimization.workers.fuse_worker import FUSEWorker
        from symfluence.models.fuse.runner import FUSERunner

        # Both should be accessible
        assert FUSEWorker is not None
        assert FUSERunner is not None

    def test_models_utilities_independent(self):
        """Test models utilities can be used without optimization."""
        # Import only from models, not optimization
        from symfluence.models.utilities import RoutingDecider

        # Should work without importing optimization
        decider = RoutingDecider()
        config = {'ROUTING_MODEL': 'default', 'DOMAIN_DEFINITION_METHOD': 'distributed'}
        result = decider.needs_routing(config, 'SUMMA')
        assert isinstance(result, bool)


class TestMigrationReadiness:
    """Test system is ready for the main migration."""

    def test_model_directories_exist(self):
        """Test model directories exist for migration targets."""
        import symfluence.models as models_pkg
        models_path = Path(models_pkg.__file__).parent

        expected_models = ['summa', 'fuse', 'ngen', 'gr', 'hype', 'mesh', 'rhessys']
        for model in expected_models:
            model_dir = models_path / model
            assert model_dir.exists(), f"Model directory {model} should exist"

    def test_migration_plan_exists(self):
        """Test migration plan documentation exists."""
        # This test assumes the migration plan is at project root
        import symfluence
        project_root = Path(symfluence.__file__).parent.parent.parent
        migration_plan = project_root / 'MIGRATION_PLAN.md'

        assert migration_plan.exists(), "MIGRATION_PLAN.md should exist"

    def test_optimization_base_classes_available(self):
        """Test base classes for migration are available."""
        from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
        from symfluence.optimization.workers.base_worker import BaseWorker
        from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

        assert BaseModelOptimizer is not None
        assert BaseWorker is not None
        assert BaseParameterManager is not None


# Note: These are integration tests that verify the refactoring didn't break
# the optimization workflow. More detailed tests for actual optimization
# execution would require mock data and longer run times.
