"""
Tests for RoutingDecider utility.

Tests routing decision logic for various model configurations.
"""

from symfluence.models.utilities.routing_decider import RoutingDecider, needs_routing


class TestRoutingDecider:
    """Test RoutingDecider class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.decider = RoutingDecider()

    def test_routing_model_none(self):
        """Test that ROUTING_MODEL=none disables routing."""
        config = {
            'ROUTING_MODEL': 'none',
            'DOMAIN_DEFINITION_METHOD': 'distributed'
        }
        assert not self.decider.needs_routing(config, 'SUMMA')

    def test_routing_model_mizuroute(self):
        """Test that ROUTING_MODEL=mizuroute enables routing."""
        config = {
            'ROUTING_MODEL': 'mizuroute',
            'DOMAIN_DEFINITION_METHOD': 'lumped'
        }
        assert self.decider.needs_routing(config, 'SUMMA')

    def test_lumped_domain_lumped_routing(self):
        """Test that lumped domain with lumped routing disables routing."""
        config = {
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'ROUTING_DELINEATION': 'lumped'
        }
        assert not self.decider.needs_routing(config, 'SUMMA')

    def test_distributed_domain(self):
        """Test that distributed domain enables routing."""
        config = {
            'DOMAIN_DEFINITION_METHOD': 'distributed',
            'ROUTING_DELINEATION': 'lumped'
        }
        assert self.decider.needs_routing(config, 'SUMMA')

    def test_network_routing_delineation(self):
        """Test that river_network routing enables routing."""
        config = {
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'ROUTING_DELINEATION': 'river_network'
        }
        assert self.decider.needs_routing(config, 'SUMMA')

    def test_needs_routing_verbose(self):
        """Test verbose routing decision with diagnostics."""
        config = {
            'DOMAIN_DEFINITION_METHOD': 'distributed',
            'ROUTING_DELINEATION': 'lumped'
        }
        needs, diagnostics = self.decider.needs_routing_verbose(config, 'SUMMA')

        assert needs is True
        assert diagnostics['model'] == 'SUMMA'
        assert 'reason' in diagnostics
        assert 'checks' in diagnostics
        assert diagnostics['checks']['domain_method'] == 'distributed'

    def test_convenience_function(self):
        """Test module-level convenience function."""
        config = {
            'ROUTING_MODEL': 'mizuroute'
        }
        assert needs_routing(config, 'FUSE')

    def test_fuse_routing_integration(self):
        """Test FUSE-specific routing integration config."""
        config = {
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'ROUTING_DELINEATION': 'lumped',
            'FUSE_ROUTING_INTEGRATION': 'mizuRoute'
        }
        assert self.decider.needs_routing(config, 'FUSE')


class TestRoutingDeciderBackwardCompatibility:
    """Test backward compatibility with old import paths."""

    def test_import_from_optimization_utilities(self):
        """Test that old import path still works."""
        # This should work due to re-export in optimization/workers/utilities/__init__.py
        from symfluence.optimization.workers.utilities import RoutingDecider as OldRoutingDecider

        decider = OldRoutingDecider()
        config = {'ROUTING_MODEL': 'mizuroute'}
        assert decider.needs_routing(config, 'SUMMA')

    def test_import_from_models_utilities(self):
        """Test new import path."""
        from symfluence.models.utilities import RoutingDecider as NewRoutingDecider

        decider = NewRoutingDecider()
        config = {'ROUTING_MODEL': 'mizuroute'}
        assert decider.needs_routing(config, 'SUMMA')
