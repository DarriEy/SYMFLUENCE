"""
Tests for Model Forcing Adapters.

Verifies that model-specific forcing adapters are properly registered
and provide expected interfaces for converting CFIF (CF-Intermediate Format)
data to model-specific formats.
"""

import pytest

from symfluence.models.adapters import ForcingAdapter, ForcingAdapterRegistry
from symfluence.models.registry import ModelRegistry


class TestForcingAdapterRegistration:
    """Test that forcing adapters are properly registered."""

    def test_summa_adapter_registered(self):
        """SUMMA forcing adapter should be registered."""
        assert ModelRegistry.has_forcing_adapter('SUMMA') is True
        assert ForcingAdapterRegistry.is_registered('SUMMA') is True

    def test_fuse_adapter_registered(self):
        """FUSE forcing adapter should be registered."""
        assert ModelRegistry.has_forcing_adapter('FUSE') is True
        assert ForcingAdapterRegistry.is_registered('FUSE') is True

    def test_hype_adapter_registered(self):
        """HYPE forcing adapter should be registered."""
        assert ModelRegistry.has_forcing_adapter('HYPE') is True
        assert ForcingAdapterRegistry.is_registered('HYPE') is True

    def test_ngen_adapter_registered(self):
        """NGEN forcing adapter should be registered."""
        assert ModelRegistry.has_forcing_adapter('NGEN') is True
        assert ForcingAdapterRegistry.is_registered('NGEN') is True

    def test_gr_adapter_registered(self):
        """GR forcing adapter should be registered."""
        assert ModelRegistry.has_forcing_adapter('GR') is True
        assert ForcingAdapterRegistry.is_registered('GR') is True

    def test_list_forcing_adapters(self):
        """Should be able to list all registered forcing adapters."""
        adapters = ModelRegistry.list_forcing_adapters()
        assert isinstance(adapters, list)
        assert 'SUMMA' in adapters
        assert 'FUSE' in adapters
        assert 'HYPE' in adapters
        assert 'NGEN' in adapters
        assert 'GR' in adapters

    def test_list_via_registry(self):
        """Should get same list from both registries."""
        from_model_registry = ModelRegistry.list_forcing_adapters()
        from_forcing_registry = ForcingAdapterRegistry.get_registered_models()
        assert from_model_registry == from_forcing_registry

    def test_nonexistent_adapter(self):
        """Should return False for non-existent adapter."""
        assert ModelRegistry.has_forcing_adapter('NONEXISTENT') is False
        assert ForcingAdapterRegistry.is_registered('NONEXISTENT') is False


class TestForcingAdapterInstantiation:
    """Test that forcing adapters can be instantiated."""

    @pytest.fixture
    def config(self):
        """Minimal config for adapter instantiation."""
        return {'domain': {'name': 'test'}}

    def test_get_summa_adapter(self, config):
        """Should be able to get SUMMA adapter instance."""
        adapter = ModelRegistry.get_forcing_adapter('SUMMA', config)
        assert adapter is not None
        assert isinstance(adapter, ForcingAdapter)

    def test_get_fuse_adapter(self, config):
        """Should be able to get FUSE adapter instance."""
        adapter = ModelRegistry.get_forcing_adapter('FUSE', config)
        assert adapter is not None
        assert isinstance(adapter, ForcingAdapter)

    def test_get_hype_adapter(self, config):
        """Should be able to get HYPE adapter instance."""
        adapter = ModelRegistry.get_forcing_adapter('HYPE', config)
        assert adapter is not None
        assert isinstance(adapter, ForcingAdapter)

    def test_get_ngen_adapter(self, config):
        """Should be able to get NGEN adapter instance."""
        adapter = ModelRegistry.get_forcing_adapter('NGEN', config)
        assert adapter is not None
        assert isinstance(adapter, ForcingAdapter)

    def test_get_gr_adapter(self, config):
        """Should be able to get GR adapter instance."""
        adapter = ModelRegistry.get_forcing_adapter('GR', config)
        assert adapter is not None
        assert isinstance(adapter, ForcingAdapter)

    def test_get_nonexistent_adapter(self, config):
        """Should return None for non-existent adapter via ModelRegistry."""
        adapter = ModelRegistry.get_forcing_adapter('NONEXISTENT', config)
        assert adapter is None

    def test_get_nonexistent_adapter_via_forcing_registry(self, config):
        """Should raise ValueError for non-existent adapter via ForcingAdapterRegistry."""
        with pytest.raises(ValueError, match="No forcing adapter registered"):
            ForcingAdapterRegistry.get_adapter('NONEXISTENT', config)


class TestSUMMAForcingAdapter:
    """Test SUMMA forcing adapter interface."""

    @pytest.fixture
    def adapter(self):
        """Get SUMMA adapter instance."""
        config = {'domain': {'name': 'test'}}
        return ModelRegistry.get_forcing_adapter('SUMMA', config)

    def test_has_variable_mapping(self, adapter):
        """SUMMA adapter should provide variable mapping."""
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)
        assert 'air_temperature' in mapping
        assert mapping['air_temperature'] == 'airtemp'
        assert 'precipitation_flux' in mapping
        assert mapping['precipitation_flux'] == 'pptrate'

    def test_has_required_variables(self, adapter):
        """SUMMA adapter should define required variables."""
        required = adapter.get_required_variables()
        assert isinstance(required, list)
        assert 'air_temperature' in required
        assert 'precipitation_flux' in required
        assert 'surface_downwelling_shortwave_flux' in required

    def test_has_optional_variables(self, adapter):
        """SUMMA adapter should define optional variables."""
        optional = adapter.get_optional_variables()
        assert isinstance(optional, list)
        # May be empty or have variables

    def test_has_transform_method(self, adapter):
        """SUMMA adapter should have transform method."""
        assert hasattr(adapter, 'transform')
        assert callable(adapter.transform)


class TestFUSEForcingAdapter:
    """Test FUSE forcing adapter interface."""

    @pytest.fixture
    def adapter(self):
        """Get FUSE adapter instance."""
        config = {'domain': {'name': 'test'}}
        return ModelRegistry.get_forcing_adapter('FUSE', config)

    def test_has_variable_mapping(self, adapter):
        """FUSE adapter should provide variable mapping."""
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_has_required_variables(self, adapter):
        """FUSE adapter should define required variables."""
        required = adapter.get_required_variables()
        assert isinstance(required, list)
        assert len(required) > 0


class TestHYPEForcingAdapter:
    """Test HYPE forcing adapter interface."""

    @pytest.fixture
    def adapter(self):
        """Get HYPE adapter instance."""
        config = {'domain': {'name': 'test'}}
        return ModelRegistry.get_forcing_adapter('HYPE', config)

    def test_has_variable_mapping(self, adapter):
        """HYPE adapter should provide variable mapping."""
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_has_required_variables(self, adapter):
        """HYPE adapter should define required variables."""
        required = adapter.get_required_variables()
        assert isinstance(required, list)
        assert len(required) > 0


class TestNGENForcingAdapter:
    """Test NGEN forcing adapter interface."""

    @pytest.fixture
    def adapter(self):
        """Get NGEN adapter instance."""
        config = {'domain': {'name': 'test'}}
        return ModelRegistry.get_forcing_adapter('NGEN', config)

    def test_has_variable_mapping(self, adapter):
        """NGEN adapter should provide variable mapping."""
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_has_required_variables(self, adapter):
        """NGEN adapter should define required variables."""
        required = adapter.get_required_variables()
        assert isinstance(required, list)
        assert len(required) > 0


class TestGRForcingAdapter:
    """Test GR forcing adapter interface."""

    @pytest.fixture
    def adapter(self):
        """Get GR adapter instance."""
        config = {'domain': {'name': 'test'}}
        return ModelRegistry.get_forcing_adapter('GR', config)

    def test_has_variable_mapping(self, adapter):
        """GR adapter should provide variable mapping."""
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_has_required_variables(self, adapter):
        """GR adapter should define required variables."""
        required = adapter.get_required_variables()
        assert isinstance(required, list)
        assert len(required) > 0


class TestForcingAdapterInterface:
    """Test that all adapters implement required interface."""

    @pytest.fixture
    def config(self):
        """Minimal config for adapter instantiation."""
        return {'domain': {'name': 'test'}}

    @pytest.mark.parametrize('model_name', ['SUMMA', 'FUSE', 'HYPE', 'NGEN', 'GR'])
    def test_adapter_has_required_methods(self, model_name, config):
        """All adapters should implement required interface methods."""
        adapter = ModelRegistry.get_forcing_adapter(model_name, config)

        assert hasattr(adapter, 'get_variable_mapping')
        assert hasattr(adapter, 'get_required_variables')
        assert hasattr(adapter, 'get_optional_variables')
        assert hasattr(adapter, 'get_unit_conversions')
        assert hasattr(adapter, 'transform')
        assert hasattr(adapter, 'rename_variables')
        assert hasattr(adapter, 'apply_unit_conversions')
        assert hasattr(adapter, 'add_metadata')

    @pytest.mark.parametrize('model_name', ['SUMMA', 'FUSE', 'HYPE', 'NGEN', 'GR'])
    def test_adapter_methods_are_callable(self, model_name, config):
        """All adapter methods should be callable."""
        adapter = ModelRegistry.get_forcing_adapter(model_name, config)

        assert callable(adapter.get_variable_mapping)
        assert callable(adapter.get_required_variables)
        assert callable(adapter.get_optional_variables)
        assert callable(adapter.get_unit_conversions)
        assert callable(adapter.transform)

    @pytest.mark.parametrize('model_name', ['SUMMA', 'FUSE', 'HYPE', 'NGEN', 'GR'])
    def test_adapter_returns_expected_types(self, model_name, config):
        """Adapters should return expected types."""
        adapter = ModelRegistry.get_forcing_adapter(model_name, config)

        # get_variable_mapping should return dict
        mapping = adapter.get_variable_mapping()
        assert isinstance(mapping, dict)

        # get_required_variables should return list
        required = adapter.get_required_variables()
        assert isinstance(required, list)

        # get_optional_variables should return list
        optional = adapter.get_optional_variables()
        assert isinstance(optional, list)

        # get_unit_conversions should return dict
        conversions = adapter.get_unit_conversions()
        assert isinstance(conversions, dict)


class TestForcingAdapterBackwardCompatibility:
    """Test backward compatibility with ForcingAdapterRegistry."""

    @pytest.fixture
    def config(self):
        """Minimal config for adapter instantiation."""
        return {'domain': {'name': 'test'}}

    def test_both_registries_return_same_adapter_class(self, config):
        """ModelRegistry and ForcingAdapterRegistry should work together."""
        # Get via ModelRegistry
        adapter1 = ModelRegistry.get_forcing_adapter('SUMMA', config)

        # Get via ForcingAdapterRegistry
        adapter2 = ForcingAdapterRegistry.get_adapter('SUMMA', config)

        # Should be same class
        assert type(adapter1) == type(adapter2)

    def test_both_registries_list_same_models(self):
        """Both registries should list the same models."""
        from_model_registry = set(ModelRegistry.list_forcing_adapters())
        from_forcing_registry = set(ForcingAdapterRegistry.get_registered_models())

        assert from_model_registry == from_forcing_registry
