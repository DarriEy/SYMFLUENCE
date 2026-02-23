"""Tests for CouplingGraphBuilder."""

import pytest
import torch

from symfluence.coupling.bmi_registry import BMIRegistry
from symfluence.coupling.graph_builder import UNIT_CONVERSIONS, CouplingGraphBuilder


class TestBMIRegistry:
    def test_known_process_models(self):
        registry = BMIRegistry()
        for name in ["SUMMA", "MIZUROUTE", "PARFLOW", "MODFLOW", "MESH", "CLM"]:
            assert registry.is_process_model(name)
            assert not registry.is_jax_model(name)

    def test_known_jax_models(self):
        registry = BMIRegistry()
        for name in ["SNOW17", "XAJ", "XINANJIANG", "SACSMA"]:
            assert registry.is_jax_model(name)
            assert not registry.is_process_model(name)

    def test_unknown_model_raises(self):
        registry = BMIRegistry()
        with pytest.raises(KeyError, match="Unknown model"):
            registry.get("NONEXISTENT")

    def test_register_custom(self):
        registry = BMIRegistry()
        registry.register("MYMODEL", "some.module.MyClass")
        assert "MYMODEL" in registry.available_models()

    def test_available_models(self):
        registry = BMIRegistry()
        models = registry.available_models()
        assert len(models) > 0
        assert "SUMMA" in models
        assert "SNOW17" in models


class TestUnitConversions:
    def test_summa_to_parflow(self):
        assert UNIT_CONVERSIONS[("SUMMA", "PARFLOW")] == 3.6

    def test_summa_to_modflow(self):
        assert UNIT_CONVERSIONS[("SUMMA", "MODFLOW")] == 86.4

    def test_snow17_to_xaj(self):
        assert UNIT_CONVERSIONS[("SNOW17", "XAJ")] == 1.0


class TestCouplingGraphBuilder:
    def test_build_requires_model(self):
        builder = CouplingGraphBuilder()
        with pytest.raises(ValueError, match="HYDROLOGICAL_MODEL"):
            builder.build({})

    def test_build_summa_only(self):
        builder = CouplingGraphBuilder()
        config = {"HYDROLOGICAL_MODEL": "SUMMA"}
        graph = builder.build(config)
        assert "land" in graph.components
        assert len(graph.components) == 1

    def test_build_summa_with_routing(self):
        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "ROUTING_MODEL": "MIZUROUTE",
        }
        graph = builder.build(config)
        assert "land" in graph.components
        assert "routing" in graph.components
        assert len(graph.connections) == 1

    def test_build_summa_with_groundwater(self):
        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "GROUNDWATER_MODEL": "PARFLOW",
        }
        graph = builder.build(config)
        assert "land" in graph.components
        assert "groundwater" in graph.components
        assert len(graph.connections) == 1
        # Check unit conversion was set
        conn = graph.connections[0]
        assert conn.unit_conversion == 3.6

    def test_build_with_conservation(self):
        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "CONSERVATION_MODE": "check",
        }
        graph = builder.build(config)
        assert graph._conservation is not None

    def test_build_snow17_xaj_coupled(self):
        """Test building a Snow-17 + XAJ coupled graph."""
        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "XAJ",
            "SNOW_MODULE": "SNOW17",
        }
        try:
            graph = builder.build(config)
            assert "land" in graph.components
            assert "snow" in graph.components
            assert len(graph.connections) == 1
        except ImportError:
            pytest.skip("JAX or SYMFLUENCE models not available")
