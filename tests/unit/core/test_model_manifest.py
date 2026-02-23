"""Tests for the model_manifest() declarative registration function."""

from __future__ import annotations

import pytest

from symfluence.core.registries import R, Registries
from symfluence.core.registry import model_manifest

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _clean_registries():
    """Save and restore all registries around each test."""
    saved = {}
    for name, reg in Registries.all_registries().items():
        saved[name] = (
            dict(reg._entries),
            dict(reg._meta),
            dict(reg._aliases),
        )
        reg.clear()
    yield
    for name, reg in Registries.all_registries().items():
        reg.clear()
        entries, meta, aliases = saved[name]
        reg._entries.update(entries)
        reg._meta.update(meta)
        reg._aliases.update(aliases)


class _MockPreprocessor:
    MODEL_NAME = "MOCK"

    def run_preprocessing(self):
        return True


class _MockRunner:
    MODEL_NAME = "MOCK"

    def run(self, **kw):
        return None


class _MockPostprocessor:
    MODEL_NAME = "MOCK"

    def extract_streamflow(self):
        return None


class _MockConfigAdapter:
    pass


class _MockExtractor:
    pass


class _MockDecisionAnalyzer:
    pass


class _MockSensitivityAnalyzer:
    pass


class _MockKoopmanAnalyzer:
    pass


class _MockPlotter:
    pass


class _MockOptimizer:
    pass


class _MockWorker:
    pass


class _MockParamManager:
    pass


class _MockForcingAdapter:
    pass


# ======================================================================
# Tests
# ======================================================================


class TestModelManifest:
    def test_registers_all_components(self):
        model_manifest(
            "MOCK",
            preprocessor=_MockPreprocessor,
            runner=_MockRunner,
            runner_method="run_mock",
            postprocessor=_MockPostprocessor,
            config_adapter=_MockConfigAdapter,
            result_extractor=_MockExtractor,
            decision_analyzer=_MockDecisionAnalyzer,
            sensitivity_analyzer=_MockSensitivityAnalyzer,
            koopman_analyzer=_MockKoopmanAnalyzer,
            plotter=_MockPlotter,
            optimizer=_MockOptimizer,
            worker=_MockWorker,
            parameter_manager=_MockParamManager,
            forcing_adapter=_MockForcingAdapter,
        )

        assert R.preprocessors["MOCK"] is _MockPreprocessor
        assert R.runners["MOCK"] is _MockRunner
        assert R.postprocessors["MOCK"] is _MockPostprocessor
        assert R.config_adapters["MOCK"] is _MockConfigAdapter
        assert R.result_extractors["MOCK"] is _MockExtractor
        assert R.decision_analyzers["MOCK"] is _MockDecisionAnalyzer
        assert R.sensitivity_analyzers["MOCK"] is _MockSensitivityAnalyzer
        assert R.koopman_analyzers["MOCK"] is _MockKoopmanAnalyzer
        assert R.plotters["MOCK"] is _MockPlotter
        assert R.optimizers["MOCK"] is _MockOptimizer
        assert R.workers["MOCK"] is _MockWorker
        assert R.parameter_managers["MOCK"] is _MockParamManager
        assert R.forcing_adapters["MOCK"] is _MockForcingAdapter

    def test_runner_method_in_metadata(self):
        model_manifest("MOCK", runner=_MockRunner, runner_method="run_mock")
        assert R.runners.meta("MOCK")["runner_method"] == "run_mock"

    def test_runner_method_absent_by_default(self):
        model_manifest("MOCK", runner=_MockRunner)
        assert R.runners.meta("MOCK") == {}

    def test_skips_none_values(self):
        model_manifest("MOCK", preprocessor=_MockPreprocessor)
        assert R.preprocessors.get("MOCK") is _MockPreprocessor
        assert R.runners.get("MOCK") is None
        assert R.postprocessors.get("MOCK") is None

    def test_build_instructions_lazy(self):
        model_manifest("MOCK", build_instructions_module="math.log")
        # Should resolve lazily
        import math
        assert R.build_instructions.get("MOCK") is math.log

    def test_case_insensitive(self):
        model_manifest("mock", preprocessor=_MockPreprocessor)
        assert R.preprocessors.get("MOCK") is _MockPreprocessor

    def test_multiple_models(self):
        model_manifest("MODEL_A", preprocessor=_MockPreprocessor)
        model_manifest("MODEL_B", runner=_MockRunner)

        assert R.preprocessors.get("MODEL_A") is _MockPreprocessor
        assert R.preprocessors.get("MODEL_B") is None
        assert R.runners.get("MODEL_B") is _MockRunner
        assert R.runners.get("MODEL_A") is None

    def test_for_model_after_manifest(self):
        model_manifest(
            "MOCK",
            preprocessor=_MockPreprocessor,
            runner=_MockRunner,
            plotter=_MockPlotter,
        )
        result = Registries.for_model("MOCK")
        assert "preprocessors" in result
        assert "runners" in result
        assert "plotters" in result

    def test_validate_after_manifest(self):
        model_manifest(
            "MOCK",
            preprocessor=_MockPreprocessor,
            runner=_MockRunner,
            postprocessor=_MockPostprocessor,
        )
        v = Registries.validate_model("MOCK")
        assert v["valid"] is True

    def test_config_components(self):
        defaults = {"timestep": 3600}
        transformers = {"field_a": ("nested", "path")}

        def validator(config):
            pass

        class Schema:
            pass

        model_manifest(
            "MOCK",
            config_adapter=_MockConfigAdapter,
            config_schema=Schema,
            config_defaults=defaults,
            config_transformers=transformers,
            config_validator=validator,
        )

        assert R.config_adapters["MOCK"] is _MockConfigAdapter
        assert R.config_schemas["MOCK"] is Schema
        assert R.config_defaults["MOCK"] == defaults
        assert R.config_transformers["MOCK"] == transformers
        assert R.config_validators["MOCK"] is validator
