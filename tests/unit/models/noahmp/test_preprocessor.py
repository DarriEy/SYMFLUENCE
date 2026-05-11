"""Tests for Noah-MP preprocessor."""
import numpy as np
import pytest


class TestNoahMPPreProcessor:
    def test_preprocessor_can_be_imported(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor is not None

    def test_preprocessor_registered(self):
        import symfluence.models.noahmp  # noqa: F401
        from symfluence.models.registry import ModelRegistry
        assert 'NOAHMP' in ModelRegistry._preprocessors

    def test_model_name(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor.MODEL_NAME == "NOAHMP"


class TestSpechumToRh:
    def test_zero_humidity(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor._spechum_to_rh(np.array([0.0]), np.array([273.15]), np.array([101325.0]))[0] == pytest.approx(0.0, abs=0.01)

    def test_saturated(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        t = np.array([293.15]); p = np.array([101325.0])
        es = 611.2 * np.exp(17.67 * 20.0 / (20.0 + 243.5))
        qs = 0.622 * es / (p[0] - 0.378 * es)
        assert NoahMPPreProcessor._spechum_to_rh(np.array([qs]), t, p)[0] == pytest.approx(100.0, abs=1.0)

    def test_clipped_to_100(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor._spechum_to_rh(np.array([1.0]), np.array([273.15]), np.array([101325.0]))[0] <= 100.0

    def test_clipped_to_0(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor._spechum_to_rh(np.array([-0.001]), np.array([273.15]), np.array([101325.0]))[0] >= 0.0

    def test_typical_winter(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        rh = NoahMPPreProcessor._spechum_to_rh(np.array([0.001]), np.array([263.15]), np.array([101325.0]))[0]
        assert 20.0 < rh < 100.0

    def test_vectorized(self):
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        rh = NoahMPPreProcessor._spechum_to_rh(np.full(100, 0.005), np.full(100, 280.0), np.full(100, 101325.0))
        assert rh.shape == (100,)
        assert np.all((rh >= 0) & (rh <= 100))
