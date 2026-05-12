"""Tests for Noah-MP preprocessor."""

import numpy as np
import pytest


class TestNoahMPPreProcessorImport:
    """Tests for Noah-MP preprocessor import and registration."""

    def test_preprocessor_can_be_imported(self):
        """Test that NoahMPPreProcessor can be imported."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor is not None

    def test_preprocessor_registered(self):
        """Test that preprocessor is registered in ModelRegistry."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor  # noqa: F401
        from symfluence.models.registry import ModelRegistry
        assert 'NOAHMP' in ModelRegistry._preprocessors

    def test_model_name(self):
        """Test MODEL_NAME class attribute."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor
        assert NoahMPPreProcessor.MODEL_NAME == 'NOAHMP'


class TestSpechum2RH:
    """Tests for specific humidity to relative humidity conversion."""

    def test_zero_humidity(self):
        """Zero specific humidity should give 0% RH."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        q = np.array([0.0])
        t_k = np.array([293.15])  # 20 degC
        p_pa = np.array([101325.0])
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert rh[0] == pytest.approx(0.0, abs=1e-10)

    def test_saturated(self):
        """Check a known near-saturation condition."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        # At 20C and 101325 Pa, saturation specific humidity ~ 0.0147 kg/kg
        # es(20C) = 611.2 * exp(17.67*20/263.5) ~ 2338 Pa
        # q_sat = 0.622 * es / (P - 0.378*es) ~ 0.622*2338/(101325 - 0.378*2338) ~ 0.01445
        q = np.array([0.01445])
        t_k = np.array([293.15])
        p_pa = np.array([101325.0])
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert 95.0 < rh[0] <= 100.0

    def test_clipped_100(self):
        """RH should be clipped at 100%."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        # Extremely high specific humidity
        q = np.array([0.05])
        t_k = np.array([293.15])
        p_pa = np.array([101325.0])
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert rh[0] == pytest.approx(100.0)

    def test_clipped_0(self):
        """RH should be clipped at 0%."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        # Negative specific humidity (physically impossible but test clipping)
        q = np.array([-0.001])
        t_k = np.array([293.15])
        p_pa = np.array([101325.0])
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert rh[0] == pytest.approx(0.0)

    def test_winter_conditions(self):
        """Winter conditions: q=0.001, T=263.15K (-10C), P=101325 Pa."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        q = np.array([0.001])
        t_k = np.array([263.15])
        p_pa = np.array([101325.0])
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert 20.0 < rh[0] < 100.0

    def test_vectorized(self):
        """Test with array of 100 values."""
        from symfluence.models.noahmp.preprocessor import NoahMPPreProcessor

        n = 100
        rng = np.random.default_rng(42)
        q = rng.uniform(0.0, 0.02, n)
        t_k = rng.uniform(250.0, 310.0, n)
        p_pa = rng.uniform(80000.0, 105000.0, n)
        rh = NoahMPPreProcessor._spechum_to_rh(q, t_k, p_pa)
        assert rh.shape == (n,)
        assert np.all(rh >= 0.0)
        assert np.all(rh <= 100.0)
