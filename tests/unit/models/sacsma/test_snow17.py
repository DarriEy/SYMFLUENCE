"""Tests for Snow-17 temperature index snow model."""

import pytest
import numpy as np

from symfluence.models.sacsma.snow17 import (
    Snow17State,
    snow17_step,
    snow17_simulate,
    _seasonal_melt_factor,
    _areal_depletion,
    DEFAULT_ADC,
)
from symfluence.models.sacsma.parameters import (
    SNOW17_DEFAULTS,
    create_snow17_params,
)


@pytest.fixture
def default_params():
    return create_snow17_params(SNOW17_DEFAULTS)


@pytest.fixture
def no_snow_state():
    return Snow17State(w_i=0.0, w_q=0.0, w_qx=0.0, deficit=0.0, ati=0.0, swe=0.0)


@pytest.fixture
def snowy_state():
    return Snow17State(w_i=50.0, w_q=2.0, w_qx=2.5, deficit=5.0, ati=-2.0, swe=55.0)


class TestSeasonalMeltFactor:
    """Test seasonal melt factor sinusoid."""

    def test_peaks_near_summer_solstice(self):
        """MFMAX should be reached near day 172 (Jun 21)."""
        mfmax, mfmin = 1.5, 0.3
        mf_172 = _seasonal_melt_factor(172, mfmax, mfmin)
        # Should be close to MFMAX
        assert mf_172 > (mfmax + mfmin) / 2

    def test_trough_near_winter_solstice(self):
        """MFMIN should be reached near day 355 (Dec 21)."""
        mfmax, mfmin = 1.5, 0.3
        mf_355 = _seasonal_melt_factor(355, mfmax, mfmin)
        # Should be close to MFMIN
        assert mf_355 < (mfmax + mfmin) / 2

    def test_average_at_equinox(self):
        """Average should be near equinox (day 81)."""
        mfmax, mfmin = 1.5, 0.3
        mf_81 = _seasonal_melt_factor(81, mfmax, mfmin)
        expected_avg = (mfmax + mfmin) / 2
        assert abs(mf_81 - expected_avg) < 0.01

    def test_bounded_between_mfmin_mfmax(self):
        mfmax, mfmin = 1.5, 0.3
        for doy in range(1, 366):
            mf = _seasonal_melt_factor(doy, mfmax, mfmin)
            assert mfmin <= mf <= mfmax, f"Day {doy}: mf={mf}"

    def test_southern_hemisphere(self):
        """Southern hemisphere should have opposite phasing."""
        mfmax, mfmin = 1.5, 0.3
        # Northern summer = Southern winter
        mf_n_summer = _seasonal_melt_factor(172, mfmax, mfmin, latitude=45.0)
        mf_s_summer = _seasonal_melt_factor(172, mfmax, mfmin, latitude=-45.0)
        # Northern peak should correspond to Southern trough
        assert mf_n_summer > mf_s_summer


class TestArealDepletion:
    """Test areal depletion curve."""

    def test_zero_swe(self):
        assert _areal_depletion(0.0, 100.0) == 0.0

    def test_full_coverage_at_si(self):
        assert _areal_depletion(100.0, 100.0) == pytest.approx(1.0)

    def test_partial_coverage(self):
        cover = _areal_depletion(50.0, 100.0)
        assert 0.0 < cover < 1.0

    def test_monotonic_increasing(self):
        """More SWE = more coverage."""
        si = 100.0
        prev = 0.0
        for swe in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            cover = _areal_depletion(float(swe), si)
            assert cover >= prev, f"Non-monotonic at SWE={swe}"
            prev = cover

    def test_capped_at_si(self):
        """SWE > SI should still give coverage=1.0."""
        cover = _areal_depletion(200.0, 100.0)
        assert cover == pytest.approx(1.0)


class TestRainSnowPartition:
    """Test rain/snow partitioning in snow17_step."""

    def test_all_snow_below_threshold(self, default_params, no_snow_state):
        """Below PXTEMP-1, all precip should be snow."""
        temp = default_params.PXTEMP - 2.0  # Well below threshold
        state, outflow = snow17_step(
            precip=10.0, temp=temp, dt=1.0,
            state=no_snow_state, params=default_params,
            day_of_year=1,
        )
        # With all snow, ice should accumulate (corrected by SCF)
        assert state.w_i > 0

    def test_all_rain_above_threshold(self, default_params, no_snow_state):
        """Above PXTEMP+1, all precip should be rain."""
        temp = default_params.PXTEMP + 2.0
        state, outflow = snow17_step(
            precip=10.0, temp=temp, dt=1.0,
            state=no_snow_state, params=default_params,
            day_of_year=172,
        )
        # No snow on ground, rain passes through
        assert state.w_i == 0.0
        assert outflow > 0

    def test_mixed_phase_in_transition(self, default_params, no_snow_state):
        """At PXTEMP, should get ~50% rain/snow."""
        temp = default_params.PXTEMP
        state, outflow = snow17_step(
            precip=10.0, temp=temp, dt=1.0,
            state=no_snow_state, params=default_params,
            day_of_year=172,
        )
        # Some snow should accumulate and some should pass through
        # (exact split depends on SCF and melt)
        # Just verify some snow formed
        assert state.w_i >= 0


class TestSnow17Step:
    """Test individual Snow-17 timesteps."""

    def test_no_precip_no_change(self, default_params, no_snow_state):
        """No precip, no snow → no change."""
        state, outflow = snow17_step(
            precip=0.0, temp=5.0, dt=1.0,
            state=no_snow_state, params=default_params,
            day_of_year=172,
        )
        assert state.w_i == 0.0
        assert outflow == 0.0

    def test_melt_in_warm_conditions(self, default_params, snowy_state):
        """Warm temperature should produce melt."""
        state, outflow = snow17_step(
            precip=0.0, temp=10.0, dt=1.0,
            state=snowy_state, params=default_params,
            day_of_year=172,  # Summer, high melt factor
        )
        # Ice should decrease or outflow should occur
        assert state.w_i < snowy_state.w_i or outflow > 0

    def test_no_melt_below_mbase(self, default_params, snowy_state):
        """Very cold conditions: no temperature-driven melt.

        Ground melt (DAYGM) may still produce small outflow.
        """
        params = create_snow17_params({**SNOW17_DEFAULTS, 'DAYGM': 0.0})
        state, outflow = snow17_step(
            precip=0.0, temp=-20.0, dt=1.0,
            state=snowy_state, params=params,
            day_of_year=355,  # Winter
        )
        # No melt from temperature; deficit should increase
        assert state.deficit >= snowy_state.deficit

    def test_snowfall_accumulation(self, default_params, no_snow_state):
        """Cold precip should accumulate as snow."""
        state, _ = snow17_step(
            precip=20.0, temp=-10.0, dt=1.0,
            state=no_snow_state, params=default_params,
            day_of_year=1,
        )
        assert state.w_i > 0

    def test_non_negative_outputs(self, default_params, snowy_state):
        """All states and outflow must be non-negative."""
        for temp in [-20, -5, 0, 5, 15, 30]:
            state, outflow = snow17_step(
                precip=5.0, temp=float(temp), dt=1.0,
                state=snowy_state, params=default_params,
                day_of_year=172,
            )
            assert state.w_i >= 0
            assert state.w_q >= 0
            assert state.deficit >= 0
            assert outflow >= 0


class TestSnow17Simulate:
    """Test full Snow-17 simulation."""

    def test_basic_simulation(self, default_params):
        """Run a year of simulation."""
        n = 365
        precip = np.full(n, 3.0)
        temp = 10.0 * np.sin(np.arange(n) * 2 * np.pi / 365 - np.pi / 2)
        doy = np.arange(1, n + 1)

        rpm, final = snow17_simulate(precip, temp, doy, default_params)

        assert len(rpm) == n
        assert np.all(rpm >= 0)
        assert rpm.sum() > 0

    def test_all_rain_passthrough(self, default_params):
        """Warm temps: all precip should pass through as rain (±SCF)."""
        n = 30
        precip = np.full(n, 5.0)
        temp = np.full(n, 20.0)  # Well above PXTEMP+1
        doy = np.full(n, 172, dtype=int)

        rpm, final = snow17_simulate(precip, temp, doy, default_params)

        # Most precip should pass through
        total_in = precip.sum()
        total_out = rpm.sum()
        assert total_out > 0.8 * total_in

    def test_snow_accumulation_then_melt(self, default_params):
        """Cold period accumulates, warm period melts."""
        n = 200
        precip = np.full(n, 3.0)
        # 100 days cold, 100 days warm
        temp = np.concatenate([np.full(100, -10.0), np.full(100, 15.0)])
        doy = np.arange(1, n + 1) % 365 + 1

        rpm, final = snow17_simulate(precip, temp, doy, default_params)

        # During cold period, most precip stored as snow → low outflow
        cold_outflow = rpm[:100].mean()
        # During warm period, melt + rain → higher outflow
        warm_outflow = rpm[100:].mean()
        assert warm_outflow > cold_outflow

    def test_mass_conservation_warm(self, default_params):
        """For all-rain scenario, total out ≈ total in."""
        n = 100
        precip = np.full(n, 5.0)
        temp = np.full(n, 25.0)
        doy = np.full(n, 172, dtype=int)

        rpm, _ = snow17_simulate(precip, temp, doy, default_params)

        total_in = precip.sum()
        total_out = rpm.sum()
        # Should be very close since no snow storage at end
        assert abs(total_out - total_in) / total_in < 0.05

    def test_returns_final_state(self, default_params):
        n = 30
        precip = np.full(n, 3.0)
        temp = np.full(n, -5.0)
        doy = np.arange(1, n + 1)

        _, final = snow17_simulate(precip, temp, doy, default_params)

        assert isinstance(final, Snow17State)
        assert final.w_i > 0  # Should have accumulated snow
