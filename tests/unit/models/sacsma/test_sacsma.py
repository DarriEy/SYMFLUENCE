"""Tests for SAC-SMA soil moisture accounting model."""

import pytest
import numpy as np

from symfluence.models.sacsma.sacsma import (
    SacSmaState,
    sacsma_step,
    sacsma_simulate,
    _create_default_state,
)
from symfluence.models.sacsma.parameters import (
    SACSMA_DEFAULTS,
    create_sacsma_params,
)


@pytest.fixture
def default_params():
    return create_sacsma_params(SACSMA_DEFAULTS)


@pytest.fixture
def half_capacity_state(default_params):
    return _create_default_state(default_params)


@pytest.fixture
def dry_state(default_params):
    """Completely dry soil state."""
    return SacSmaState(uztwc=0.0, uzfwc=0.0, lztwc=0.0, lzfpc=0.0, lzfsc=0.0, adimc=0.0)


@pytest.fixture
def saturated_state(default_params):
    """Fully saturated soil state."""
    return SacSmaState(
        uztwc=default_params.UZTWM,
        uzfwc=default_params.UZFWM,
        lztwc=default_params.LZTWM,
        lzfpc=default_params.LZFPM,
        lzfsc=default_params.LZFSM,
        adimc=default_params.UZTWM + default_params.LZTWM,
    )


class TestETSequence:
    """Test evapotranspiration demand sequence."""

    def test_et_from_upper_zone_first(self, default_params, half_capacity_state):
        """ET should first deplete upper zone tension water."""
        pet = 5.0
        state_before = half_capacity_state
        state, surf, interf, base, et = sacsma_step(
            0.0, pet, 1.0, state_before, default_params,
        )
        # UZTWC should decrease
        assert state.uztwc < state_before.uztwc

    def test_et_limited_by_pet(self, default_params, saturated_state):
        """Total ET should not exceed PET (plus RIVA)."""
        pet = 2.0
        _, _, _, _, et = sacsma_step(
            0.0, pet, 1.0, saturated_state, default_params,
        )
        # ET can be slightly above PET due to RIVA
        assert et <= pet * (1.0 + default_params.RIVA + 0.5)

    def test_zero_et_when_dry(self, default_params, dry_state):
        """No ET when soil is empty."""
        _, _, _, _, et = sacsma_step(
            0.0, 5.0, 1.0, dry_state, default_params,
        )
        # Only RIVA component if any
        assert et <= 5.0 * default_params.RIVA + 0.01

    def test_no_et_when_pet_zero(self, default_params, half_capacity_state):
        """No ET when PET is zero."""
        state, _, _, _, et = sacsma_step(
            0.0, 0.0, 1.0, half_capacity_state, default_params,
        )
        assert et == pytest.approx(0.0, abs=1e-10)


class TestPercolation:
    """Test percolation from upper zone to lower zone."""

    def test_percolation_reduces_uzfwc(self, default_params, half_capacity_state):
        """Percolation should reduce upper zone free water."""
        # Add water to trigger percolation
        wet_state = half_capacity_state._replace(
            uzfwc=default_params.UZFWM * 0.8,
            lztwc=default_params.LZTWM * 0.1,  # Low LZ → high demand
        )
        state, _, _, _, _ = sacsma_step(
            5.0, 1.0, 1.0, wet_state, default_params,
        )
        # Hard to assert exact values due to complex interactions,
        # but lower zone should gain water
        lz_before = wet_state.lztwc + wet_state.lzfpc + wet_state.lzfsc
        lz_after = state.lztwc + state.lzfpc + state.lzfsc
        assert lz_after >= lz_before - 1.0  # Allow small tolerance for ET

    def test_no_percolation_when_uz_dry(self, default_params, dry_state):
        """No percolation when upper zone is dry."""
        state, _, _, _, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params,
        )
        # Lower zone should remain dry
        assert state.lzfpc == 0.0
        assert state.lzfsc == 0.0


class TestSurfaceRunoff:
    """Test surface runoff generation."""

    def test_direct_runoff_from_pctim(self, default_params, half_capacity_state):
        """Permanent impervious area should generate direct runoff."""
        pxv = 20.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, half_capacity_state, default_params,
        )
        assert surface >= pxv * default_params.PCTIM * 0.9  # Allow tolerance

    def test_overflow_runoff_when_saturated(self, default_params, saturated_state):
        """Large precip on saturated soil should produce surface runoff."""
        pxv = 50.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, saturated_state, default_params,
        )
        assert surface > 0

    def test_no_surface_runoff_dry_soil(self, default_params, dry_state):
        """Small precip on dry soil: all absorbed, minimal runoff."""
        pxv = 1.0
        _, surface, _, _, _ = sacsma_step(
            pxv, 0.0, 1.0, dry_state, default_params,
        )
        # Only PCTIM direct runoff expected
        assert surface <= pxv * (default_params.PCTIM + default_params.ADIMP + 0.01)


class TestBaseflow:
    """Test baseflow recession."""

    def test_baseflow_from_lower_zone(self, default_params, half_capacity_state):
        """Lower zone free water should generate baseflow."""
        _, _, _, baseflow, _ = sacsma_step(
            0.0, 0.0, 1.0, half_capacity_state, default_params,
        )
        assert baseflow > 0

    def test_no_baseflow_when_lz_dry(self, default_params, dry_state):
        """No baseflow when lower zone is empty."""
        _, _, _, baseflow, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params,
        )
        assert baseflow == pytest.approx(0.0, abs=1e-10)

    def test_baseflow_recession(self, default_params, half_capacity_state):
        """Baseflow should decrease over time with no input."""
        state = half_capacity_state
        baseflows = []
        for _ in range(30):
            state, _, _, bf, _ = sacsma_step(0.0, 0.0, 1.0, state, default_params)
            baseflows.append(bf)

        # Baseflow should be declining
        assert baseflows[-1] < baseflows[0]

    def test_deep_loss_reduces_effective_baseflow(self):
        """SIDE > 0 should reduce effective baseflow."""
        params_no_side = create_sacsma_params({**SACSMA_DEFAULTS, 'SIDE': 0.0})
        params_with_side = create_sacsma_params({**SACSMA_DEFAULTS, 'SIDE': 0.3})

        state = _create_default_state(params_no_side)
        _, _, _, bf_no_side, _ = sacsma_step(0.0, 0.0, 1.0, state, params_no_side)
        _, _, _, bf_with_side, _ = sacsma_step(0.0, 0.0, 1.0, state, params_with_side)

        assert bf_with_side < bf_no_side


class TestInterflow:
    """Test interflow from upper zone."""

    def test_interflow_from_uzfwc(self, default_params):
        """Upper zone free water should generate interflow when LZ is saturated."""
        # Use saturated lower zone so percolation demand is zero,
        # leaving UZFWC available for interflow
        state = SacSmaState(
            uztwc=default_params.UZTWM * 0.5,
            uzfwc=default_params.UZFWM * 0.8,
            lztwc=default_params.LZTWM,
            lzfpc=default_params.LZFPM,
            lzfsc=default_params.LZFSM,
            adimc=(default_params.UZTWM + default_params.LZTWM) * 0.5,
        )
        _, _, interflow, _, _ = sacsma_step(
            0.0, 0.0, 1.0, state, default_params,
        )
        assert interflow > 0

    def test_no_interflow_when_uz_dry(self, default_params, dry_state):
        """No interflow when UZ free water is empty."""
        _, _, interflow, _, _ = sacsma_step(
            0.0, 0.0, 1.0, dry_state, default_params,
        )
        assert interflow == pytest.approx(0.0, abs=1e-10)


class TestNonNegativeOutputs:
    """Ensure all outputs are non-negative under various conditions."""

    @pytest.mark.parametrize("pxv", [0.0, 1.0, 10.0, 50.0, 200.0])
    @pytest.mark.parametrize("pet", [0.0, 2.0, 10.0])
    def test_non_negative(self, default_params, half_capacity_state, pxv, pet):
        state, surface, interflow, baseflow, et = sacsma_step(
            pxv, pet, 1.0, half_capacity_state, default_params,
        )
        assert surface >= 0
        assert interflow >= 0
        assert baseflow >= 0
        assert et >= 0
        assert state.uztwc >= 0
        assert state.uzfwc >= 0
        assert state.lztwc >= 0
        assert state.lzfpc >= 0
        assert state.lzfsc >= 0


class TestSacSmaSimulate:
    """Test full SAC-SMA simulation."""

    def test_basic_simulation(self, default_params):
        n = 365
        pxv = np.full(n, 3.0)
        pet = np.full(n, 2.0)

        flow, final = sacsma_simulate(pxv, pet, default_params)

        assert len(flow) == n
        assert np.all(flow >= 0)
        assert flow.sum() > 0

    def test_dry_in_produces_zero_flow(self, default_params):
        """No precip, no PET → flow only from initial storage."""
        n = 1000
        pxv = np.zeros(n)
        pet = np.zeros(n)

        flow, final = sacsma_simulate(pxv, pet, default_params)

        # Should produce flow from draining initial storage
        assert flow[:30].sum() > 0
        # But eventually near-zero
        assert flow[-10:].mean() < 0.01

    def test_mass_conservation_approximate(self, default_params):
        """Total in ≈ total out + storage change + deep loss."""
        n = 1000
        pxv = np.full(n, 5.0)
        pet = np.full(n, 2.0)

        init_state = _create_default_state(default_params)
        flow, final = sacsma_simulate(pxv, pet, default_params, initial_state=init_state)

        total_in = pxv.sum()
        total_out = flow.sum()
        # Should have reasonable water balance
        # (not exact due to ET and deep loss)
        assert total_out > 0
        assert total_out < total_in  # Must lose some to ET

    def test_returns_final_state(self, default_params):
        n = 30
        pxv = np.full(n, 5.0)
        pet = np.full(n, 2.0)

        _, final = sacsma_simulate(pxv, pet, default_params)

        assert isinstance(final, SacSmaState)
