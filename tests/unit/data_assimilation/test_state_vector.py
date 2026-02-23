"""Tests for StateVector assembly/disassembly."""

import numpy as np
import pytest

from symfluence.data_assimilation.enkf.state_vector import StateVariableSpec, StateVector


class TestStateVector:
    """Tests for StateVector assemble/disassemble roundtrip."""

    def _make_specs(self):
        return [
            StateVariableSpec('snow', 1, lower_bound=0.0),
            StateVariableSpec('sm', 1, lower_bound=0.0, upper_bound=500.0),
            StateVariableSpec('routing_buffer', 5, lower_bound=0.0),
        ]

    def test_assemble_disassemble_roundtrip(self):
        specs = self._make_specs()
        sv = StateVector(specs)

        members = [
            {'snow': np.array([10.0]), 'sm': np.array([150.0]), 'routing_buffer': np.arange(5, dtype=float)},
            {'snow': np.array([20.0]), 'sm': np.array([200.0]), 'routing_buffer': np.ones(5)},
        ]

        X = sv.assemble(members)
        assert X.shape == (2, 7)  # 1 + 1 + 5

        restored = sv.disassemble(X)
        assert len(restored) == 2

        np.testing.assert_allclose(restored[0]['snow'], 10.0)
        np.testing.assert_allclose(restored[0]['sm'], 150.0)
        np.testing.assert_allclose(restored[0]['routing_buffer'], np.arange(5, dtype=float))

        np.testing.assert_allclose(restored[1]['snow'], 20.0)
        np.testing.assert_allclose(restored[1]['sm'], 200.0)

    def test_enforce_bounds(self):
        specs = self._make_specs()
        sv = StateVector(specs)

        X = np.array([[-5.0, 600.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        X_clipped = sv.enforce_bounds(X)

        # snow >= 0
        assert X_clipped[0, 0] == 0.0
        # sm <= 500
        assert X_clipped[0, 1] == 500.0
        # routing_buffer >= 0
        assert np.all(X_clipped[0, 2:] >= 0.0)

    def test_augment_and_split(self):
        specs = self._make_specs()
        sv = StateVector(specs)

        X = np.random.randn(10, 7)
        preds = np.random.randn(10, 1)

        augmented = sv.augment_with_predictions(X, preds)
        assert augmented.shape == (10, 8)

        X_back, preds_back = sv.split_augmented(augmented, 1)
        np.testing.assert_allclose(X_back, X)
        np.testing.assert_allclose(preds_back, preds)

    def test_augment_1d_predictions(self):
        specs = self._make_specs()
        sv = StateVector(specs)

        X = np.random.randn(5, 7)
        preds_1d = np.random.randn(5)

        augmented = sv.augment_with_predictions(X, preds_1d)
        assert augmented.shape == (5, 8)

    def test_n_state(self):
        specs = self._make_specs()
        sv = StateVector(specs)
        assert sv.n_state == 7
