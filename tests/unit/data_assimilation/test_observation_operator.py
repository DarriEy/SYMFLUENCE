"""Tests for observation operators."""

import numpy as np
import pytest

from symfluence.data_assimilation.enkf.observation_operator import (
    StreamflowObservationOperator,
    SWEObservationOperator,
)


class TestStreamflowObservationOperator:
    """Tests for StreamflowObservationOperator."""

    def test_single_obs(self):
        op = StreamflowObservationOperator(n_obs=1)
        state = np.array([10.0, 150.0, 5.0, 3.0, 42.0])

        y = op.apply(state)
        assert y.shape == (1,)
        assert y[0] == 42.0

    def test_matrix_form(self):
        op = StreamflowObservationOperator(n_obs=1)
        H = op.get_matrix(n_state=5)
        assert H.shape == (1, 5)
        assert H[0, 4] == 1.0
        assert np.sum(H) == 1.0

    def test_consistency(self):
        """H @ x should equal apply(x)."""
        op = StreamflowObservationOperator(n_obs=1)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = op.get_matrix(5)

        np.testing.assert_allclose(H @ state, op.apply(state))


class TestSWEObservationOperator:
    """Tests for SWEObservationOperator."""

    def test_uniform_weights(self):
        indices = np.array([0, 1, 2])
        op = SWEObservationOperator(indices)

        state = np.array([10.0, 20.0, 30.0, 99.0])
        y = op.apply(state)
        assert y.shape == (1,)
        np.testing.assert_allclose(y[0], 20.0)  # mean of 10, 20, 30

    def test_custom_weights(self):
        indices = np.array([0, 2])
        weights = np.array([0.3, 0.7])
        op = SWEObservationOperator(indices, weights)

        state = np.array([10.0, 99.0, 30.0])
        y = op.apply(state)
        expected = 0.3 * 10.0 + 0.7 * 30.0
        np.testing.assert_allclose(y[0], expected, rtol=1e-10)

    def test_matrix_consistency(self):
        indices = np.array([1, 3])
        weights = np.array([0.5, 0.5])
        op = SWEObservationOperator(indices, weights)

        state = np.random.randn(5)
        H = op.get_matrix(5)
        np.testing.assert_allclose(H @ state, op.apply(state))
