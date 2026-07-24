# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Index-alignment contract of the core metric input preparation.

Two equal-length series covering shifted periods must be compared over their
index overlap, never element-by-element: positional pairing makes a
time-shifted simulation look perfect (RMSE 0) when the overlap RMSE is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symfluence.core.metrics.metrics_core import _clean_data, nse, rmse


def _series(values, start):
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq="D"),
    )


def test_shifted_series_are_paired_by_timestamp_not_position():
    observed = _series([1, 2, 3], "2020-01-01")   # Jan 1-3
    simulated = _series([1, 2, 3], "2020-01-02")  # Jan 2-4

    # Overlap is Jan 2-3: obs (2, 3) vs sim (1, 2) -> RMSE 1, not 0.
    assert rmse(observed, simulated) == pytest.approx(1.0)


def test_identical_index_fast_path_unchanged():
    observed = _series([1, 2, 3, 4], "2020-01-01")
    simulated = observed.copy()
    assert rmse(observed, simulated) == pytest.approx(0.0)
    assert nse(observed, simulated) == pytest.approx(1.0)


def test_plain_arrays_keep_positional_pairing():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.0, 3.0])
    assert rmse(obs, sim) == pytest.approx(0.0)


def test_nan_rows_dropped_after_alignment():
    observed = _series([1, np.nan, 3], "2020-01-01")
    simulated = _series([1, 2, 5], "2020-01-01")
    obs, sim = _clean_data(observed, simulated)
    np.testing.assert_array_equal(obs, [1.0, 3.0])
    np.testing.assert_array_equal(sim, [1.0, 5.0])


def test_duplicate_index_labels_are_rejected():
    observed = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-01"]),
    )
    simulated = _series([1, 2], "2020-01-01")
    with pytest.raises(ValueError, match="duplicate index"):
        _clean_data(observed, simulated)


def test_disjoint_periods_yield_nan_metric_not_a_number_salad():
    observed = _series([1, 2, 3], "2020-01-01")
    simulated = _series([1, 2, 3], "2021-01-01")
    assert np.isnan(rmse(observed, simulated))
