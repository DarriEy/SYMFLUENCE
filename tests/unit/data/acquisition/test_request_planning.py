# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from symfluence.data.acquisition.request_planning import expected_forcing_times, forcing_request_facts


def test_expected_forcing_times_uses_dataset_resolution() -> None:
    actual = expected_forcing_times("cerra", "2020-01-01 00:00", "2020-01-01 06:00")
    assert actual is not None
    assert actual.equals(pd.date_range("2020-01-01", periods=3, freq="3h"))


def test_expected_forcing_times_skips_unknown_or_invalid_windows() -> None:
    assert expected_forcing_times("era5", "2020-01-01", "2020-01-02") is None
    assert expected_forcing_times("carra", "2020-01-02", "2020-01-01") is None
    assert expected_forcing_times("carra", "not-a-date", "2020-01-02") is None
    assert expected_forcing_times("carra", None, "2020-01-02") is None


def test_dataset_variables_override_defaults() -> None:
    window, variables = forcing_request_facts(
        "2020-01-01", "2020-01-02", ["temperature"], ["precipitation"]
    )
    assert window == ("2020-01-01", "2020-01-02")
    assert variables == frozenset({"temperature"})
