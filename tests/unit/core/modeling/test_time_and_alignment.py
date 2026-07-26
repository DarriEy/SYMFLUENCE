# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavior checks for model time-window and dataset-alignment contracts."""
from __future__ import annotations

import pandas as pd
import xarray as xr

from symfluence.core.modeling.utilities import DatasetAlignmentManager, TimeWindowManager


def test_time_window_parses_supported_formats_and_validates_order():
    manager = TimeWindowManager({
        "SIMULATION_START_DATE": "2020/01/01",
        "SIMULATION_END_DATE": "2020-01-03 12:00",
    })

    start, end = manager.get_simulation_times()

    assert start == pd.Timestamp("2020-01-01")
    assert end == pd.Timestamp("2020-01-03 12:00")


def test_alignment_manager_finds_dataset_intersection():
    first = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=4)})
    second = xr.Dataset(coords={"time": pd.date_range("2020-01-03", periods=4)})

    start, end = DatasetAlignmentManager().find_common_time_period([first, second])

    assert start == pd.Timestamp("2020-01-03")
    assert end == pd.Timestamp("2020-01-04")
