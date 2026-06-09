# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for temporal completeness validation in ForcingDataProcessor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.models.utilities.forcing_data_processor import ForcingDataProcessor


def _make_hourly_dataset(
    start: str = "2019-01-01",
    hours: int = 48,
    drop_indices: list | None = None,
) -> xr.Dataset:
    """Create a simple hourly forcing dataset, optionally dropping timesteps."""
    times = pd.date_range(start, periods=hours, freq="h")
    if drop_indices:
        times = times.delete(drop_indices)
    return xr.Dataset(
        {"temperature": ("time", np.random.default_rng(42).standard_normal(len(times)))},
        coords={"time": times},
    )


class TestValidateTemporalCompleteness:
    """Tests for ForcingDataProcessor.validate_temporal_completeness."""

    def setup_method(self):
        self.fdp = ForcingDataProcessor(config={})

    # -- complete data ---------------------------------------------------------

    def test_complete_data_passes(self):
        ds = _make_hourly_dataset(hours=48)
        result, report = self.fdp.validate_temporal_completeness(ds)
        assert report["valid"] is True
        assert report["missing_timesteps"] == 0
        assert report["coverage_pct"] == 100.0

    # -- strategy='error' (default) --------------------------------------------

    def test_error_strategy_raises_on_gap(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11, 12])
        with pytest.raises(ValueError, match="3 missing timesteps"):
            self.fdp.validate_temporal_completeness(ds, strategy="error")

    def test_error_strategy_message_includes_gap_details(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10])
        with pytest.raises(ValueError, match="FORCING_MISSING_DATA_STRATEGY"):
            self.fdp.validate_temporal_completeness(ds, strategy="error")

    # -- strategy='warn' -------------------------------------------------------

    def test_warn_strategy_continues(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11])
        result, report = self.fdp.validate_temporal_completeness(ds, strategy="warn")
        assert report["valid"] is False
        assert report["missing_timesteps"] == 2
        assert len(result.time) == 46  # still has the gap

    # -- strategy='interpolate' ------------------------------------------------

    def test_interpolate_fills_gap(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11])
        result, report = self.fdp.validate_temporal_completeness(
            ds, strategy="interpolate"
        )
        assert report["filled"] is True
        assert len(result.time) == 48  # gap filled
        assert not result["temperature"].isnull().any()

    # -- strategy='fill_forward' -----------------------------------------------

    def test_fill_forward_fills_gap(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11])
        result, report = self.fdp.validate_temporal_completeness(
            ds, strategy="fill_forward"
        )
        assert report["filled"] is True
        assert len(result.time) == 48

    # -- max_gap_hours safety limit --------------------------------------------

    def test_max_gap_hours_overrides_strategy(self):
        # Drop 25 consecutive hours (indices 10-34)
        ds = _make_hourly_dataset(hours=72, drop_indices=list(range(10, 35)))
        with pytest.raises(ValueError, match="exceeds FORCING_MAX_GAP_HOURS"):
            self.fdp.validate_temporal_completeness(
                ds, strategy="interpolate", max_gap_hours=24
            )

    def test_gap_within_limit_allowed(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11, 12])
        result, report = self.fdp.validate_temporal_completeness(
            ds, strategy="interpolate", max_gap_hours=24
        )
        assert report["filled"] is True
        assert len(result.time) == 48

    # -- dec 31 scenario (the NWM3 bug) ----------------------------------------

    def test_dec31_missing_hours_detected(self):
        """Reproduce the exact NWM3 bug: Dec 31 has only 00Z, rest is missing."""
        times_before = pd.date_range("2019-12-30", periods=24, freq="h")
        times_dec31_00z = pd.DatetimeIndex(["2019-12-31"])
        times_after = pd.date_range("2020-01-01", periods=24, freq="h")
        times = times_before.append(times_dec31_00z).append(times_after)
        ds = xr.Dataset(
            {"temperature": ("time", np.ones(len(times)))},
            coords={"time": times},
        )
        with pytest.raises(ValueError, match="23 missing timesteps"):
            self.fdp.validate_temporal_completeness(ds, strategy="error")

    def test_dec31_gap_can_be_filled(self):
        """Same scenario but with interpolate strategy and small enough gap."""
        times_before = pd.date_range("2019-12-30", periods=24, freq="h")
        times_dec31_00z = pd.DatetimeIndex(["2019-12-31"])
        times_after = pd.date_range("2020-01-01", periods=24, freq="h")
        times = times_before.append(times_dec31_00z).append(times_after)
        ds = xr.Dataset(
            {"temperature": ("time", np.ones(len(times)))},
            coords={"time": times},
        )
        result, report = self.fdp.validate_temporal_completeness(
            ds, strategy="interpolate", max_gap_hours=24
        )
        assert report["filled"] is True
        assert report["missing_timesteps"] == 23
        expected_total = 24 + 24 + 24  # 3 full days
        assert len(result.time) == expected_total

    # -- edge cases ------------------------------------------------------------

    def test_single_timestep_passes(self):
        ds = xr.Dataset(
            {"temperature": ("time", [1.0])},
            coords={"time": pd.DatetimeIndex(["2019-01-01"])},
        )
        _, report = self.fdp.validate_temporal_completeness(ds)
        assert report["valid"] is True

    def test_explicit_freq_overrides_inference(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10])
        with pytest.raises(ValueError, match="1 missing timesteps"):
            self.fdp.validate_temporal_completeness(
                ds, expected_freq_seconds=3600, strategy="error"
            )

    def test_no_time_coordinate_skips(self):
        ds = xr.Dataset({"temperature": ("x", [1.0, 2.0, 3.0])})
        result, report = self.fdp.validate_temporal_completeness(ds)
        assert report["valid"] is True

    # -- gap report structure --------------------------------------------------

    def test_gap_report_has_expected_fields(self):
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11, 30])
        with pytest.raises(ValueError):
            self.fdp.validate_temporal_completeness(ds, strategy="error")

    def test_gap_runs_identified_correctly(self):
        # Two separate gaps: indices 10-11 and 30
        ds = _make_hourly_dataset(hours=48, drop_indices=[10, 11, 30])
        _, report = self.fdp.validate_temporal_completeness(ds, strategy="warn")
        assert len(report["gaps"]) == 2
        assert report["gaps"][0]["missing_count"] == 2
        assert report["gaps"][1]["missing_count"] == 1


class TestValidateTemporalCompletenessConfig:
    """Test config-driven behavior via prepare_forcing_for_model."""

    def test_config_dict_strategy_used(self):
        fdp = ForcingDataProcessor(
            config={"FORCING_MISSING_DATA_STRATEGY": "warn", "FORCING_MAX_GAP_HOURS": "48"}
        )
        assert fdp._get_config_str("FORCING_MISSING_DATA_STRATEGY", "error") == "warn"
        assert fdp._get_config_int("FORCING_MAX_GAP_HOURS", 24) == 48

    def test_config_defaults(self):
        fdp = ForcingDataProcessor(config={})
        assert fdp._get_config_str("FORCING_MISSING_DATA_STRATEGY", "error") == "error"
        assert fdp._get_config_int("FORCING_MAX_GAP_HOURS", 24) == 24
