# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.core.calibration.optimizers.lifecycle import adjust_end_time_for_forcing, fallback_simulation_dir


@pytest.mark.parametrize(
    ("timestep", "expected"),
    [(3600, "2020-01-01 23:00"), (10800, "2020-01-01 21:00"), (21600, "2020-01-01 18:00")],
)
def test_adjust_end_time_for_forcing(timestep: int, expected: str) -> None:
    assert adjust_end_time_for_forcing("2020-01-01 23:00", timestep) == expected


@pytest.mark.parametrize(
    ("timestep", "expected"),
    [(5400, "2020-01-01 22:00"), (9000, "2020-01-01 20:00")],
)
def test_non_integer_hour_timesteps_round_to_whole_last_hour(timestep: int, expected: str) -> None:
    assert adjust_end_time_for_forcing("2020-01-01 23:00", timestep) == expected


def test_subhourly_end_time_is_unchanged() -> None:
    assert adjust_end_time_for_forcing("2020-01-01 23:45", 900) == "2020-01-01 23:45"


def test_fallback_simulation_dir_is_deterministic() -> None:
    assert fallback_simulation_dir("/tmp/example", "bow", "dds") == Path(
        "/tmp/example/symfluence/bow/run_dds"
    )
