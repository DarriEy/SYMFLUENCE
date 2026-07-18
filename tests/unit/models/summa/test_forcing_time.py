# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from symfluence.models.summa.forcing_time import infer_forcing_step_from_filenames


def test_infers_median_positive_forcing_step() -> None:
    files = [
        "forcing_2020-01-01-03-00-00.nc",
        "forcing_2020-01-01-00-00-00.nc",
        "forcing_2020-01-01-01-00-00.nc",
    ]

    assert infer_forcing_step_from_filenames(files) == 5400


def test_ignores_unparseable_and_duplicate_timestamps() -> None:
    files = [
        "forcing_invalid.nc",
        "forcing_2020-01-01-00-00-00.nc",
        "copy_2020-01-01-00-00-00.nc",
    ]

    assert infer_forcing_step_from_filenames(files) is None
