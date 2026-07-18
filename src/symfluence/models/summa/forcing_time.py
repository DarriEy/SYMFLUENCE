# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure time-axis utilities used by SUMMA forcing preparation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import median


def infer_forcing_step_from_filenames(forcing_files: list[str]) -> int | None:
    """Infer the forcing interval from timestamped filenames."""
    forcing_times: list[datetime] = []
    for forcing_file in forcing_files:
        time_token = Path(forcing_file).stem.split("_")[-1]
        try:
            forcing_times.append(datetime.strptime(time_token, "%Y-%m-%d-%H-%M-%S"))
        except ValueError:
            continue

    forcing_times.sort()
    diffs = [
        (current - previous).total_seconds()
        for previous, current in zip(forcing_times, forcing_times[1:])
        if current > previous
    ]
    return int(median(diffs)) if diffs else None
