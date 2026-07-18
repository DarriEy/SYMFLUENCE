# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure request-planning helpers for acquisition orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def expected_forcing_times(dataset: str, start: Any, end: Any) -> pd.DatetimeIndex | None:
    """Return the expected cached time axis for datasets with fixed resolution."""
    resolution_hours = {"CARRA": 1, "CERRA": 3}
    hours = resolution_hours.get(dataset.upper())
    if hours is None:
        return None

    try:
        parsed_start, parsed_end = pd.to_datetime(start), pd.to_datetime(end)
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed_start) or pd.isna(parsed_end) or parsed_end < parsed_start:
        return None
    return pd.date_range(parsed_start, parsed_end, freq=f"{hours}h")


def forcing_request_facts(
    start: Any,
    end: Any,
    dataset_variables: Iterable[str] | None,
    default_variables: Iterable[str] | None,
) -> tuple[tuple[str, str] | None, frozenset[str] | None]:
    """Normalize a forcing request window and explicit variable selection."""
    window = (str(start), str(end)) if start and end else None
    selected = dataset_variables if dataset_variables is not None else default_variables
    variables = frozenset(selected) if isinstance(selected, list) and selected else None
    return window, variables
