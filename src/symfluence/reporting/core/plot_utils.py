# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility exports for model-facing plotting utilities."""
from __future__ import annotations

from symfluence.core.reporting.plot_utils import (
    add_north_arrow,
    align_timeseries,
    calculate_flow_duration_curve,
    calculate_metrics,
    calculate_summary_statistics,
    format_metrics_for_display,
    resample_timeseries,
)

__all__ = [
    "add_north_arrow",
    "align_timeseries",
    "calculate_flow_duration_curve",
    "calculate_metrics",
    "calculate_summary_statistics",
    "format_metrics_for_display",
    "resample_timeseries",
]
