# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-facing plotting contracts and pure reporting helpers."""
from __future__ import annotations

from symfluence.core.reporting.base_plotter import BasePlotter
from symfluence.core.reporting.plot_config import DEFAULT_PLOT_CONFIG, PlotConfig
from symfluence.core.reporting.plot_utils import (
    add_north_arrow,
    align_timeseries,
    calculate_flow_duration_curve,
    calculate_metrics,
    calculate_summary_statistics,
    format_metrics_for_display,
    resample_timeseries,
)
from symfluence.core.reporting.shapefile_helper import ShapefileHelper, resolve_default_name

__all__ = [
    "BasePlotter",
    "DEFAULT_PLOT_CONFIG",
    "PlotConfig",
    "ShapefileHelper",
    "add_north_arrow",
    "align_timeseries",
    "calculate_flow_duration_curve",
    "calculate_metrics",
    "calculate_summary_statistics",
    "format_metrics_for_display",
    "resample_timeseries",
    "resolve_default_name",
]
