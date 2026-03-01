# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility facade for model comparison plotting.

Public API remains stable at:
    symfluence.reporting.plotters.model_comparison_plotter.ModelComparisonPlotter

Implementation now lives in the dedicated package:
    symfluence.reporting.plotters.model_comparison
"""

from symfluence.reporting.plotters.model_comparison.plotter import ModelComparisonPlotter

__all__ = ["ModelComparisonPlotter"]
