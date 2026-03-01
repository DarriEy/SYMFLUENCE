# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model comparison plotter composed from focused mixins."""

from __future__ import annotations

from symfluence.reporting.core.base_plotter import BasePlotter
from symfluence.reporting.plotters.model_comparison._comparison import (
    ModelComparisonDefaultVsCalibratedMixin,
)
from symfluence.reporting.plotters.model_comparison._core import ModelComparisonCoreMixin
from symfluence.reporting.plotters.model_comparison._data_loading import (
    ModelComparisonDataLoadingMixin,
)
from symfluence.reporting.plotters.model_comparison._overview import (
    ModelComparisonOverviewMixin,
)


class ModelComparisonPlotter(
    ModelComparisonOverviewMixin,
    ModelComparisonDefaultVsCalibratedMixin,
    ModelComparisonDataLoadingMixin,
    ModelComparisonCoreMixin,
    BasePlotter,
):
    """Creates model comparison overview sheets for observations and model runs."""

    pass
