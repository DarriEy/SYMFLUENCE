# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Hydrological performance metrics (core contract surface).

Promoted from ``symfluence.evaluation`` so that calibration workers and model
packages can compute metrics without depending on the evaluation capability
package. ``symfluence.core.metrics`` re-exports the stable facade from
``.metrics``; the focused submodules (``metrics_core``, ``metrics_hydrograph``,
``metrics_registry``, ``metrics_types``, ``metric_transformer``,
``streamflow_metrics``) remain importable individually.
"""
from __future__ import annotations

from symfluence.core.metrics import metrics as _facade
from symfluence.core.metrics.metrics import *  # noqa: F401,F403

__all__ = list(_facade.__all__)


def __getattr__(name: str):
    return getattr(_facade, name)
