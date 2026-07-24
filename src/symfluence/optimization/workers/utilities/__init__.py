# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Worker utilities.

- StreamflowMetrics: shared metric calculation utilities
- RoutingDecider: lives in ``symfluence.models.utilities``; resolved lazily
  here for backward compatibility (optimization must not import models at
  module level).
"""
from __future__ import annotations

from .streamflow_metrics import StreamflowMetrics

__all__ = ['RoutingDecider', 'StreamflowMetrics']


def __getattr__(name: str):
    if name == 'RoutingDecider':
        from symfluence.models.utilities.routing_decider import RoutingDecider
        globals()[name] = RoutingDecider
        return RoutingDecider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
