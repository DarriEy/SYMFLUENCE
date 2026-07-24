# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: this module moved to ``symfluence.core.metrics.metric_transformer``.

Kept so external plugins and downstream code importing the old
``symfluence.evaluation.metric_transformer`` path keep working. New code should import from ``symfluence.core.metrics.metric_transformer``.
"""
from __future__ import annotations

import symfluence.core.metrics.metric_transformer as _impl
from symfluence.core.metrics.metric_transformer import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
