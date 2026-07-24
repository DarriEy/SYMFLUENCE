# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: this module moved to ``symfluence.core.metrics.metrics_types``.

Kept so external plugins and downstream code importing the old
``symfluence.evaluation.metrics_types`` path keep working. New code should import from ``symfluence.core.metrics.metrics_types``.
"""
from __future__ import annotations

import symfluence.core.metrics.metrics_types as _impl
from symfluence.core.metrics.metrics_types import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
