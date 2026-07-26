# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the core calibration implementation."""
from __future__ import annotations

import symfluence.core.calibration.multi_gauge.metrics as _impl
from symfluence.core.calibration.multi_gauge.metrics import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
