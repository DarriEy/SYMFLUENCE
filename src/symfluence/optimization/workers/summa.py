# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Deprecated shim: the SUMMA worker implementation moved to
``symfluence.models.summa.calibration.worker_impl`` (it is SUMMA-specific and
belongs with the model package). Resolved lazily so that optimization never
imports the models layer at import time.
"""
from __future__ import annotations

import importlib

_CANONICAL = "symfluence.models.summa.calibration.worker_impl"


def __getattr__(name: str):
    value = getattr(importlib.import_module(_CANONICAL), name)
    globals()[name] = value
    return value
