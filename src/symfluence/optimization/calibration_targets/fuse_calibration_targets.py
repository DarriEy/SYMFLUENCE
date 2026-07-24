# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Deprecated shim: import from ``symfluence.models.fuse.calibration.targets`` instead.

Resolved lazily so this module never imports the models layer at import
time (optimization must not depend on models).
"""
from __future__ import annotations

import importlib

_EXPORTS = ['FUSEStreamflowTarget', 'FUSESnowTarget']
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(importlib.import_module("symfluence.models.fuse.calibration.targets"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
