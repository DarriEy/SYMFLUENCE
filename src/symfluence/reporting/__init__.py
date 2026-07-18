# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Reporting and visualization utilities for SYMFLUENCE.
"""
from __future__ import annotations

from typing import TYPE_CHECKING


# Lazy re-export (PEP 562): ReportingManager pulls matplotlib and the plotter
# stack (~0.5 s); model plotter modules import this package at plugin-discovery
# time and must not pay for it.
def __getattr__(name: str):
    if name == 'ReportingManager':
        from .reporting_manager import ReportingManager
        globals()[name] = ReportingManager
        return ReportingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .reporting_manager import ReportingManager

__all__ = ['ReportingManager']
