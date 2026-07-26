# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted multi-gauge mapping contract."""
from __future__ import annotations

from symfluence.core.calibration.multi_gauge.gauge_mapping import ensure_gauge_mapping

__all__ = ["ensure_gauge_mapping"]
