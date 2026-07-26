# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-agnostic multi-gauge calibration contract."""
from __future__ import annotations

from .gauge_mapping import ensure_gauge_mapping
from .metrics import MultiGaugeMetrics, create_multi_gauge_config

__all__ = ['MultiGaugeMetrics', 'create_multi_gauge_config', 'ensure_gauge_mapping']
