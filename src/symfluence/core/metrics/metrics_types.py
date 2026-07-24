# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Types shared by evaluation metric modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MetricInfo:
    """Metadata for a performance metric."""

    name: str
    full_name: str
    range: Tuple[float, float]
    optimal: float
    direction: str
    units: str
    description: str
    reference: str
