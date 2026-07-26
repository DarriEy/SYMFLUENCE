# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility exports for model-facing shapefile helpers."""
from __future__ import annotations

from symfluence.core.reporting.shapefile_helper import ShapefileHelper, resolve_default_name

__all__ = ["ShapefileHelper", "resolve_default_name"]
