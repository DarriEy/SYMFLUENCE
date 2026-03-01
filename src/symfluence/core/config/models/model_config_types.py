# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared type aliases for model-specific config models."""

from typing import Literal

# Spatial mode types for hydrological models
SpatialModeType = Literal['lumped', 'semi_distributed', 'distributed', 'auto']

__all__ = ['SpatialModeType']
