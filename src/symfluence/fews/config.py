# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FEWS adapter configuration models (re-export).

The FEWS config schemas now live in
:mod:`symfluence.core.config.models.fews` because the root config model
composes them. This module re-exports them so the FEWS adapters can keep
importing from ``symfluence.fews.config``. ``fews`` -> ``core`` is the correct
dependency direction.
"""
from __future__ import annotations

from symfluence.core.config.models.fews import FEWSConfig, IDMapEntry

__all__ = ["FEWSConfig", "IDMapEntry"]
