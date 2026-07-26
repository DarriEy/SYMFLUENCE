# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility exports for the canonical forcing naming contract."""
from __future__ import annotations

from symfluence.core.modeling.forcing_naming import (
    discretization_key_from_name,
    discretization_token,
    forcing_name_matches_discretization,
    select_forcing_files,
)

__all__ = [
    "discretization_key_from_name",
    "discretization_token",
    "forcing_name_matches_discretization",
    "select_forcing_files",
]
