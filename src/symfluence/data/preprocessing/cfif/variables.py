# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted CFIF variable contract."""
from __future__ import annotations

from symfluence.core.modeling.cfif.variables import (
    CFIF_TO_SUMMA_MAPPING,
    CFIF_VARIABLES,
    SUMMA_TO_CFIF_MAPPING,
    get_cfif_standard_name,
    get_cfif_units,
    get_cfif_variable,
    normalize_to_cfif,
    validate_cfif_dataset,
)

__all__ = [
    "CFIF_TO_SUMMA_MAPPING",
    "CFIF_VARIABLES",
    "SUMMA_TO_CFIF_MAPPING",
    "get_cfif_standard_name",
    "get_cfif_units",
    "get_cfif_variable",
    "normalize_to_cfif",
    "validate_cfif_dataset",
]
