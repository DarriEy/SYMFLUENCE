# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Read-only model-ready artifact contract for model adapters."""
from __future__ import annotations

from .attributes_reader import AttributesReader, open_canonical_attributes
from .cf_conventions import (
    CANONICAL_FORCING,
    CANONICAL_FORCING_ALIASES,
    CF_STANDARD_NAMES,
    build_global_attrs,
    resolve_forcing_var,
)
from .forcing_reader import (
    assert_consistent_spatial_dims,
    assert_consistent_within_discretization,
    forcing_timestep_seconds,
    open_canonical_forcing,
    resample_canonical_forcing,
    validated_timestep_seconds,
)
from .path_resolver import resolve_model_ready_path

__all__ = [
    "CANONICAL_FORCING",
    "CANONICAL_FORCING_ALIASES",
    "CF_STANDARD_NAMES",
    "AttributesReader",
    "assert_consistent_spatial_dims",
    "assert_consistent_within_discretization",
    "build_global_attrs",
    "forcing_timestep_seconds",
    "open_canonical_forcing",
    "open_canonical_attributes",
    "resample_canonical_forcing",
    "resolve_forcing_var",
    "resolve_model_ready_path",
    "validated_timestep_seconds",
]
