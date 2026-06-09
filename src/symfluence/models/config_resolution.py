# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Backward-compatibility shim.

The model configuration resolution helpers now live in
:mod:`symfluence.core.config.config_resolution` (they depend only on the
unified ``R`` registry, so they belong in ``core``). This module re-exports
them so existing ``from symfluence.models.config_resolution import ...`` call
sites keep working. ``models`` → ``core`` is the correct dependency direction.
"""
from __future__ import annotations

from symfluence.core.config.config_resolution import (
    get_config_adapter,
    get_config_defaults,
    get_config_schema,
    get_config_transformers,
    get_config_validator,
    validate_model_config,
)

__all__ = [
    'get_config_adapter',
    'get_config_schema',
    'get_config_defaults',
    'get_config_transformers',
    'get_config_validator',
    'validate_model_config',
]
