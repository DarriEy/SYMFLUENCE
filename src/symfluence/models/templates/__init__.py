# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model Templates Module.

Provides base classes and templates for implementing new hydrological models
using the unified execution framework.
"""

from .model_template import (
    ModelRunResult,
    UnifiedModelRunner,
    create_model_runner,
)

__all__ = [
    'UnifiedModelRunner',
    'ModelRunResult',
    'create_model_runner',
]
