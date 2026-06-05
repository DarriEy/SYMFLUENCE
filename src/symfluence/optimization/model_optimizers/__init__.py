# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model-Specific Optimizers

Optimizers that inherit from BaseModelOptimizer for each supported model.
These provide a unified interface while handling model-specific setup.

Model-specific optimizers are available via:
1. Direct import: from symfluence.models.{model}.calibration.optimizer import {Model}ModelOptimizer
2. Registry pattern: OptimizerRegistry.get_optimizer('{MODEL}')

Registration happens via ``@OptimizerRegistry.register_optimizer``
decorators.  This module auto-discovers all model packages at import time
so that every ``calibration/optimizer.py`` is imported and its decorator
fires.
"""


def _register_optimizers():
    """Auto-discover and import model optimizers from all model packages.

    Scans ``symfluence.models.*`` for sub-packages that contain a
    ``calibration.optimizer`` module and imports each one to trigger its
    ``@register_optimizer`` decorator.  Models with no calibration support are
    skipped silently; models whose optimizer module *exists but fails to
    import* are surfaced at WARNING (see ``discover_calibration_components``)
    so a missing dependency no longer masquerades as "No optimizer registered".
    """
    import logging

    from symfluence.optimization._autodiscover import discover_calibration_components

    discover_calibration_components('optimizer', logging.getLogger(__name__))


# Trigger registration on import
_register_optimizers()

__all__: list[str] = []
