# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Calibration Targets (Backward Compatibility)

.. deprecated::
    Moved to symfluence.models.summa.calibration.targets
"""

from symfluence.models.summa.calibration.targets import SUMMAETTarget, SUMMASnowTarget, SUMMAStreamflowTarget

__all__ = ['SUMMAStreamflowTarget', 'SUMMASnowTarget', 'SUMMAETTarget']
