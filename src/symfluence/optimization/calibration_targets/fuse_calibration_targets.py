# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Calibration Targets (Backward Compatibility)

.. deprecated::
    Moved to symfluence.models.fuse.calibration.targets
"""

from symfluence.models.fuse.calibration.targets import FUSESnowTarget, FUSEStreamflowTarget

__all__ = ['FUSEStreamflowTarget', 'FUSESnowTarget']
