"""
FUSE Calibration Targets (Backward Compatibility)

.. deprecated::
    Moved to symfluence.models.fuse.calibration.targets
"""

from symfluence.models.fuse.calibration.targets import FUSESnowTarget, FUSEStreamflowTarget

__all__ = ['FUSEStreamflowTarget', 'FUSESnowTarget']
