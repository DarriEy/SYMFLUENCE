"""
SUMMA Calibration Targets (Backward Compatibility)

.. deprecated::
    Moved to symfluence.models.summa.calibration.targets
"""

from symfluence.models.summa.calibration.targets import SUMMAETTarget, SUMMASnowTarget, SUMMAStreamflowTarget

__all__ = ['SUMMAStreamflowTarget', 'SUMMASnowTarget', 'SUMMAETTarget']
