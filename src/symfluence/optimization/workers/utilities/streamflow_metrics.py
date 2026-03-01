# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Streamflow Metrics Utility (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.evaluation.utilities.streamflow_metrics
    as streamflow metrics calculation is evaluation logic, not optimization-specific.

    This re-export is provided for backward compatibility. Please update imports to:

        from symfluence.evaluation.utilities import StreamflowMetrics

Migration Context:
    As part of the pre-migration refactoring to improve separation of concerns,
    evaluation and metrics calculation code has been moved out of the optimization
    package to the evaluation package where it belongs conceptually.

    Old location: symfluence.optimization.workers.utilities.streamflow_metrics
    New location: symfluence.evaluation.utilities.streamflow_metrics
"""

# Backward compatibility re-export
from symfluence.evaluation.utilities.streamflow_metrics import StreamflowMetrics

__all__ = ['StreamflowMetrics']
