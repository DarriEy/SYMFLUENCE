# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA-Specific Optimizer Mixin (Backward Compatibility)

.. deprecated::
    This module has been moved to symfluence.models.summa.calibration.optimizer_mixin
    as part of the effort to consolidate model-specific code in the models package.

    This re-export is provided for backward compatibility. Please update imports to:

        from symfluence.models.summa.calibration import SUMMAOptimizerMixin

Migration Context:
    As part of the pre-migration refactoring to separate model-specific concerns,
    SUMMA-specific optimization code has been moved to the SUMMA model package.
    This establishes the pattern for the main migration where all model-specific
    optimizers, workers, parameter managers, and calibration targets will move to
    their respective model directories.

    Old location: symfluence.optimization.mixins.summa_optimizer_mixin
    New location: symfluence.models.summa.calibration.optimizer_mixin
"""

# Backward compatibility re-export
from symfluence.models.summa.calibration.optimizer_mixin import SUMMAOptimizerMixin

__all__ = ['SUMMAOptimizerMixin']
