# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic calibration engine (contract surface for model packages).

Promoted from ``symfluence.optimization`` so model adapter packages can
subclass the calibration bases while depending only on ``symfluence.core``:

- :class:`~symfluence.core.calibration.optimizers.base_model_optimizer.BaseModelOptimizer`
- :class:`~symfluence.core.calibration.workers.base_worker.BaseWorker`
- :class:`~symfluence.core.calibration.parameters.base_parameter_manager.BaseParameterManager`
- the algorithm suite under ``.optimizers.algorithms``
- the parameter bounds registry under ``.parameters``

Model-agnostic only: anything SUMMA-/FUSE-/...-specific stays with its model
package. The historical ``symfluence.optimization.*`` import paths remain as
back-compat shims.
"""
from __future__ import annotations

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.parameters import BaseParameterManager, ParameterBoundsRegistry
from symfluence.core.calibration.targets import create_calibration_target, resolve_calibration_target
from symfluence.core.calibration.workers import (
    BaseWorker,
    InMemoryModelWorker,
    WorkerResult,
    WorkerTask,
)

__all__ = [
    'BaseModelOptimizer',
    'BaseParameterManager',
    'create_calibration_target',
    'ParameterBoundsRegistry',
    'resolve_calibration_target',
    'BaseWorker',
    'InMemoryModelWorker',
    'WorkerResult',
    'WorkerTask',
]
