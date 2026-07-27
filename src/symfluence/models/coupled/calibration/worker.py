# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.calibration.coupled.worker``.

Importing this module still fires the ``@R.workers.add('COUPLED')``
registration (via the canonical module), which is how
``optimization._autodiscover`` discovers the coupled worker.
"""
from __future__ import annotations

import symfluence.core.calibration.coupled.worker as _impl
from symfluence.core.calibration.coupled.worker import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
