# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.modeling.utilities``.

Historical time-window and dataset-alignment re-exports now resolve through
their canonical core modeling utilities home.
"""
from __future__ import annotations

import symfluence.core.modeling.utilities as _impl
from symfluence.core.modeling.utilities import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
