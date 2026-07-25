# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.modeling.base`` (adapter contract tier)."""
from __future__ import annotations

import symfluence.core.modeling.base as _impl
from symfluence.core.modeling.base import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
