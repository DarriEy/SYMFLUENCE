# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.project.model_manager`` (orchestration layer)."""
from __future__ import annotations

import symfluence.project.model_manager as _impl
from symfluence.project.model_manager import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
