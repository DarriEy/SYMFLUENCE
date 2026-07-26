# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for promoted model-ready CF conventions."""
from __future__ import annotations

import symfluence.core.modeling.model_ready.cf_conventions as _impl
from symfluence.core.modeling.model_ready.cf_conventions import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
