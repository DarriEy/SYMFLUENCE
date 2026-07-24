# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.build.build_snippet_catalog``."""
from __future__ import annotations

import symfluence.core.build.build_snippet_catalog as _impl
from symfluence.core.build.build_snippet_catalog import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
