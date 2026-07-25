# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: shared acquisition infrastructure moved upstream to
``symfluence.data.acquisition.appeears_base`` (it stays with the framework when handlers lift to the community
services)."""
from __future__ import annotations

import symfluence.data.acquisition.appeears_base as _impl
from symfluence.data.acquisition.appeears_base import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)
