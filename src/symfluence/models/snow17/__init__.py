# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.snow17 import ...``.

The Snow-17 model has been extracted into the standalone ``jsnow17`` package.
This module re-exports from ``jsnow17`` so existing code continues to work.
"""

from jsnow17 import *  # noqa: F401,F403
from jsnow17 import __all__, register  # noqa: F401
