# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.xinanjiang import ...``.

The Xinanjiang model has been extracted into the standalone ``jxaj`` package.
This module re-exports from ``jxaj`` so existing code continues to work.
"""

from jxaj import *  # noqa: F401,F403
from jxaj import __all__, register  # noqa: F401
