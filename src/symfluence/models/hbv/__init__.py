# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.hbv import ...``.

The HBV model has been extracted into the standalone ``jhbv`` package.
This module re-exports from ``jhbv`` so existing code continues to work.
"""

from jhbv import *  # noqa: F401,F403
from jhbv import __all__, register  # noqa: F401
