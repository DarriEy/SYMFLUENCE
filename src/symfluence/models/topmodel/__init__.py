# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.topmodel import ...``.

The TOPMODEL model has been extracted into the standalone ``jtopmodel`` package.
This module re-exports from ``jtopmodel`` so existing code continues to work.
"""

from jtopmodel import *  # noqa: F401,F403
from jtopmodel import __all__, register  # noqa: F401
