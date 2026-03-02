# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.hechms import ...``.

The HEC-HMS model has been extracted into the standalone ``jhechms`` package.
This module re-exports from ``jhechms`` so existing code continues to work.
"""

from jhechms import *  # noqa: F401,F403
from jhechms import __all__, register  # noqa: F401
