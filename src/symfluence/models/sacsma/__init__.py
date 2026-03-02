# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatibility shim for ``from symfluence.models.sacsma import ...``.

The SAC-SMA model has been extracted into the standalone ``jsacsma`` package.
This module re-exports from ``jsacsma`` so existing code continues to work.
"""

from jsacsma import *  # noqa: F401,F403
from jsacsma import __all__, register  # noqa: F401
