# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Allow running SYMFLUENCE as ``python -m symfluence``."""
from __future__ import annotations

import sys

from symfluence.main_cli import main

sys.exit(main())
