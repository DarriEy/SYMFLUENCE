# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Common imports and base configuration for config models.

This module provides shared imports and the base ConfigDict used
across all configuration model classes.
"""

from pydantic import ConfigDict

# Standard ConfigDict for all config models
FROZEN_CONFIG = ConfigDict(extra='allow', populate_by_name=True, frozen=True)
