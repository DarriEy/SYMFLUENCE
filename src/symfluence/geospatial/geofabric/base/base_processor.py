# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base processor for common geospatial operations.

Provides utility methods for geometry processing that can be used as a mixin.
Used primarily by coastal delineation methods.

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any


class BaseGeospatialProcessor:
    """
    Mixin class for common geospatial processing operations.

    This class is designed to be used as a base class or mixin,
    providing geometry utility methods to other classes.
    """

    def __init__(self, logger: Any):
        """
        Initialize base processor.

        Args:
            logger: Logger instance
        """
        self.logger = logger
