# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Discretization attribute calculators.

This module provides calculators for deriving various geospatial attributes
used in hydrological model discretization including elevation, aspect,
land class, soil class, and radiation characteristics.

Each submodule provides a `discretize` function for its respective attribute.
Import submodules directly to access their discretize functions:

    from symfluence.geospatial.discretization.attributes import elevation
    result = elevation.discretize(discretizer)
"""

from . import aspect, combined, elevation, grus, landclass, radiation, soilclass

__all__ = [
    "aspect",
    "combined",
    "elevation",
    "grus",
    "landclass",
    "radiation",
    "soilclass",
]
