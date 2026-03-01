# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Domain discretization module for SYMFLUENCE geospatial processing.

This package provides tools for discretizing hydrological domains into
computational units such as Grouped Response Units (GRUs), Hydrologic
Response Units (HRUs), or grid cells for various hydrological models.

Key components:
    DiscretizationArtifacts: Container for discretization output files
    DomainDiscretizationRunner: High-level runner for discretization workflows
    DomainDiscretizer: Core discretization logic and algorithms

Example:
    >>> from symfluence.geospatial.discretization import DomainDiscretizer
    >>> discretizer = DomainDiscretizer(config)
    >>> discretizer.run()
"""
from .artifacts import DiscretizationArtifacts
from .core import DomainDiscretizationRunner, DomainDiscretizer

__all__ = ["DiscretizationArtifacts", "DomainDiscretizationRunner", "DomainDiscretizer"]
