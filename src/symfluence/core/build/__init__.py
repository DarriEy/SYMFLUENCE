# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Build-environment helpers for model build instructions.

Promoted from ``symfluence.cli.services`` so model packages can describe how
to compile their binaries while depending only on ``symfluence.core``
(models must not import the cli layer).
"""
from __future__ import annotations

from .build_snippet_catalog import BuildSnippetCatalog
from .build_snippets import (
    get_all_snippets,
    get_bison_detection_and_build,
    get_common_build_environment,
    get_flex_detection_and_build,
    get_geos_proj_detection,
    get_hdf5_detection,
    get_netcdf_detection,
    get_netcdf_lib_detection,
    get_safe_build_path,
    get_udunits2_detection_and_build,
)

__all__ = [
    'BuildSnippetCatalog',
    'get_all_snippets',
    'get_bison_detection_and_build',
    'get_common_build_environment',
    'get_flex_detection_and_build',
    'get_geos_proj_detection',
    'get_hdf5_detection',
    'get_netcdf_detection',
    'get_netcdf_lib_detection',
    'get_safe_build_path',
    'get_udunits2_detection_and_build',
]
