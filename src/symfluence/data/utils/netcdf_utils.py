# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted NetCDF encoding helpers."""
from __future__ import annotations

from symfluence.core.modeling.netcdf_utils import create_minimal_encoding, create_netcdf_encoding

__all__ = ["create_minimal_encoding", "create_netcdf_encoding"]
