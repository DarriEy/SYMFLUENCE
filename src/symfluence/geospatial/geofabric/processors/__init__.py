# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geofabric processing modules.

Provides processing backends for geofabric generation:
    taudem_executor: TauDEM command execution with MPI support
    gdal_processor: GDAL/OGR raster and vector processing
    geometry_processor: Shapely geometry cleaning and validation
    graph_processor: River network graph construction and traversal
    stream_burner: DEM stream burning for flat-terrain delineation
"""
