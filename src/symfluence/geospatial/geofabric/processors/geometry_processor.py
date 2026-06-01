# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Geometry processing operations for geofabric data.

Provides geometry cleaning, simplification, and winding order correction.
Used primarily by coastal delineation methods.

Refactored from geofabric_utils.py (2026-01-01)
"""

from typing import Any, Optional

from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid


class GeometryProcessor:
    """
    Geometry operations for geofabric processing.

    All methods are static since they don't require instance state.
    """

    @staticmethod
    def clean_geometries(geometry) -> Optional[Any]:
        """
        Clean and validate geometry.

        Uses make_valid() to fix invalid geometries (self-intersections, etc.).

        Args:
            geometry: Shapely geometry object

        Returns:
            Cleaned geometry, or None if geometry is None or invalid
        """
        if geometry is None or not geometry.is_valid:
            return None
        try:
            return make_valid(geometry)
        except (ValueError, AttributeError, GEOSException):
            # Shapely geometry operations can raise various errors
            return None

    @staticmethod
    def simplify_geometry(geometry, tolerance: float = 1) -> Any:
        """
        Simplify geometry while preserving topology.

        Args:
            geometry: Shapely geometry object
            tolerance: Simplification tolerance (default: 1)

        Returns:
            Simplified geometry, or original geometry if simplification fails
        """
        try:
            return geometry.simplify(tolerance, preserve_topology=True)
        except (ValueError, AttributeError, GEOSException):
            # Return original geometry if simplification fails
            return geometry

    @staticmethod
    def fix_polygon_winding(geometry) -> Optional[Any]:
        """
        Ensure correct winding order for polygon geometries.

        OGC standard requires exterior ring to be counter-clockwise (CCW)
        and holes to be clockwise (CW).

        Handles both Shapely 2.0+ (orient method) and older versions.

        Args:
            geometry: Shapely Polygon or MultiPolygon

        Returns:
            Geometry with corrected winding order, or None if geometry is None
        """
        if geometry is None:
            return None

        try:
            # Try Shapely 2.0+ method first
            if geometry.geom_type == 'Polygon':
                return geometry.orient(1.0)
            elif geometry.geom_type == 'MultiPolygon':
                return geometry.__class__([geom.orient(1.0) for geom in geometry.geoms])
        except AttributeError:
            # Fallback for older Shapely versions
            if geometry.geom_type == 'Polygon':
                # Make exterior ring counter-clockwise
                if not geometry.exterior.is_ccw:
                    geometry = Polygon(
                        list(geometry.exterior.coords)[::-1],
                        [list(interior.coords)[::-1] for interior in geometry.interiors]
                    )
            elif geometry.geom_type == 'MultiPolygon':
                # Fix each polygon in the multipolygon
                polygons = []
                for poly in geometry.geoms:
                    if not poly.exterior.is_ccw:
                        poly = Polygon(
                            list(poly.exterior.coords)[::-1],
                            [list(interior.coords)[::-1] for interior in poly.interiors]
                        )
                    polygons.append(poly)
                geometry = MultiPolygon(polygons)

        return geometry

    @staticmethod
    def remove_spikes(
        geometry,
        resolution: float = 50.0,
        max_iterations: int = 6,
        min_fill_ratio: float = 0.10,
        keep_area_fraction: float = 0.6,
    ) -> Any:
        """Remove thin "tentacle" artifacts from a watershed polygon.

        Raster→vector watershed delineation can produce polygons consisting of a
        compact body plus a near-zero-area tentacle that threads far across the
        domain (a flow-routing leak toward the coast/outlet). These are *valid*
        rings, so ``make_valid`` / ``simplify`` / a winding fix cannot remove
        them, and an area filter passes them because the body clears the
        threshold. This applies an adaptive raster morphological *opening* to
        shave such tentacles while preserving the body.

        The operation is **safe**: it only acts on low-"fill" (spiky) polygons,
        grows the opening only until the shape is compact again, and falls back
        to the original geometry whenever opening would erode real area below
        ``keep_area_fraction`` — so a genuinely thin basin is never damaged.

        Args:
            geometry: A shapely Polygon/MultiPolygon in a **projected** CRS
                (units of metres); see :meth:`despike_geodataframe` for CRS handling.
            resolution: Raster cell size (m) used for the morphological opening.
            max_iterations: Maximum opening iterations (≈ ``max_iterations`` *
                ``resolution`` metres of tentacle width that can be removed).
            min_fill_ratio: Target area / bounding-box-area; polygons already at
                or above this are left untouched (deemed compact).
            keep_area_fraction: Stop (and keep the last good result) once opening
                would drop the area below this fraction of the original.

        Returns:
            The despiked geometry, or the original if no safe improvement found.
        """
        if geometry is None or geometry.is_empty:
            return geometry

        from rasterio import features
        from rasterio import transform as rtransform
        from scipy import ndimage
        from shapely.geometry import shape as shapely_shape

        minx, miny, maxx, maxy = geometry.bounds
        bbox_area = max((maxx - minx) * (maxy - miny), 1e-9)
        if geometry.area / bbox_area >= min_fill_ratio:
            return geometry  # already compact — nothing to do

        pad = resolution * (max_iterations + 2)
        minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1
        if width * height > 8_000_000:  # pathological bbox — skip rather than thrash
            return geometry

        affine = rtransform.from_origin(minx, maxy, resolution, resolution)
        mask = features.rasterize(
            [(geometry, 1)], out_shape=(height, width), transform=affine,
            fill=0, dtype="uint8", all_touched=False,
        )
        if mask.sum() == 0:
            return geometry

        best = geometry
        for iterations in range(1, max_iterations + 1):
            opened = ndimage.binary_opening(mask, iterations=iterations)
            if opened.sum() == 0:
                break
            parts = [
                shapely_shape(geom)
                for geom, val in features.shapes(opened.astype("uint8"), mask=opened, transform=affine)
                if val == 1
            ]
            if not parts:
                break
            candidate = max(parts, key=lambda p: p.area)
            if candidate.area < keep_area_fraction * geometry.area:
                break  # eroding into the real body — keep the last safe result
            best = candidate
            cb = candidate.bounds
            if candidate.area / max((cb[2] - cb[0]) * (cb[3] - cb[1]), 1e-9) >= min_fill_ratio:
                break  # compact enough now
        return best

    @staticmethod
    def despike_geodataframe(gdf, resolution: float = 50.0, **kwargs):
        """Despike every geometry in a GeoDataFrame (handles CRS automatically).

        Reprojects a geographic GeoDataFrame to its local UTM CRS so the
        metre-based opening is meaningful, despikes each geometry via
        :meth:`remove_spikes`, then reprojects back. Compact basins are returned
        untouched, so only the spiky minority pays the raster cost.
        """
        if gdf is None or gdf.empty:
            return gdf
        src_crs = gdf.crs
        reproject = bool(src_crs and src_crs.is_geographic)
        work = gdf.to_crs(gdf.estimate_utm_crs()) if reproject else gdf
        work = work.copy()
        work["geometry"] = work.geometry.apply(
            lambda g: GeometryProcessor.remove_spikes(g, resolution=resolution, **kwargs)
        )
        return work.to_crs(src_crs) if reproject else work
