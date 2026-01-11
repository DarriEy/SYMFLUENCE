"""
Geospatial validation utilities for SYMFLUENCE tests.

Functions for comparing shapefiles and validating geospatial outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 100.0


def _project_for_signature(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Project GeoDataFrame to Web Mercator for area calculations.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        GeoDataFrame projected to EPSG:3857, or original if projection fails
    """
    if gdf.crs is None:
        return gdf
    try:
        return gdf.to_crs("EPSG:3857")
    except Exception:
        return gdf


def load_shapefile_signature(path: Path) -> Dict[str, object]:
    """
    Load a shapefile and extract its signature for comparison.

    Extracts metadata including CRS, columns, feature count, bounds, and area statistics.

    Args:
        path: Path to shapefile

    Returns:
        Dict containing signature metadata:
            - crs: Coordinate reference system string
            - columns: Sorted tuple of non-geometry column names
            - count: Number of features
            - bounds: Total bounds as numpy array
            - area_stats: Tuple of (min_area, max_area, total_area)
    """
    gdf = gpd.read_file(path)
    gdf_projected = _project_for_signature(gdf)
    columns = tuple(sorted([col for col in gdf.columns if col != "geometry"]))
    count = len(gdf_projected)
    if count:
        areas = gdf_projected.geometry.area
        area_stats: Tuple[float, float, float] = (
            float(areas.min()),
            float(areas.max()),
            float(areas.sum()),
        )
        bounds = gdf_projected.total_bounds
    else:
        area_stats = (0.0, 0.0, 0.0)
        bounds = np.array([0.0, 0.0, 0.0, 0.0])

    return {
        "crs": str(gdf.crs),
        "columns": columns,
        "count": count,
        "bounds": bounds,
        "area_stats": area_stats,
    }


def assert_shapefile_signature_matches(
    actual_path: Path,
    expected_signature: Dict[str, object],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """
    Assert that a shapefile matches an expected signature.

    Compares feature count, columns, CRS, bounds, and area statistics.
    Uses numpy.testing.assert_allclose for numerical comparisons.

    Args:
        actual_path: Path to shapefile to validate
        expected_signature: Expected signature dictionary from load_shapefile_signature
        rtol: Relative tolerance for numerical comparisons (default: 1e-4)
        atol: Absolute tolerance for numerical comparisons (default: 100.0)

    Raises:
        AssertionError: If signatures don't match
    """
    actual_signature = load_shapefile_signature(actual_path)

    assert (
        actual_signature["count"] == expected_signature["count"]
    ), f"Feature count mismatch for {actual_path}"
    assert (
        actual_signature["columns"] == expected_signature["columns"]
    ), f"Column mismatch for {actual_path}"
    assert (
        actual_signature["crs"] == expected_signature["crs"]
    ), f"CRS mismatch for {actual_path}"

    np.testing.assert_allclose(
        actual_signature["bounds"],
        expected_signature["bounds"],
        rtol=rtol,
        atol=atol,
        err_msg=f"Bounds mismatch for {actual_path}",
    )
    np.testing.assert_allclose(
        actual_signature["area_stats"],
        expected_signature["area_stats"],
        rtol=rtol,
        atol=atol,
        err_msg=f"Area stats mismatch for {actual_path}",
    )
