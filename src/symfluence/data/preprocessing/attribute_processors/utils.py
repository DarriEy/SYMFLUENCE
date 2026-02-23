"""
Utility functions for attribute processing in SYMFLUENCE.

Provides helper functions for spatial attribute extraction and processing,
including raster operations, zonal statistics validation, and data quality checks.

Helper Functions:
    - crop_raster_to_bbox: Clip raster to bounding box (re-exported from spatial_utils)
    - count_pixels_in_catchment: Count valid pixels within catchment geometry
    - check_zonal_stats_outcomes: Validate and clean zonal statistics results

These utilities support the attribute processor modules (ElevationProcessor,
SoilProcessor, LandcoverProcessor, etc.) by providing common spatial operations.

See Also:
    - symfluence.data.utils.spatial_utils: Core spatial operations
    - symfluence.data.preprocessing.attribute_processors: Specialized processors
"""

from typing import Dict, List, Union

import numpy as np

# Re-export from consolidated spatial utilities


def count_pixels_in_catchment(raster_src, catchment_geometry) -> int:
    """
    Count pixels in catchment.

    Args:
        raster_src: Rasterio source
        catchment_geometry: Shapely geometry

    Returns:
        Number of pixels
    """
    # Placeholder for implementation
    return 0


def check_zonal_stats_outcomes(zonal_out: List[Dict], new_val: Union[float, int] = np.nan) -> List[Dict]:
    """
    Validate and clean zonal statistics by replacing None values.

    Processes output from rasterstats.zonal_stats to handle None values that
    occur when zones don't intersect with valid raster data or contain only
    NoData values.

    Args:
        zonal_out: List of zonal statistics dictionaries from rasterstats.
                   Each dict contains keys like 'mean', 'min', 'max', etc.
        new_val: Replacement value for None entries (default: np.nan)

    Returns:
        List of cleaned statistics dictionaries with None values replaced

    Example:
        >>> stats = [{'mean': 150.5, 'max': 200}, {'mean': None, 'max': None}]
        >>> cleaned = check_zonal_stats_outcomes(stats, new_val=-9999)
        >>> cleaned[1]['mean']  # -9999 instead of None

    Note:
        - Common when HRUs partially overlap raster edges
        - Using np.nan allows easy filtering with pandas.dropna()
        - Using -9999 matches common hydrological missing value convention
        - Preserves dictionary structure and non-None values
    """
    cleaned = []
    for stats in zonal_out:
        cleaned_stats = {}
        for key, value in stats.items():
            if value is None:
                cleaned_stats[key] = new_val
            else:
                cleaned_stats[key] = value
        cleaned.append(cleaned_stats)
    return cleaned
