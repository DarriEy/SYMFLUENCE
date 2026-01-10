"""
Utility functions for attribute processing.

Helper functions for:
- Raster clipping
- Pixel counting
- Data validation
- I/O operations
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union

# Re-export from consolidated spatial utilities
from symfluence.data.utilities.spatial_utils import crop_raster_to_bbox


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
    Check and clean zonal statistics outcomes.

    Args:
        zonal_out: List of zonal statistics dictionaries
        new_val: Value to replace None with

    Returns:
        Cleaned zonal statistics
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
