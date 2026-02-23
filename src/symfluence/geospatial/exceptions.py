"""
Geospatial-specific exception hierarchy for SYMFLUENCE.

Provides specific exception types for geospatial processing failures,
enabling more precise error handling and clearer error messages.

Exception Hierarchy:
    SYMFLUENCEError (from core.exceptions)
    └── GeospatialError
        ├── DelineationError
        │   ├── TauDEMError
        │   ├── GridCreationError
        │   └── SubsettingError
        ├── ShapefileError
        ├── RasterError
        └── TopologyError

Usage:
    from symfluence.geospatial.exceptions import (
        DelineationError,
        TauDEMError,
        geospatial_error_handler,
    )

    # Explicit exception handling
    try:
        result = delineator.delineate()
    except TauDEMError as e:
        logger.error(f"TauDEM failed: {e}")
        # Handle TauDEM-specific failure

    # Context manager for standardized handling
    with geospatial_error_handler("grid creation", logger, GridCreationError):
        grid = create_grid_from_bbox(...)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Type

from symfluence.core.exceptions import GeospatialError, ShapefileError

# =============================================================================
# Delineation-Specific Exceptions
# =============================================================================

class DelineationError(GeospatialError):
    """
    Base exception for domain delineation failures.

    Raised when watershed or grid delineation operations fail.

    Examples:
        - Pour point outside DEM extent
        - Invalid bounding box coordinates
        - Delineation method not supported
    """
    pass


class TauDEMError(DelineationError):
    """
    TauDEM command execution failures.

    Raised when TauDEM tools (pitremove, d8flowdir, aread8, etc.) fail.

    Examples:
        - TauDEM binary not found
        - TauDEM command returns non-zero exit code
        - DEM processing errors (corrupted data, wrong format)
        - MPI configuration errors
    """
    pass


class GridCreationError(DelineationError):
    """
    Grid creation failures.

    Raised when creating regular grid cells fails.

    Examples:
        - Invalid grid cell size
        - Bounding box parsing errors
        - Native grid forcing file not found
        - Grid clipping to watershed fails
    """
    pass


class SubsettingError(DelineationError):
    """
    Geofabric subsetting failures.

    Raised when extracting a domain from existing geofabric fails.

    Examples:
        - Geofabric files not found
        - Pour point doesn't intersect any basin
        - Upstream tracing fails
        - Invalid geofabric topology
    """
    pass


# =============================================================================
# File Operation Exceptions
# =============================================================================

class RasterError(GeospatialError):
    """
    Raster processing failures.

    Raised when reading, writing, or processing raster data fails.

    Examples:
        - DEM file not found or corrupted
        - Raster reprojection fails
        - Raster to polygon conversion fails
        - NoData value handling errors
    """
    pass


# =============================================================================
# Topology Exceptions
# =============================================================================

class TopologyError(GeospatialError):
    """
    River network topology failures.

    Raised when building or validating river network topology fails.

    Examples:
        - Disconnected river network
        - Invalid D8 flow directions
        - Circular references in upstream/downstream links
        - Missing LINKNO or DSLINKNO fields
    """
    pass


# =============================================================================
# Coordinate/CRS Exceptions
# =============================================================================

class CoordinateError(GeospatialError):
    """
    Coordinate system and transformation failures.

    Raised when coordinate operations fail.

    Examples:
        - Invalid CRS specification
        - Coordinate transformation fails
        - Coordinates outside valid range
        - UTM zone detection fails
    """
    pass


# =============================================================================
# Context Manager for Standardized Error Handling
# =============================================================================

@contextmanager
def geospatial_error_handler(
    operation: str,
    logger: Optional[logging.Logger] = None,
    error_type: Type[GeospatialError] = GeospatialError,
    reraise: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager for standardized geospatial error handling.

    Provides consistent error handling across geospatial operations:
    - Logs errors with operation context
    - Converts generic exceptions to specific geospatial types
    - Preserves exception chaining for debugging
    - Optionally suppresses re-raising for non-critical operations

    Args:
        operation: Description of the operation (for error messages)
        logger: Logger instance for error messages. If None, errors are not logged.
        error_type: GeospatialError subclass to convert generic exceptions to.
        reraise: Whether to re-raise exceptions after handling (default: True).

    Yields:
        None - just provides exception handling context

    Raises:
        The original exception if it's already a GeospatialError subclass,
        or error_type wrapping the original exception if not.

    Example:
        >>> # With specific error type
        >>> with geospatial_error_handler("grid creation", logger, GridCreationError):
        ...     grid = create_grid_from_bbox(bbox)

        >>> # Non-critical operation (don't reraise)
        >>> with geospatial_error_handler("cleanup", logger, reraise=False):
        ...     cleanup_intermediate_files()

        >>> # Explicit error handling
        >>> try:
        ...     with geospatial_error_handler("TauDEM pitremove", logger, TauDEMError):
        ...         run_pitremove(dem_path)
        ... except TauDEMError as e:
        ...     # Fallback to alternative method
        ...     use_alternative_pit_filling()
    """
    try:
        yield
    except GeospatialError:
        # Already a geospatial error, preserve it
        if logger:
            logger.error(f"Error during {operation}", exc_info=True)
        if reraise:
            raise
    except Exception as e:
        # Convert to specified geospatial error type
        if logger:
            logger.error(f"Error during {operation}: {e}", exc_info=True)
        if reraise:
            raise error_type(f"Failed during {operation}: {e}") from e


def handle_taudem_error(
    operation: str,
    exit_code: int,
    stderr: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Handle TauDEM command failure with informative error message.

    Args:
        operation: TauDEM operation name (e.g., 'pitremove', 'd8flowdir')
        exit_code: Command exit code
        stderr: Standard error output from command
        logger: Logger for error messages

    Raises:
        TauDEMError: Always raises with descriptive message
    """
    error_msg = f"TauDEM {operation} failed with exit code {exit_code}"
    if stderr:
        error_msg += f": {stderr[:500]}"  # Truncate long stderr

    if logger:
        logger.error(error_msg)

    raise TauDEMError(error_msg)


def validate_shapefile_columns(
    gdf,  # gpd.GeoDataFrame but not type hinted to avoid import
    required_columns: list,
    shapefile_name: str = "shapefile",
) -> None:
    """
    Validate that required columns exist in a GeoDataFrame.

    Args:
        gdf: GeoDataFrame to validate
        required_columns: List of required column names
        shapefile_name: Name for error messages

    Raises:
        ShapefileError: If any required column is missing
    """
    missing = [col for col in required_columns if col not in gdf.columns]
    if missing:
        raise ShapefileError(
            f"{shapefile_name} missing required columns: {missing}. "
            f"Available columns: {list(gdf.columns)}"
        )


def validate_raster_exists(
    raster_path,
    raster_name: str = "raster",
) -> None:
    """
    Validate that a raster file exists.

    Args:
        raster_path: Path to raster file
        raster_name: Name for error messages

    Raises:
        RasterError: If raster file doesn't exist
    """
    from pathlib import Path
    path = Path(raster_path)
    if not path.exists():
        raise RasterError(f"{raster_name} not found: {path}")


# =============================================================================
# Export all exceptions
# =============================================================================

__all__ = [
    # Exception classes
    'GeospatialError',
    'DelineationError',
    'TauDEMError',
    'GridCreationError',
    'SubsettingError',
    'ShapefileError',
    'RasterError',
    'TopologyError',
    'CoordinateError',
    # Helper functions
    'geospatial_error_handler',
    'handle_taudem_error',
    'validate_shapefile_columns',
    'validate_raster_exists',
]
