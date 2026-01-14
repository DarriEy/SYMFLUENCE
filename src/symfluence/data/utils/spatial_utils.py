"""
Spatial Subsetting Utilities

Consolidated functions for spatial operations on raster and xarray data:
- Rasterio windowed reading and cropping
- xarray spatial subsetting
- Coordinate system handling (0-360 vs -180-180)

Usage:
    from symfluence.data.utils.spatial_utils import crop_raster_to_bbox, subset_xarray

    # Raster operations
    crop_raster_to_bbox(input_path, output_path, bbox)
    data, transform, meta = read_raster_window(raster_path, bbox)

    # xarray operations
    ds_subset = subset_xarray_to_bbox(ds, bbox, lat_dim='latitude', lon_dim='longitude')
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Type alias for bounding box
BBox = Dict[str, float]  # {'lat_min': ..., 'lat_max': ..., 'lon_min': ..., 'lon_max': ...}


def validate_bbox(
    bbox: BBox,
    context: str = "spatial operation",
    comprehensive: bool = True
) -> BBox:
    """
    Validate bounding box has required keys and valid values.

    Args:
        bbox: Bounding box dictionary
        context: Context for error messages
        comprehensive: If True, validate ranges and warn about edge cases

    Returns:
        The validated bbox

    Raises:
        ValueError: If bbox is invalid
    """
    if comprehensive:
        from symfluence.core.validation import validate_bounding_box
        return validate_bounding_box(bbox, context=context, logger=logger)
    else:
        # Basic validation only (keys)
        required = {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        missing = required - set(bbox.keys())
        if missing:
            raise ValueError(f"Bounding box missing keys: {missing}")
        return bbox


# =============================================================================
# Rasterio-based spatial operations
# =============================================================================

def crop_raster_to_bbox(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    bbox: BBox,
    compress: str = 'lzw'
) -> Path:
    """
    Crop a raster to a bounding box and save to a new file.

    Args:
        source_path: Path to source raster
        output_path: Path to output cropped raster
        bbox: Bounding box dict with lat_min, lat_max, lon_min, lon_max
        compress: Compression algorithm for output (default: 'lzw')

    Returns:
        Path to output raster

    Example:
        >>> bbox = {'lat_min': 40.0, 'lat_max': 41.0, 'lon_min': -112.0, 'lon_max': -111.0}
        >>> crop_raster_to_bbox('input.tif', 'output.tif', bbox)
    """
    import rasterio
    from rasterio.windows import from_bounds

    validate_bbox(bbox)
    source_path = Path(source_path)
    output_path = Path(output_path)

    with rasterio.open(source_path) as src:
        window = from_bounds(
            bbox['lon_min'],
            bbox['lat_min'],
            bbox['lon_max'],
            bbox['lat_max'],
            src.transform
        )

        data = src.read(1, window=window)
        meta = src.meta.copy()
        meta.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': src.window_transform(window),
            'compress': compress
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)

    return output_path


def read_raster_window(
    raster_path: Union[str, Path],
    bbox: BBox,
    band: int = 1,
    padding: float = 0.0
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Read a window of a raster based on bounding box.

    Args:
        raster_path: Path to raster file
        bbox: Bounding box dict
        band: Band number to read (default: 1)
        padding: Padding to add around bbox in degrees (default: 0.0)

    Returns:
        Tuple of (data array, transform, metadata dict)

    Example:
        >>> data, transform, meta = read_raster_window('dem.tif', bbox, padding=0.001)
    """
    import rasterio
    from rasterio.windows import from_bounds

    validate_bbox(bbox)

    with rasterio.open(raster_path) as src:
        window = from_bounds(
            bbox['lon_min'] - padding,
            bbox['lat_min'] - padding,
            bbox['lon_max'] + padding,
            bbox['lat_max'] + padding,
            src.transform
        )

        data = src.read(band, window=window)
        transform = src.window_transform(window)
        nodata = src.nodata if src.nodata is not None else -9999

        meta = {
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': transform,
            'nodata': nodata,
            'crs': src.crs,
            'dtype': src.dtypes[band - 1]
        }

    return data, transform, meta


def read_raster_multiband_window(
    raster_path: Union[str, Path],
    bbox: BBox,
    bands: Optional[list] = None,
    padding: float = 0.0
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Read multiple bands from a raster window.

    Args:
        raster_path: Path to raster file
        bbox: Bounding box dict
        bands: List of band numbers to read (default: all bands)
        padding: Padding around bbox in degrees

    Returns:
        Tuple of (data array [bands, rows, cols], transform, metadata)
    """
    import rasterio
    from rasterio.windows import from_bounds

    validate_bbox(bbox)

    with rasterio.open(raster_path) as src:
        window = from_bounds(
            bbox['lon_min'] - padding,
            bbox['lat_min'] - padding,
            bbox['lon_max'] + padding,
            bbox['lat_max'] + padding,
            src.transform
        )

        if bands is None:
            bands = list(range(1, src.count + 1))

        data = src.read(bands, window=window)
        transform = src.window_transform(window)

        meta = {
            'height': data.shape[1],
            'width': data.shape[2],
            'count': len(bands),
            'transform': transform,
            'nodata': src.nodata,
            'crs': src.crs,
            'dtype': src.dtypes[0]
        }

    return data, transform, meta


# =============================================================================
# xarray-based spatial operations
# =============================================================================

def create_spatial_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: BBox,
    lon_convention: str = 'standard'
) -> np.ndarray:
    """
    Create a 2D boolean mask for a bounding box.

    Args:
        lat: Latitude array (1D or 2D)
        lon: Longitude array (1D or 2D)
        bbox: Bounding box dict
        lon_convention: 'standard' (-180 to 180) or 'positive' (0 to 360)

    Returns:
        Boolean mask array

    Example:
        >>> mask = create_spatial_mask(ds.latitude.values, ds.longitude.values, bbox)
    """
    validate_bbox(bbox)

    # Create 2D arrays if 1D
    if lat.ndim == 1 and lon.ndim == 1:
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    else:
        lat_2d, lon_2d = lat, lon

    # Handle longitude convention
    if lon_convention == 'positive':
        # Convert bbox from standard to 0-360
        lon_min = bbox['lon_min'] % 360
        lon_max = bbox['lon_max'] % 360

        # Handle wrapping at 0/360 boundary
        if lon_min > lon_max:
            lon_mask = (lon_2d >= lon_min) | (lon_2d <= lon_max)
        else:
            lon_mask = (lon_2d >= lon_min) & (lon_2d <= lon_max)
    else:
        # Standard -180 to 180
        lon_mask = (lon_2d >= bbox['lon_min']) & (lon_2d <= bbox['lon_max'])

    lat_mask = (lat_2d >= bbox['lat_min']) & (lat_2d <= bbox['lat_max'])

    return lat_mask & lon_mask


def subset_xarray_to_bbox(
    ds,  # xr.Dataset or xr.DataArray
    bbox: BBox,
    lat_dim: str = 'latitude',
    lon_dim: str = 'longitude',
    buffer: int = 0,
    lon_convention: str = 'standard'
):
    """
    Subset an xarray Dataset/DataArray to a bounding box.

    Handles both:
    - 1D regular grids (uses sel())
    - 2D irregular grids (uses isel() with masking)

    Args:
        ds: xarray Dataset or DataArray
        bbox: Bounding box dict
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension
        buffer: Extra grid cells to include around the bbox
        lon_convention: 'standard' (-180 to 180) or 'positive' (0 to 360)

    Returns:
        Subsetted Dataset/DataArray

    Example:
        >>> ds_subset = subset_xarray_to_bbox(ds, bbox, lat_dim='lat', lon_dim='lon')
    """
    import xarray as xr

    validate_bbox(bbox)

    # Check if coordinates are 1D (regular grid)
    if lat_dim in ds.dims and lon_dim in ds.dims:
        lat = ds[lat_dim].values
        lon = ds[lon_dim].values

        if lat.ndim == 1 and lon.ndim == 1:
            # 1D regular grid - use efficient sel()
            return _subset_regular_grid(ds, bbox, lat_dim, lon_dim, lon_convention)

    # 2D irregular grid - use masking approach
    return _subset_irregular_grid(ds, bbox, lat_dim, lon_dim, buffer, lon_convention)


def _subset_regular_grid(
    ds,
    bbox: BBox,
    lat_dim: str,
    lon_dim: str,
    lon_convention: str
):
    """Subset a regular grid using xarray sel()."""
    lat = ds[lat_dim].values
    lon = ds[lon_dim].values

    # Handle longitude convention
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    if lon_convention == 'positive':
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    # Determine if latitude is ascending or descending
    lat_ascending = lat[0] < lat[-1] if len(lat) > 1 else True

    if lat_ascending:
        lat_sel = slice(bbox['lat_min'], bbox['lat_max'])
    else:
        lat_sel = slice(bbox['lat_max'], bbox['lat_min'])

    # Handle longitude selection
    if lon_convention == 'positive' and lon_min > lon_max:
        # Need to handle wraparound
        ds1 = ds.sel({lon_dim: slice(lon_min, 360)})
        ds2 = ds.sel({lon_dim: slice(0, lon_max)})
        import xarray as xr
        ds = xr.concat([ds1, ds2], dim=lon_dim)
        ds = ds.sel({lat_dim: lat_sel})
    else:
        ds = ds.sel({lat_dim: lat_sel, lon_dim: slice(lon_min, lon_max)})

    return ds


def _subset_irregular_grid(
    ds,
    bbox: BBox,
    lat_dim: str,
    lon_dim: str,
    buffer: int,
    lon_convention: str
):
    """Subset an irregular 2D grid using isel() with masking."""
    # Get 2D lat/lon arrays
    lat = ds['latitude'].values if 'latitude' in ds else ds[lat_dim].values
    lon = ds['longitude'].values if 'longitude' in ds else ds[lon_dim].values

    # Create mask
    mask = create_spatial_mask(lat, lon, bbox, lon_convention)

    # Find indices
    indices = np.where(mask)
    if len(indices) < 2 or len(indices[0]) == 0:
        logger.warning(f"No grid points found in bbox {bbox}")
        return ds

    y_idx, x_idx = indices

    # Determine dimension names
    # Common patterns: y/x, rlat/rlon, lat/lon
    if 'y' in ds.dims:
        y_dim, x_dim = 'y', 'x'
    elif 'rlat' in ds.dims:
        y_dim, x_dim = 'rlat', 'rlon'
    else:
        y_dim, x_dim = lat_dim, lon_dim

    if y_dim not in ds.dims or x_dim not in ds.dims:
        logger.warning(f"Could not find spatial dimensions in {list(ds.dims)}")
        return ds

    # Apply buffer and bounds checking
    y_min = max(0, y_idx.min() - buffer)
    y_max = min(ds.dims[y_dim] - 1, y_idx.max() + buffer)
    x_min = max(0, x_idx.min() - buffer)
    x_max = min(ds.dims[x_dim] - 1, x_idx.max() + buffer)

    return ds.isel({y_dim: slice(y_min, y_max + 1), x_dim: slice(x_min, x_max + 1)})


def normalize_longitude(lon: np.ndarray, convention: str = 'standard') -> np.ndarray:
    """
    Normalize longitude values to a specific convention.

    Args:
        lon: Longitude array
        convention: 'standard' (-180 to 180) or 'positive' (0 to 360)

    Returns:
        Normalized longitude array
    """
    if convention == 'standard':
        # Convert 0-360 to -180-180
        return np.where(lon > 180, lon - 360, lon)
    else:
        # Convert -180-180 to 0-360
        return np.where(lon < 0, lon + 360, lon)


# =============================================================================
# Convenience classes (Mixins)
# =============================================================================

class SpatialSubsetMixin:
    """
    Mixin providing spatial subsetting capabilities.

    Requires:
        - self.bbox: Dict with lat_min, lat_max, lon_min, lon_max
        - self.logger: Logger instance (optional)

    Usage:
        class MyHandler(BaseHandler, SpatialSubsetMixin):
            def process(self):
                data, transform, meta = self.read_raster_window('dem.tif')
                ds = self.subset_xarray(ds)
    """

    def crop_raster(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        compress: str = 'lzw'
    ) -> Path:
        """Crop raster to instance's bbox."""
        return crop_raster_to_bbox(source_path, output_path, self.bbox, compress)

    def read_raster_window(
        self,
        raster_path: Union[str, Path],
        band: int = 1,
        padding: float = 0.0
    ) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """Read raster window using instance's bbox."""
        return read_raster_window(raster_path, self.bbox, band, padding)

    def subset_xarray(
        self,
        ds,
        lat_dim: str = 'latitude',
        lon_dim: str = 'longitude',
        buffer: int = 0
    ):
        """Subset xarray data to instance's bbox."""
        lon_convention = getattr(self, 'lon_convention', 'standard')
        return subset_xarray_to_bbox(ds, self.bbox, lat_dim, lon_dim, buffer, lon_convention)

    def create_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray
    ) -> np.ndarray:
        """Create spatial mask using instance's bbox."""
        lon_convention = getattr(self, 'lon_convention', 'standard')
        return create_spatial_mask(lat, lon, self.bbox, lon_convention)
