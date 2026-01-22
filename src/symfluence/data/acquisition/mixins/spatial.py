"""
Spatial Subset Mixin for Data Acquisition Handlers.

Provides utilities for spatial subsetting of datasets:
- xarray Dataset subsetting by bounding box
- Rasterio window-based subsetting for GeoTIFFs
- Bbox format conversions for various APIs
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import xarray as xr


class SpatialSubsetMixin:
    """
    Mixin for spatial subsetting operations.

    Provides methods for:
    - Finding coordinate names in datasets
    - Subsetting xarray Datasets by bounding box
    - Subsetting GeoTIFFs using rasterio windows
    - Converting bbox to various API formats

    Expects the class to have:
    - self.bbox: Dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    - self.logger: logging.Logger instance
    """

    def get_coord_names(
        self,
        ds: xr.Dataset,
        lat_candidates: Tuple[str, ...] = ("lat", "latitude", "y", "rlat"),
        lon_candidates: Tuple[str, ...] = ("lon", "longitude", "x", "rlon")
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find latitude and longitude coordinate names in a dataset.

        Args:
            ds: xarray Dataset to search
            lat_candidates: Possible names for latitude coordinate
            lon_candidates: Possible names for longitude coordinate

        Returns:
            Tuple of (lat_name, lon_name), either may be None if not found
        """
        lat_name = None
        lon_name = None

        # Check both coords and sizes (dims is deprecated for returning keys)
        all_names = set(ds.coords.keys()) | set(ds.sizes.keys())

        for candidate in lat_candidates:
            if candidate in all_names:
                lat_name = candidate
                break

        for candidate in lon_candidates:
            if candidate in all_names:
                lon_name = candidate
                break

        return lat_name, lon_name

    def subset_xarray_bbox(
        self,
        ds: xr.Dataset,
        bbox: Dict[str, float] = None,
        lat_name: str = None,
        lon_name: str = None,
        handle_lon_wrap: bool = True,
        time_slice: Tuple[Any, Any] = None,
        buffer: float = 0.0
    ) -> xr.Dataset:
        """
        Subset an xarray Dataset using a bounding box.

        Handles various coordinate conventions and longitude wrapping.

        Args:
            ds: xarray Dataset to subset
            bbox: Bounding box dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
                  (uses self.bbox if not provided)
            lat_name: Name of latitude coordinate (auto-detected if not provided)
            lon_name: Name of longitude coordinate (auto-detected if not provided)
            handle_lon_wrap: If True, handle 0-360 vs -180-180 longitude conventions
            time_slice: Optional (start, end) tuple for time subsetting
            buffer: Buffer in degrees to add around bbox (default: 0)

        Returns:
            Subsetted xarray Dataset

        Raises:
            ValueError: If coordinates cannot be found
        """
        logger = getattr(self, 'logger', logging.getLogger(__name__))

        if bbox is None:
            bbox = getattr(self, 'bbox', None)
            if bbox is None:
                raise ValueError("No bounding box provided")

        # Auto-detect coordinate names if not provided
        if lat_name is None or lon_name is None:
            detected_lat, detected_lon = self.get_coord_names(ds)
            lat_name = lat_name or detected_lat
            lon_name = lon_name or detected_lon

        if not lat_name or not lon_name:
            raise ValueError(
                f"Could not find lat/lon coordinates. "
                f"Available: {list(ds.coords.keys())}"
            )

        # Extract and sort bbox values with buffer
        lat_min = min(bbox['lat_min'], bbox['lat_max']) - buffer
        lat_max = max(bbox['lat_min'], bbox['lat_max']) + buffer
        lon_min = min(bbox['lon_min'], bbox['lon_max']) - buffer
        lon_max = max(bbox['lon_min'], bbox['lon_max']) + buffer

        # Handle longitude convention (0-360 vs -180-180)
        if handle_lon_wrap:
            lon_vals = ds[lon_name].values
            if lon_vals.max() > 180 and (lon_min < 0 or lon_max < 0):
                # Dataset uses 0-360, query uses -180-180
                lon_min = lon_min % 360
                lon_max = lon_max % 360
                logger.debug(f"Converted lon range to 0-360: [{lon_min}, {lon_max}]")
            elif lon_vals.min() < 0 and (lon_min > 180 or lon_max > 180):
                # Dataset uses -180-180, query uses 0-360
                lon_min = lon_min - 360 if lon_min > 180 else lon_min
                lon_max = lon_max - 360 if lon_max > 180 else lon_max
                logger.debug(f"Converted lon range to -180-180: [{lon_min}, {lon_max}]")

        # Perform subsetting
        # Note: xarray slice is inclusive on both ends
        try:
            # Check if latitude is ascending or descending
            lat_vals = ds[lat_name].values
            if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
                # Descending latitude (common in climate data)
                subset = ds.sel({
                    lat_name: slice(lat_max, lat_min),
                    lon_name: slice(lon_min, lon_max)
                })
            else:
                # Ascending latitude
                subset = ds.sel({
                    lat_name: slice(lat_min, lat_max),
                    lon_name: slice(lon_min, lon_max)
                })

            # Handle longitude wraparound (e.g., Pacific crossing dateline)
            if handle_lon_wrap and lon_min > lon_max:
                # Need to concatenate two slices
                subset1 = ds.sel({lon_name: slice(lon_min, lon_vals.max())})
                subset2 = ds.sel({lon_name: slice(lon_vals.min(), lon_max)})
                subset = xr.concat([subset1, subset2], dim=lon_name)
                if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
                    subset = subset.sel({lat_name: slice(lat_max, lat_min)})
                else:
                    subset = subset.sel({lat_name: slice(lat_min, lat_max)})

        except Exception as e:
            logger.warning(f"Standard slicing failed: {e}. Trying sel with method='nearest'")
            # Fallback for irregular grids
            subset = ds.sel(
                {lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)},
                method='nearest'
            )

        # Apply time slice if specified
        if time_slice is not None and 'time' in subset.coords:
            start, end = time_slice
            subset = subset.sel(time=slice(start, end))

        return subset

    def subset_numpy_mask(
        self,
        ds: xr.Dataset,
        bbox: Dict[str, float] = None,
        lat_name: str = 'lat',
        lon_name: str = 'lon',
        grid_dims: Tuple[str, str] = None,
        buffer_cells: int = 0
    ) -> xr.Dataset:
        """
        Subset using NumPy boolean mask (for rotated/curvilinear grids).

        Use this when standard sel() doesn't work due to non-rectilinear grids.

        Args:
            ds: xarray Dataset to subset
            bbox: Bounding box dict
            lat_name: Name of 2D latitude variable
            lon_name: Name of 2D longitude variable
            grid_dims: Tuple of (y_dim, x_dim) names for the grid
            buffer_cells: Number of grid cells to add as buffer

        Returns:
            Subsetted xarray Dataset
        """
        logger = getattr(self, 'logger', logging.getLogger(__name__))

        if bbox is None:
            bbox = getattr(self, 'bbox', None)

        lat_min = min(bbox['lat_min'], bbox['lat_max'])
        lat_max = max(bbox['lat_min'], bbox['lat_max'])
        lon_min = min(bbox['lon_min'], bbox['lon_max'])
        lon_max = max(bbox['lon_min'], bbox['lon_max'])

        # Create boolean mask
        mask = (
            (ds[lat_name] >= lat_min) & (ds[lat_name] <= lat_max) &
            (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max)
        )

        # Find bounding indices
        y_indices, x_indices = np.where(mask.values)

        if len(y_indices) == 0 or len(x_indices) == 0:
            logger.warning("No grid cells found within bounding box")
            return ds

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Apply buffer
        if grid_dims:
            y_dim, x_dim = grid_dims
            y_min = max(0, y_min - buffer_cells)
            y_max = min(ds.sizes[y_dim] - 1, y_max + buffer_cells)
            x_min = max(0, x_min - buffer_cells)
            x_max = min(ds.sizes[x_dim] - 1, x_max + buffer_cells)

            subset = ds.isel({
                y_dim: slice(y_min, y_max + 1),
                x_dim: slice(x_min, x_max + 1)
            })
        else:
            # Try to infer dims from mask shape
            dims = mask.dims
            subset = ds.isel({
                dims[0]: slice(y_min, y_max + 1),
                dims[1]: slice(x_min, x_max + 1)
            })

        return subset

    def subset_rasterio_window(
        self,
        src_path: Path,
        output_path: Path,
        bbox: Dict[str, float] = None,
        **write_kwargs
    ) -> Path:
        """
        Subset a GeoTIFF using rasterio window operations.

        Args:
            src_path: Path to input GeoTIFF
            output_path: Path for output subset
            bbox: Bounding box dict (uses self.bbox if not provided)
            **write_kwargs: Additional arguments for rasterio write (e.g., compress='lzw')

        Returns:
            Path to the subsetted output file
        """
        import rasterio
        from rasterio.windows import from_bounds

        if bbox is None:
            bbox = getattr(self, 'bbox', None)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(src_path) as src:
            # Calculate window from bbox
            window = from_bounds(
                bbox['lon_min'],
                bbox['lat_min'],
                bbox['lon_max'],
                bbox['lat_max'],
                src.transform
            )

            # Read windowed data
            data = src.read(window=window)

            # Update metadata
            meta = src.meta.copy()
            meta.update({
                'height': data.shape[1],
                'width': data.shape[2],
                'transform': src.window_transform(window),
            })
            meta.update(write_kwargs)

            # Write output
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data)

        return output_path

    def bbox_to_geojson(self, bbox: Dict[str, float] = None) -> Dict:
        """
        Convert bounding box to GeoJSON polygon.

        Useful for APIs that accept GeoJSON (AppEEARS, CMR).

        Args:
            bbox: Bounding box dict (uses self.bbox if not provided)

        Returns:
            GeoJSON FeatureCollection with polygon
        """
        if bbox is None:
            bbox = getattr(self, 'bbox', None)

        lat_min = min(bbox['lat_min'], bbox['lat_max'])
        lat_max = max(bbox['lat_min'], bbox['lat_max'])
        lon_min = min(bbox['lon_min'], bbox['lon_max'])
        lon_max = max(bbox['lon_min'], bbox['lon_max'])

        coordinates = [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]  # Close the polygon
        ]]

        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {}
            }]
        }

    def bbox_to_cds_area(self, bbox: Dict[str, float] = None) -> List[float]:
        """
        Convert bounding box to CDS area format [North, West, South, East].

        Args:
            bbox: Bounding box dict (uses self.bbox if not provided)

        Returns:
            List in CDS format: [lat_max, lon_min, lat_min, lon_max]
        """
        if bbox is None:
            bbox = getattr(self, 'bbox', None)

        return [
            max(bbox['lat_min'], bbox['lat_max']),  # North
            min(bbox['lon_min'], bbox['lon_max']),  # West
            min(bbox['lat_min'], bbox['lat_max']),  # South
            max(bbox['lon_min'], bbox['lon_max']),  # East
        ]

    def bbox_to_wcs_params(
        self,
        bbox: Dict[str, float] = None,
        crs: str = "EPSG/0/4326"
    ) -> List[Tuple[str, str]]:
        """
        Generate WCS SUBSET parameters from bounding box.

        Args:
            bbox: Bounding box dict (uses self.bbox if not provided)
            crs: Coordinate reference system (default: WGS84)

        Returns:
            List of (key, value) tuples for URL parameters
        """
        if bbox is None:
            bbox = getattr(self, 'bbox', None)

        lat_min = min(bbox['lat_min'], bbox['lat_max'])
        lat_max = max(bbox['lat_min'], bbox['lat_max'])
        lon_min = min(bbox['lon_min'], bbox['lon_max'])
        lon_max = max(bbox['lon_min'], bbox['lon_max'])

        return [
            ("SUBSETTINGCRS", f"http://www.opengis.net/def/crs/{crs}"),
            ("OUTPUTCRS", f"http://www.opengis.net/def/crs/{crs}"),
            ("SUBSET", f"Lat({lat_min},{lat_max})"),
            ("SUBSET", f"Lon({lon_min},{lon_max})")
        ]


__all__ = ['SpatialSubsetMixin']
