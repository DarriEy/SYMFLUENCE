"""
WMFire Fire Grid Module

Provides georeferenced grid management for WMFire fire spread modeling.
Handles creation of patch and DEM grids with proper spatial registration.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FireGrid:
    """
    Container for fire spread grid data with geospatial metadata.

    Stores grid data along with affine transform, CRS, and resolution
    for proper georeferencing. Supports export to both RHESSys text format
    and GeoTIFF for visualization.

    Attributes:
        data: 2D numpy array of grid values (patch IDs or elevations)
        transform: Affine transform tuple (a, b, c, d, e, f) for geolocation
        crs: Coordinate reference system string (e.g., 'EPSG:32610')
        resolution: Grid cell resolution in meters
        nodata: NoData value for the grid
    """
    data: np.ndarray
    transform: Tuple[float, float, float, float, float, float]
    crs: str
    resolution: float
    nodata: float = -9999.0

    @property
    def nrows(self) -> int:
        """Number of rows in the grid."""
        return self.data.shape[0]

    @property
    def ncols(self) -> int:
        """Number of columns in the grid."""
        return self.data.shape[1]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Grid bounds as (minx, miny, maxx, maxy)."""
        a, b, c, d, e, f = self.transform
        minx = c
        maxy = f
        maxx = minx + a * self.ncols
        miny = maxy + e * self.nrows
        return (minx, miny, maxx, maxy)

    def to_text(self, include_header: bool = False) -> str:
        """
        Convert grid to RHESSys text format.

        RHESSys expects tab-separated values without headers by default.
        The grid is written row by row from north to south.

        Args:
            include_header: If True, include ESRI ASCII grid header

        Returns:
            String representation in RHESSys text format
        """
        lines = []

        if include_header:
            minx, miny, _, _ = self.bounds
            lines.extend([
                f"ncols {self.ncols}",
                f"nrows {self.nrows}",
                f"xllcorner {minx:.6f}",
                f"yllcorner {miny:.6f}",
                f"cellsize {self.resolution}",
                f"NODATA_value {self.nodata}",
            ])

        # Write data rows (tab-separated)
        for row in self.data:
            # Format based on data type
            if np.issubdtype(self.data.dtype, np.integer):
                line = '\t'.join(str(int(v)) for v in row)
            else:
                line = '\t'.join(f'{v:.1f}' for v in row)
            lines.append(line)

        return '\n'.join(lines) + '\n'

    def to_geotiff(self, path: Union[str, Path], compress: str = 'lzw') -> None:
        """
        Write grid to GeoTIFF file with georeferencing metadata.

        Args:
            path: Output file path
            compress: Compression method ('lzw', 'deflate', or None)

        Raises:
            ImportError: If rasterio is not available
        """
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            logger.warning("rasterio not available, skipping GeoTIFF export")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine data type for output
        if np.issubdtype(self.data.dtype, np.integer):
            dtype = 'int32'
        else:
            dtype = 'float32'

        # Create affine transform from tuple
        transform = Affine(*self.transform)

        # Write GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': dtype,
            'width': self.ncols,
            'height': self.nrows,
            'count': 1,
            'crs': self.crs,
            'transform': transform,
            'nodata': self.nodata,
        }

        if compress:
            profile['compress'] = compress

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(self.data.astype(dtype), 1)

        logger.info(f"GeoTIFF written: {path} ({self.nrows}x{self.ncols}, {self.crs})")


class FireGridManager:
    """
    Creates georeferenced fire grids from catchment geometry and DEM.

    Handles all aspects of fire grid creation including:
    - CRS estimation and reprojection
    - Grid extent calculation from catchment bounds
    - Patch ID rasterization
    - DEM resampling to target resolution
    """

    def __init__(self, config, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the FireGridManager.

        Args:
            config: SymfluenceConfig object with domain and model settings
            logger_instance: Optional logger for status messages
        """
        self.config = config
        self.logger = logger_instance or logger

        # Get resolution from config
        self._resolution = self._get_resolution()

    def _get_resolution(self) -> int:
        """Get grid resolution from WMFire config or use default."""
        try:
            if (hasattr(self.config, 'model') and
                hasattr(self.config.model, 'rhessys') and
                self.config.model.rhessys is not None):
                rhessys = self.config.model.rhessys
                if hasattr(rhessys, 'wmfire') and rhessys.wmfire is not None:
                    return rhessys.wmfire.grid_resolution
        except AttributeError:
            pass
        return 30  # Default resolution

    @property
    def resolution(self) -> int:
        """Grid resolution in meters."""
        return self._resolution

    def create_fire_grid(
        self,
        catchment_gdf,
        dem_path: Optional[Path] = None
    ) -> Tuple[FireGrid, FireGrid]:
        """
        Create georeferenced patch and DEM grids for fire modeling.

        Args:
            catchment_gdf: GeoDataFrame with catchment/HRU polygons
            dem_path: Optional path to DEM raster for elevation data

        Returns:
            Tuple of (patch_grid, dem_grid) as FireGrid objects
        """
        import geopandas as gpd
        from shapely.validation import make_valid

        # Ensure we have a GeoDataFrame
        if not isinstance(catchment_gdf, gpd.GeoDataFrame):
            raise TypeError("catchment_gdf must be a GeoDataFrame")

        # Fix invalid geometries
        if not catchment_gdf.is_valid.all():
            self.logger.info(f"Fixing {(~catchment_gdf.is_valid).sum()} invalid geometries")
            catchment_gdf = catchment_gdf.copy()
            catchment_gdf['geometry'] = catchment_gdf['geometry'].apply(
                lambda g: make_valid(g) if g is not None and not g.is_valid else g
            )

        # Check if already in a projected CRS
        if catchment_gdf.crs is not None and not catchment_gdf.crs.is_geographic:
            # Already in a projected CRS, use it directly
            utm_crs = str(catchment_gdf.crs)
            self.logger.info(f"Using existing projected CRS: {utm_crs}")
        else:
            # Get centroid for CRS estimation (geographic coordinates)
            centroid = catchment_gdf.geometry.unary_union.centroid
            lon, lat = centroid.x, centroid.y

            # Estimate UTM CRS for accurate area/distance calculations
            utm_crs = self._estimate_utm_crs(lat, lon)
            self.logger.info(f"Using UTM CRS: {utm_crs}")

        # Reproject to UTM
        gdf_utm = catchment_gdf.to_crs(utm_crs)

        # Calculate grid extent from bounds
        minx, miny, maxx, maxy = gdf_utm.total_bounds
        self.logger.info(f"Catchment bounds (UTM): {minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f}")

        # Add buffer and snap to resolution
        buffer = self.resolution * 2
        minx = np.floor((minx - buffer) / self.resolution) * self.resolution
        miny = np.floor((miny - buffer) / self.resolution) * self.resolution
        maxx = np.ceil((maxx + buffer) / self.resolution) * self.resolution
        maxy = np.ceil((maxy + buffer) / self.resolution) * self.resolution

        # Calculate grid dimensions
        ncols = int((maxx - minx) / self.resolution)
        nrows = int((maxy - miny) / self.resolution)

        self.logger.info(f"Fire grid dimensions: {nrows} rows x {ncols} cols @ {self.resolution}m")

        # Create affine transform (north-up)
        # Format: (pixel_width, row_rotation, x_origin, col_rotation, pixel_height, y_origin)
        transform = (self.resolution, 0.0, minx, 0.0, -self.resolution, maxy)

        # Rasterize patches
        patch_data = self._rasterize_patches(gdf_utm, nrows, ncols, transform)
        patch_grid = FireGrid(
            data=patch_data,
            transform=transform,
            crs=utm_crs,
            resolution=self.resolution,
            nodata=-9999
        )

        # Create or resample DEM grid
        if dem_path and Path(dem_path).exists():
            dem_data = self._resample_dem(dem_path, nrows, ncols, transform, utm_crs)
        else:
            # Create synthetic DEM from HRU attributes if available
            dem_data = self._create_synthetic_dem(gdf_utm, patch_data, nrows, ncols)

        dem_grid = FireGrid(
            data=dem_data,
            transform=transform,
            crs=utm_crs,
            resolution=self.resolution,
            nodata=-9999.0
        )

        return patch_grid, dem_grid

    def _estimate_utm_crs(self, lat: float, lon: float) -> str:
        """
        Estimate appropriate UTM CRS from latitude/longitude.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            CRS string (e.g., 'EPSG:32610')
        """
        # Calculate UTM zone
        utm_zone = int((lon + 180) / 6) + 1

        # Determine hemisphere
        if lat >= 0:
            epsg = 32600 + utm_zone  # Northern hemisphere
        else:
            epsg = 32700 + utm_zone  # Southern hemisphere

        return f"EPSG:{epsg}"

    def _rasterize_patches(
        self,
        gdf,
        nrows: int,
        ncols: int,
        transform: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Rasterize patch polygons to grid.

        Args:
            gdf: GeoDataFrame with patch polygons
            nrows: Number of grid rows
            ncols: Number of grid columns
            transform: Affine transform tuple

        Returns:
            2D numpy array of patch IDs
        """
        try:
            from rasterio.features import rasterize
            from rasterio.transform import Affine
        except ImportError:
            self.logger.warning("rasterio not available, using fallback rasterization")
            return self._rasterize_patches_fallback(gdf, nrows, ncols, transform)

        # Get patch ID column
        id_col = None
        for col in ['HRU_ID', 'hru_id', 'patch_id', 'PATCH_ID', 'ID', 'id']:
            if col in gdf.columns:
                id_col = col
                break

        if id_col is None:
            # Use index as ID
            gdf = gdf.copy()
            gdf['_patch_id'] = range(1, len(gdf) + 1)
            id_col = '_patch_id'

        # Create geometry-value pairs for rasterization
        shapes = [(geom, int(pid)) for geom, pid in zip(gdf.geometry, gdf[id_col])]

        # Rasterize
        patch_grid = rasterize(
            shapes=shapes,
            out_shape=(nrows, ncols),
            transform=Affine(*transform),
            fill=0,
            dtype='int32'
        )

        num_patches = len(np.unique(patch_grid)) - 1  # Exclude 0
        self.logger.info(f"Rasterized {num_patches} patches to grid")

        return patch_grid

    def _rasterize_patches_fallback(
        self,
        gdf,
        nrows: int,
        ncols: int,
        transform: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Fallback rasterization using point sampling.

        Used when rasterio is not available.
        """
        from shapely.geometry import Point

        a, _, c, _, e, f = transform
        patch_grid = np.zeros((nrows, ncols), dtype='int32')

        # Get patch ID column
        id_col = None
        for col in ['HRU_ID', 'hru_id', 'patch_id', 'PATCH_ID', 'ID', 'id']:
            if col in gdf.columns:
                id_col = col
                break

        if id_col is None:
            gdf = gdf.copy()
            gdf['_patch_id'] = range(1, len(gdf) + 1)
            id_col = '_patch_id'

        # Sample each cell center
        for row in range(nrows):
            for col in range(ncols):
                x = c + (col + 0.5) * a
                y = f + (row + 0.5) * e
                pt = Point(x, y)

                for idx, geom in enumerate(gdf.geometry):
                    if geom.contains(pt):
                        patch_grid[row, col] = int(gdf[id_col].iloc[idx])  # type: ignore[index]
                        break

        return patch_grid

    def _resample_dem(
        self,
        dem_path: Path,
        nrows: int,
        ncols: int,
        transform: Tuple[float, ...],
        target_crs: str
    ) -> np.ndarray:
        """
        Resample DEM to target grid resolution and extent.

        Args:
            dem_path: Path to source DEM
            nrows: Target number of rows
            ncols: Target number of columns
            transform: Target affine transform
            target_crs: Target CRS

        Returns:
            Resampled DEM as 2D numpy array
        """
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import Affine
        except ImportError:
            self.logger.warning("rasterio not available, using constant DEM")
            return np.full((nrows, ncols), 1500.0, dtype='float32')

        dem_path = Path(dem_path)

        with rasterio.open(dem_path) as src:
            # Create output array
            dem_data = np.zeros((nrows, ncols), dtype='float32')

            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=dem_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=Affine(*transform),
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

        # Handle nodata
        dem_data[dem_data < -1000] = np.nan
        dem_data[np.isnan(dem_data)] = np.nanmean(dem_data)

        self.logger.info(f"DEM resampled: min={dem_data.min():.0f}m, max={dem_data.max():.0f}m")

        return dem_data

    def _create_synthetic_dem(
        self,
        gdf,
        patch_grid: np.ndarray,
        nrows: int,
        ncols: int
    ) -> np.ndarray:
        """
        Create synthetic DEM from HRU elevation attributes.

        Args:
            gdf: GeoDataFrame with elevation attributes
            patch_grid: Rasterized patch IDs
            nrows: Number of rows
            ncols: Number of columns

        Returns:
            Synthetic DEM array
        """
        # Find elevation column
        elev_col = None
        for col in ['elev_mean', 'elevation', 'dem_mean', 'z', 'ELEV']:
            if col in gdf.columns:
                elev_col = col
                break

        # Get ID column
        id_col = None
        for col in ['HRU_ID', 'hru_id', 'patch_id', 'PATCH_ID', 'ID', 'id']:
            if col in gdf.columns:
                id_col = col
                break

        if elev_col is None or id_col is None:
            self.logger.warning("No elevation data available, using constant DEM")
            return np.full((nrows, ncols), 1500.0, dtype='float32')

        # Create lookup from patch ID to elevation
        elev_lookup = dict(zip(gdf[id_col].astype(int), gdf[elev_col]))

        # Fill DEM from patch elevations
        dem_data = np.full((nrows, ncols), np.nan, dtype='float32')
        for pid, elev in elev_lookup.items():
            dem_data[patch_grid == pid] = elev

        # Fill nodata with mean
        mean_elev = np.nanmean(dem_data) if not np.all(np.isnan(dem_data)) else 1500.0
        dem_data[np.isnan(dem_data)] = mean_elev
        dem_data[patch_grid == 0] = mean_elev  # Background cells

        self.logger.info(f"Synthetic DEM created from HRU attributes: "
                        f"min={dem_data.min():.0f}m, max={dem_data.max():.0f}m")

        return dem_data

    def create_simple_grid(
        self,
        patch_info: list,
        arrange_by: str = 'elevation'
    ) -> Tuple[FireGrid, FireGrid]:
        """
        Create simple fire grid from patch information list.

        For domains without proper spatial data, creates a grid
        with patches arranged by elevation or ID.

        Args:
            patch_info: List of dicts with 'patch_id', 'elev', etc.
            arrange_by: How to arrange patches ('elevation' or 'id')

        Returns:
            Tuple of (patch_grid, dem_grid) as FireGrid objects
        """
        num_patches = len(patch_info)

        # Sort patches
        if arrange_by == 'elevation':
            patch_info = sorted(patch_info, key=lambda p: p.get('elev', 0))
        else:
            patch_info = sorted(patch_info, key=lambda p: p.get('patch_id', 0))

        # Create grid (3 columns, N rows)
        ncols = 3
        nrows = num_patches

        patch_data = np.zeros((nrows, ncols), dtype='int32')
        dem_data = np.zeros((nrows, ncols), dtype='float32')

        for i, pinfo in enumerate(patch_info):
            pid = pinfo.get('patch_id', i + 1)
            elev = pinfo.get('elev', 1500.0)
            patch_data[i, :] = pid
            dem_data[i, :] = elev

        # Create simple transform (arbitrary origin)
        transform = (float(self.resolution), 0.0, 0.0, 0.0, -float(self.resolution), float(nrows * self.resolution))

        patch_grid = FireGrid(
            data=patch_data,
            transform=transform,
            crs='EPSG:32610',  # Placeholder UTM
            resolution=self.resolution,
            nodata=-9999
        )

        dem_grid = FireGrid(
            data=dem_data,
            transform=transform,
            crs='EPSG:32610',
            resolution=self.resolution,
            nodata=-9999.0
        )

        return patch_grid, dem_grid
