"""
DEM stream burning processor.

Rasterizes a stream vector network onto a DEM grid and lowers elevation
along stream pixels by a configurable burn depth. This forces TauDEM
flow direction to follow mapped streams, improving delineation on flat terrain.

Used as step 0 in the TauDEM pipeline when DEM_CONDITIONING_METHOD = 'burn_streams'.
"""

from pathlib import Path
from typing import Any

import numpy as np


class StreamBurner:
    """
    Burns stream vectors into a DEM by lowering elevation at stream pixels.

    Reads a DEM raster and a stream vector file (shapefile, geopackage, or
    parquet), rasterizes the streams onto the DEM grid with ``all_touched=True``,
    and subtracts a burn depth from every stream pixel. The result is written
    as a new GeoTIFF; the original DEM is never modified.

    Args:
        logger: Logger instance for status messages.
    """

    def __init__(self, logger: Any):
        self.logger = logger

    def burn_streams(
        self,
        dem_path: Path,
        stream_path: Path,
        output_path: Path,
        burn_depth: float = 5.0,
    ) -> Path:
        """
        Burn stream network into a DEM.

        Args:
            dem_path: Path to source DEM raster (GeoTIFF).
            stream_path: Path to stream vector file (.shp, .gpkg, .parquet).
            output_path: Path for the burned DEM output.
            burn_depth: Metres to subtract from DEM at stream pixels.

        Returns:
            Path to the burned DEM file.

        Raises:
            FileNotFoundError: If dem_path or stream_path do not exist.
            RuntimeError: If rasterio or geopandas are unavailable.
        """
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize

        dem_path = Path(dem_path)
        stream_path = Path(stream_path)
        output_path = Path(output_path)

        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        if not stream_path.exists():
            raise FileNotFoundError(f"Stream file not found: {stream_path}")

        # 1. Read DEM
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            dem_crs = src.crs
            nodata = src.nodata

        self.logger.info(
            f"Read DEM: {dem_array.shape}, CRS={dem_crs}, nodata={nodata}"
        )

        # 2. Load stream vectors
        streams = gpd.read_file(stream_path)
        self.logger.info(
            f"Loaded {len(streams)} stream features from {stream_path.name}"
        )

        if streams.empty:
            self.logger.warning("Stream file is empty — returning unmodified DEM")
            self._write_dem(dem_array, profile, output_path)
            return output_path

        # 3. Reproject streams to DEM CRS if needed
        if streams.crs is not None and dem_crs is not None and streams.crs != dem_crs:
            self.logger.info(
                f"Reprojecting streams from {streams.crs} to {dem_crs}"
            )
            streams = streams.to_crs(dem_crs)

        # 4. Filter to line geometries only
        line_mask = streams.geometry.geom_type.isin(
            ['LineString', 'MultiLineString']
        )
        if not line_mask.all():
            n_dropped = (~line_mask).sum()
            self.logger.warning(
                f"Dropping {n_dropped} non-line features from stream file"
            )
            streams = streams[line_mask]

        if streams.empty:
            self.logger.warning(
                "No line geometries in stream file — returning unmodified DEM"
            )
            self._write_dem(dem_array, profile, output_path)
            return output_path

        # 5. Rasterize streams onto DEM grid (all_touched=True for thin lines)
        geometries = list(streams.geometry)
        stream_mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=dem_array.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        n_burned = int(stream_mask.sum())
        self.logger.info(
            f"Stream mask: {n_burned} pixels to burn "
            f"({100 * n_burned / dem_array.size:.2f}% of DEM)"
        )

        # 6. Subtract burn depth where mask=1 and value != nodata
        burned = dem_array.copy()
        burn_where = stream_mask == 1
        if nodata is not None:
            burn_where = burn_where & (dem_array != nodata)
        burned[burn_where] -= burn_depth

        # 7. Write burned DEM with LZW compression
        profile.update(compress='lzw')
        self._write_dem(burned, profile, output_path)

        self.logger.info(f"Burned DEM written to {output_path}")
        return output_path

    @staticmethod
    def _write_dem(array: np.ndarray, profile: dict, output_path: Path) -> None:
        """Write a DEM array to GeoTIFF."""
        import rasterio

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(array, 1)
