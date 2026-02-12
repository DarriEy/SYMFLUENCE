"""DEM Acquisition Handlers

Cloud-based acquisition of global elevation data from multiple sources:
- Copernicus GLO-30 (30m): AWS S3, cloud-optimized GeoTIFF
- FABDEM (30m): Forest/building-removed variant from Source Cooperative
- NASADEM Local (30m): Local tile discovery and merging

Key Features:
    Tile Management:
    - 1x1 degree tile scheme (standard for global DEMs)
    - Automatic tile merging for domains spanning multiple tiles
    - Local caching to avoid re-downloads

    Retry Logic:
    - Exponential backoff for transient failures
    - Configurable max retries and backoff factors
    - Robust session creation with connection pooling

Data Sources:
    Copernicus DEM:
    - URL: AWS S3 (copernicus-dem-30m bucket)
    - Format: COG (Cloud-Optimized GeoTIFF)
    - Advantages: Fast S3 access, consistent quality

    FABDEM:
    - URL: Source Cooperative (AWS S3)
    - Processing: Copernicus DEM + GEDI + landcover masking
    - Advantages: Forest/building removal for hydrology

    NASADEM Local:
    - Source: Pre-downloaded local tiles
    - Format: Flexible (HGT or GeoTIFF)
    - Advantages: Offline operation, custom DEMs

References:
    - Copernicus DEM: https://registry.opendata.aws/copernicus-dem/
    - Hawker et al. (2022). FABDEM: Global forest and building height maps
      Scientific Data, 9, 488
    - USGS NASADEM: https://lpdaac.usgs.gov/products/nasadem_hgt/
"""

import math
from pathlib import Path
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds, Window
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ..mixins import RetryMixin
from ..utils import create_robust_session


@AcquisitionRegistry.register('COPDEM30')
class CopDEM30Acquirer(BaseAcquisitionHandler, RetryMixin):
    """Copernicus DEM GLO-30 acquisition via AWS S3 with tile management.

    Downloads and merges global 30m resolution Digital Elevation Model (DEM)
    from the Copernicus DEM collection hosted on AWS S3. Uses cloud-optimized
    GeoTIFF (COG) format for efficient cloud access with per-tile retry logic.

    Copernicus DEM GLO-30:
        Data Type: Global Digital Elevation Model (raster)
        Resolution: 30m (1 arc-second)
        Coverage: Global (90°S - 90°N, 180°W - 180°E)
        Source: Copernicus Programme / DLR & Airbus
        Format: Cloud-Optimized GeoTIFF (COG)
        Datum: WGS84 (EPSG:4326)
        Units: Meters above sea level

    Tile Scheme:
        Organization: 1°×1° degree tiles
        Naming: Copernicus_DSM_COG_10_{LAT}_{LON}_00_DEM
        Example: Copernicus_DSM_COG_10_N40_00_E105_00_DEM
        Coordinates: N/S for latitude (00-89), E/W for longitude (000-179)

    Acquisition Workflow:
        1. Calculate tile indices from domain bounding box (floor/ceil)
        2. For each required tile:
           a. Generate S3 URL with proper tile naming convention
           b. Download with retry logic (max 5 retries, exponential backoff)
           c. Cache locally to avoid re-downloads
        3. Merge tiles to single output GeoTIFF
        4. Clip to exact domain bounding box
        5. Apply LZW compression

    AWS S3 Configuration:
        Bucket: copernicus-dem-30m
        Region: eu-central-1
        Base URL: https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com
        Access: Public dataset (no credentials required)
        Performance: Fast S3 access, regional caching

    Error Handling:
        - Per-tile retry: 5 attempts with exponential backoff (2x factor)
        - Partial downloads: Detected via HTTP Content-Length
        - Missing tiles: Logged as warning, attempt continues with remaining
        - Complete failure: Raises FileNotFoundError if no tiles found

    Output:
        GeoTIFF file: domain_{domain_name}_elv.tif
        - Format: Cloud-Optimized GeoTIFF (COG)
        - Compression: LZW (lossless)
        - Data Type: Typically 16-bit integers or 32-bit floats
        - NoData Value: -32768 (void areas, if present)

    Advantages:
        - Fast cloud-native access (COG format)
        - Consistent global coverage
        - Well-documented data source
        - Reliable AWS S3 infrastructure

    References:
        - Copernicus DEM: https://www.copernicus.eu/en/access-data/copernicus-data
        - AWS Public Dataset: https://registry.opendata.aws/copernicus-dem/
        - Product Specification: https://www.dlr.de/eoc/en/desktopdefault.aspx/
    """

    def download(self, output_dir: Path) -> Path:
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading Copernicus DEM GLO-30 for bbox: {self.bbox}")

        # AWS S3 Public Dataset: eu-central-1
        base_url = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"

        # Tiles are 1x1 degree
        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            # Create session with retry logic
            session = create_robust_session(max_retries=5, backoff_factor=2.0)

            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                    url = f"{base_url}/{tile_name}/{tile_name}.tif"

                    local_tile = dem_dir / f"temp_{tile_name}.tif"
                    if local_tile.exists():
                        # Validate cached tile is readable
                        if not self._validate_tile(local_tile, tile_name):
                            local_tile.unlink()
                        else:
                            self.logger.info(f"Using cached tile: {tile_name}")
                            tile_paths.append(local_tile)
                            continue

                    self.logger.info(f"Fetching tile: {tile_name}")
                    tile_result = self._download_tile_with_retry(
                        session, url, local_tile, tile_name
                    )
                    if tile_result:
                        tile_paths.append(tile_result)

            if not tile_paths:
                raise FileNotFoundError(f"No Copernicus DEM tiles found for bbox: {self.bbox}")

            if len(tile_paths) == 1:
                if out_path.exists(): out_path.unlink()
                tile_paths[0].rename(out_path)
            else:
                self.logger.info(f"Merging {len(tile_paths)} tiles into {out_path}")
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw"
                })
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for src in src_files: src.close()
                for p in tile_paths: p.unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Error downloading/processing Copernicus DEM: {e}")
            for p in tile_paths:
                if p.exists() and p != out_path: p.unlink(missing_ok=True)
            raise

        return out_path

    def _validate_tile(self, tile_path: Path, tile_name: str) -> bool:
        """Validate a cached tile is readable by rasterio."""
        try:
            with rasterio.open(tile_path) as src:
                src.read(1, window=Window(0, 0, 1, 1))
            return True
        except Exception:
            self.logger.warning(f"Cached tile {tile_name} is corrupted, will re-download")
            return False

    def _download_tile_with_retry(
        self, session, url: str, local_tile: Path, tile_name: str
    ) -> Path | None:
        """Download a single tile with retry logic using RetryMixin."""

        def do_download():
            try:
                with session.get(url, stream=True, timeout=300) as r:
                    if r.status_code == 200:
                        expected_size = int(r.headers.get('Content-Length', 0))
                        bytes_written = 0
                        with open(local_tile, "wb") as f:
                            for chunk in r.iter_content(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    bytes_written += len(chunk)
                        if expected_size and bytes_written < expected_size:
                            raise IOError(
                                f"Incomplete download for {tile_name}: "
                                f"got {bytes_written}, expected {expected_size}"
                            )
                        self.logger.info(f"✓ Downloaded {tile_name}")
                        return local_tile
                    elif r.status_code == 404:
                        self.logger.warning(f"Tile {tile_name} not found (404)")
                        return None
                    else:
                        raise requests.exceptions.HTTPError(
                            f"HTTP {r.status_code} for {tile_name}"
                        )
            except (requests.RequestException, OSError, IOError) as download_err:
                # Clean up partial download before retry
                if local_tile.exists():
                    local_tile.unlink()
                raise download_err

        # For 404s, don't retry - just return None
        try:
            return self.execute_with_retry(
                do_download,
                max_retries=3,
                base_delay=2,
                backoff_factor=2.0,
                retryable_exceptions=(
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    BrokenPipeError,
                    IOError,
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to download {tile_name}: {e}")
            raise


@AcquisitionRegistry.register('FABDEM')
class FABDEMAcquirer(BaseAcquisitionHandler):
    """FABDEM acquisition handler for forest/building-removed elevation data.

    Downloads and processes FABDEM (Forest And Buildings removed DEM) v1-2,
    a global 30m elevation model with vegetation and anthropogenic structures
    removed. Useful for hydrological modeling where bare-earth DEM is required.

    FABDEM v1-2 Overview:
        Data Type: Digital Elevation Model with forest/building removal
        Resolution: 30m (1 arc-second)
        Coverage: Global (90°S - 90°N)
        Source: Hawker et al. (2022), Source Cooperative
        Processing: Copernicus DEM + GEDI + landcover masking
        Format: Cloud-Optimized GeoTIFF (COG)
        Datum: WGS84 (EPSG:4326)
        Units: Meters above sea level

    Forest and Building Removal:
        Preprocessing steps in FABDEM:
        1. Start with Copernicus DEM 30m baseline
        2. Apply GEDI space-based lidar canopy height measurements
        3. Use forest masks (ESA World Cover, GEDI) for vegetation identification
        4. Remove building pixels using OpenStreetMap and OSM data
        5. Interpolate removed pixels for hydrologically valid surface

        Benefits for hydrological modeling:
        - More accurate streamflow routing (no artificial dams from buildings)
        - Better representation of flood pathways (forest gaps opened)
        - Reduced artifacts from tall vegetation over actual terrain
        - Useful for flood risk and rainfall-runoff modeling

    Tile Scheme:
        Organization: 1°×1° degree tiles
        Naming: {LAT}{LON}_FABDEM_V1-2.tif
        Example: N46W122_FABDEM_V1-2.tif
        Coordinates: N/S for latitude (00-89), E/W for longitude (000-179)

    Acquisition Workflow:
        1. Calculate tile indices from domain bounding box
        2. For each required tile:
           a. Generate Source Cooperative URL
           b. Download GeoTIFF tile
           c. Cache locally to avoid re-downloads
        3. Merge multiple tiles (if needed) via rasterio.merge()
        4. Clip to exact domain bounding box
        5. Output single GeoTIFF

    Source Cooperative Configuration:
        Provider: Source Cooperative (AWS S3)
        Base URL: https://data.source.coop/c_6_6/fabdem/tiles/
        Format: COG (Cloud-Optimized GeoTIFF)
        Access: Public dataset (no credentials)
        Performance: AWS S3 access with global caching

    Error Handling:
        - Missing tiles: HTTP 404, logged as warning, continue with available
        - Network failures: Immediate exception (no retry)
        - Single tile: Direct copy/crop (no mosaic needed)
        - Multiple tiles: Automated rasterio merge

    Output:
        GeoTIFF file: domain_{domain_name}_elv.tif
        - Format: GeoTIFF
        - Compression: Inherited from source tiles
        - Data Type: Typically 16-bit or 32-bit elevation
        - Spatial extent: Exact domain bounding box

    Use Cases:
        - Hydrological modeling (flood routing, streamflow)
        - Urban hydrology (better representation of flood pathways)
        - Wildlife habitat modeling (accurate terrain without tall vegetation)
        - Landslide/avalanche risk assessment
        - Visibility/line-of-sight analysis

    References:
        - Hawker et al. (2022). A 30m global map of elevation corrected for
          vegetation bias and national boundaries. Scientific Data, 9, 488
        - Source Cooperative: https://source.coop/
        - GEDI Data: https://daac.ornl.gov/GEDI/
    """

    def download(self, output_dir: Path) -> Path:
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if self._skip_if_exists(out_path):
            return out_path

        self.logger.info(f"Downloading FABDEM for bbox: {self.bbox}")
        # Source Cooperative (AWS)
        base_url = "https://data.source.coop/c_6_6/fabdem/tiles"

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    # FABDEM format: N46W122_FABDEM_V1-2.tif
                    tile_name = f"{lat_str}{lon_str}_FABDEM_V1-2"
                    url = f"{base_url}/{tile_name}.tif"

                    local_tile = dem_dir / f"temp_fab_{tile_name}.tif"
                    if not local_tile.exists():
                        self.logger.info(f"Fetching FABDEM tile: {tile_name}")
                        with requests.get(url, stream=True, timeout=60) as r:
                            if r.status_code == 200:
                                with open(local_tile, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                                tile_paths.append(local_tile)
                    else:
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No FABDEM tiles found for bbox: {self.bbox}")

            if len(tile_paths) == 1:
                if out_path.exists(): out_path.unlink()
                tile_paths[0].rename(out_path)
            else:
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
                with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
                for src in src_files: src.close()
                for p in tile_paths: p.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Error with FABDEM: {e}")
            raise
        return out_path


@AcquisitionRegistry.register('NASADEM_LOCAL')
class NASADEMLocalAcquirer(BaseAcquisitionHandler):
    """NASADEM local tile acquisition for pre-downloaded elevation data.

    Discovers and merges NASADEM or compatible local elevation tiles to create
    a domain-specific DEM. Enables offline operation and use of pre-downloaded
    tiles or alternative bare-earth elevation products (e.g., commercial DEMs).

    NASADEM Overview:
        Data Type: Merged SRTM v3 + ASTER DEM elevation
        Resolution: 30m (1 arc-second)
        Coverage: Global (±60° latitude)
        Source: USGS EROS Data Center
        Vertical Accuracy: ±20m (SRTM regions), ±30m (ASTER)
        Format: Flexible (HGT or GeoTIFF)
        Datum: WGS84 (EPSG:4326)
        Units: Meters above sea level

    Local Tile Organization:
        Directory Structure:
        nasadem_tiles_dir/
        ├── n46w122.tif  (or .hgt)
        ├── n46w121.tif
        ├── n47w122.tif
        └── ...

        Naming Convention:
        - {LAT}{LON}.tif or {LAT}{LON}.hgt
        - Latitude: n00-n60 or s00-s60 (North/South of equator)
        - Longitude: e000-e179 or w000-w179 (East/West of Greenwich)
        - Example: n46w122.tif (46°N, 122°W)

    Acquisition Workflow:
        1. Validate NASADEM_LOCAL_DIR configuration
        2. Calculate required tile indices from bounding box
        3. For each tile:
           a. Glob for matching tiles (.tif or .hgt format)
           b. Collect found tiles into list
        4. Single tile: Direct crop to domain bbox
        5. Multiple tiles: Automated merge via rasterio.merge()
        6. Output to domain-specific GeoTIFF

    Configuration:
        Required setting:
        - config.data.geospatial.nasadem.local_dir: Path to tile directory
          Example: /data/elevation_data/nasadem_tiles/
          Can be local path or network-mounted directory

        Directory must exist and contain ≥1 tile covering domain

    Flexible Tile Format:
        Supports both formats:
        - GeoTIFF (.tif): Preferred, includes georeferencing
        - HGT (.hgt): Legacy SRTM raw format, auto-georeferenced
        - Pattern matching: {lat_str}{lon_str}*.tif or .hgt
        - First match used if multiple versions exist

    Error Handling:
        - Directory not configured: ValueError
        - Directory not found: FileNotFoundError
        - No tiles covering bbox: FileNotFoundError with bbox info
        - Single tile: Crop and copy (preserves originals)
        - Multiple tiles: Merge with window clipping

    Output:
        GeoTIFF file: domain_{domain_name}_elv.tif
        - Format: GeoTIFF
        - Compression: Inherited from source tiles
        - Data Type: Typically 16-bit elevation
        - Spatial extent: Exact domain bounding box

    Use Cases:
        - Air-gapped systems (no internet access)
        - Pre-downloaded tile archives
        - Custom commercial DEMs (when converted to .tif)
        - Verified/validated elevation datasets
        - Local high-resolution DEMs (LiDAR, InSAR)

    Tile Source Options:
        Official USADEM:
        - USGS EROS Data Center: earthexplorer.usgs.gov
        - Download NASADEM or SRTM tiles
        - HGT or GeoTIFF formats supported

        Commercial Alternatives:
        - COPDEM (Copernicus processed)
        - SRTM+ processed variants
        - Proprietary LiDAR-based DEMs
        - InSAR-derived elevation models

    References:
        - USGS NASADEM: https://lpdaac.usgs.gov/products/nasadem_hgt/
        - SRTM v3: https://lpdaac.usgs.gov/products/srtmgl1elev/
        - Earth Explorer: https://earthexplorer.usgs.gov/
    """
    def download(self, output_dir: Path) -> Path:
        local_src_dir_cfg = self._get_config_value(
            lambda: self.config.data.geospatial.nasadem.local_dir
        )
        if not local_src_dir_cfg:
            raise ValueError("NASADEM_LOCAL_DIR must be configured for NASADEM_LOCAL acquirer")
        local_src_dir = Path(local_src_dir_cfg)
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if not local_src_dir.exists():
            raise FileNotFoundError(f"NASADEM_LOCAL_DIR not found: {local_src_dir}")

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        for lat in range(lat_min, lat_max):
            for lon in range(lon_min, lon_max):
                lat_str = f"n{lat:02d}" if lat >= 0 else f"s{-lat:02d}"
                lon_str = f"e{lon:03d}" if lon >= 0 else f"w{-lon:03d}"
                # Common NASADEM format: n46w122.hgt or .tif
                pattern = f"{lat_str}{lon_str}*.tif"
                matches = list(local_src_dir.glob(pattern))
                if not matches:
                    pattern = f"{lat_str}{lon_str}*.hgt"
                    matches = list(local_src_dir.glob(pattern))

                if matches:
                    tile_paths.append(matches[0])

        if not tile_paths:
            raise FileNotFoundError(f"No NASADEM tiles found in {local_src_dir} for bbox {self.bbox}")

        if len(tile_paths) == 1:
            # We don't want to move original files, so we crop/copy
            with rasterio.open(tile_paths[0]) as src:
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                data = src.read(1, window=win)
                meta = src.meta.copy()
                meta.update({"height": data.shape[0], "width": data.shape[1], "transform": src.window_transform(win)})
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(data, 1)
        else:
            src_files = [rasterio.open(p) for p in tile_paths]
            mosaic, out_trans = rio_merge(src_files)
            out_meta = src_files[0].meta.copy()
            out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
            with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
            for src in src_files: src.close()

        return out_path
