"""MERIT-Hydro Acquisition Handler

Cloud-based acquisition of MERIT-Hydro hydrologically conditioned DEM and
river network data from the University of Tokyo.

MERIT-Hydro Overview:
    Data Type: Hydrologically conditioned DEM + river network attributes
    Resolution: 3 arc-seconds (~90m)
    Coverage: Global (90N-60S)
    Variables: elv (elevation), dir (flow direction), upa (upstream area),
               wth (river width), hand (height above nearest drainage)
    Source: Yamazaki et al. (2019), University of Tokyo

Tile Structure:
    Tar archives: 30x30 degree blocks containing 5x5 degree GeoTIFF tiles
    Tile naming: {lat_code}{lon_code}_{variable}.tif
    Tar naming: {variable}_{lat_code}{lon_code}.tar

Data Access:
    HTTP: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0.1/
    No authentication required

References:
    Yamazaki, D., et al. (2019). MERIT Hydro: A high-resolution global
    hydrography map based on latest topography datasets. Water Resources
    Research, 55, 5053-5073.
"""

import math
import tarfile
from pathlib import Path

import rasterio
from rasterio.merge import merge as rio_merge

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session, download_file_streaming

# Base URL for MERIT-Hydro v1.0.1
_BASE_URL = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0.1"

# Available variables
_ALL_VARIABLES = ['elv', 'dir', 'upa', 'wth', 'hand']

# Default variables to download
_DEFAULT_VARIABLES = ['elv', 'dir', 'upa']


def _lat_code(lat: int) -> str:
    """Convert latitude to MERIT-Hydro tile code (e.g., n30, s60)."""
    if lat >= 0:
        return f"n{lat:02d}"
    return f"s{abs(lat):02d}"


def _lon_code(lon: int) -> str:
    """Convert longitude to MERIT-Hydro tile code (e.g., e000, w120)."""
    if lon >= 0:
        return f"e{lon:03d}"
    return f"w{abs(lon):03d}"


def _get_tar_origin(coord: int, block_size: int = 30) -> int:
    """Get the origin coordinate of the containing tar block."""
    if coord >= 0:
        return (coord // block_size) * block_size
    return -((abs(coord) + block_size - 1) // block_size) * block_size


def _get_tile_origin(coord: int, tile_size: int = 5) -> int:
    """Get the origin coordinate of the containing 5x5 degree tile."""
    if coord >= 0:
        return (coord // tile_size) * tile_size
    return -((abs(coord) + tile_size - 1) // tile_size) * tile_size


@AcquisitionRegistry.register('MERIT_HYDRO')
class MERITHydroAcquirer(BaseAcquisitionHandler, RetryMixin):
    """MERIT-Hydro acquisition via HTTP tile download.

    Downloads hydrologically conditioned DEM and river network data from
    MERIT-Hydro. Data is organized in 30x30 degree tar archives containing
    5x5 degree GeoTIFF tiles.

    Acquisition Strategy:
        1. Calculate which 5x5 degree tiles cover the domain
        2. Determine which 30x30 degree tar archives contain those tiles
        3. Download tar archives with retry logic
        4. Extract relevant 5x5 degree tiles
        5. Mosaic tiles into domain-wide GeoTIFF per variable

    Configuration:
        MERIT_HYDRO_VARIABLES: List of variables to download
            (default: ['elv', 'dir', 'upa'])
            Options: elv, dir, upa, wth, hand

    Output:
        Per-variable GeoTIFF files in project_dir/attributes/elevation/merit_hydro/
        e.g., domain_{domain_name}_merit_elv.tif

    References:
        Yamazaki et al. (2019). MERIT Hydro. WRR, 55, 5053-5073.
    """

    def download(self, output_dir: Path) -> Path:
        merit_dir = self._attribute_dir("elevation") / "merit_hydro"
        merit_dir.mkdir(parents=True, exist_ok=True)

        variables = self._get_config_value(lambda: None, default=_DEFAULT_VARIABLES, dict_key='MERIT_HYDRO_VARIABLES')
        variables = [v for v in variables if v in _ALL_VARIABLES]
        if not variables:
            raise ValueError(
                f"No valid MERIT-Hydro variables configured. Choose from: {_ALL_VARIABLES}"
            )

        self.logger.info(
            f"Acquiring MERIT-Hydro data for bbox: {self.bbox}, variables: {variables}"
        )

        # Calculate 5x5 tile coverage
        lat_min_tile = _get_tile_origin(math.floor(self.bbox['lat_min']))
        lat_max_tile = _get_tile_origin(math.ceil(self.bbox['lat_max']))
        lon_min_tile = _get_tile_origin(math.floor(self.bbox['lon_min']))
        lon_max_tile = _get_tile_origin(math.ceil(self.bbox['lon_max']))

        tiles_needed = []
        for lat in range(lat_min_tile, lat_max_tile + 1, 5):
            for lon in range(lon_min_tile, lon_max_tile + 1, 5):
                tiles_needed.append((lat, lon))

        self.logger.info(f"Need {len(tiles_needed)} tiles for domain coverage")

        session = create_robust_session(max_retries=5, backoff_factor=2.0)
        output_paths = {}

        for variable in variables:
            out_path = merit_dir / f"domain_{self.domain_name}_merit_{variable}.tif"

            if self._skip_if_exists(out_path):
                output_paths[variable] = out_path
                continue

            tile_paths = []
            cache_dir = merit_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Group tiles by their 30x30 tar archive
            tar_groups: dict[tuple, list] = {}
            for lat, lon in tiles_needed:
                tar_lat = _get_tar_origin(lat)
                tar_lon = _get_tar_origin(lon)
                key = (tar_lat, tar_lon)
                if key not in tar_groups:
                    tar_groups[key] = []
                tar_groups[key].append((lat, lon))

            for (tar_lat, tar_lon), tile_list in tar_groups.items():
                lat_c = _lat_code(tar_lat)
                lon_c = _lon_code(tar_lon)
                tar_name = f"{variable}_{lat_c}{lon_c}.tar"
                tar_url = f"{_BASE_URL}/{variable}/{tar_name}"
                tar_path = cache_dir / tar_name

                # Download tar if not cached
                if not tar_path.exists() or tar_path.stat().st_size == 0:
                    self.logger.info(f"Downloading MERIT-Hydro tar: {tar_name}")
                    try:
                        self.execute_with_retry(
                            lambda u=tar_url, t=tar_path: download_file_streaming(
                                u, t, session=session, timeout=600
                            ),
                            max_retries=3,
                            base_delay=10.0,
                            backoff_factor=2.0,
                        )
                        # Validate: MERIT-Hydro is now password-protected;
                        # if we got an HTML login page, detect and report it
                        if tar_path.exists():
                            with open(tar_path, 'rb') as f:
                                header = f.read(16)
                            if b'<html' in header.lower() or b'<!doctype' in header.lower():
                                tar_path.unlink()
                                raise PermissionError(
                                    "MERIT-Hydro requires registration. "
                                    "Register at https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ "
                                    "and set MERIT_HYDRO_USERNAME / MERIT_HYDRO_PASSWORD in config, "
                                    "or use Google Earth Engine (MERIT/Hydro/v1_0_1) as alternative."
                                )
                    except Exception as e:
                        self.logger.warning(f"Failed to download {tar_name}: {e}")
                        continue

                # Validate cached file is actually a tar (not an HTML login page)
                if tar_path.exists():
                    with open(tar_path, 'rb') as f:
                        header = f.read(16)
                    if b'<html' in header.lower() or b'<!doctype' in header.lower():
                        tar_path.unlink()
                        self.logger.warning(
                            f"Removed stale HTML file masquerading as {tar_name}. "
                            "MERIT-Hydro requires registration at "
                            "https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/"
                        )
                        continue

                # Extract needed tiles from tar
                try:
                    with tarfile.open(tar_path, 'r') as tf:
                        for tile_lat, tile_lon in tile_list:
                            tile_lat_c = _lat_code(tile_lat)
                            tile_lon_c = _lon_code(tile_lon)
                            tile_filename = f"{tile_lat_c}{tile_lon_c}_{variable}.tif"

                            tile_out = cache_dir / tile_filename
                            if tile_out.exists():
                                tile_paths.append(tile_out)
                                continue

                            # Search tar for matching tile
                            members = tf.getnames()
                            match = None
                            for m in members:
                                if m.endswith(tile_filename) or m == tile_filename:
                                    match = m
                                    break

                            if match:
                                member = tf.getmember(match)
                                member.name = tile_filename
                                tf.extract(member, cache_dir)
                                tile_paths.append(tile_out)
                                self.logger.info(f"Extracted tile: {tile_filename}")
                            else:
                                self.logger.warning(
                                    f"Tile {tile_filename} not found in {tar_name}"
                                )
                except Exception as e:
                    self.logger.warning(f"Error processing tar {tar_name}: {e}")
                    continue

            if not tile_paths:
                raise FileNotFoundError(
                    f"No MERIT-Hydro tiles found for variable '{variable}' in bbox: {self.bbox}"
                )

            # Mosaic tiles
            if len(tile_paths) == 1:
                tile_paths[0].rename(out_path)
            else:
                self.logger.info(f"Merging {len(tile_paths)} MERIT-Hydro tiles for {variable}")
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                })
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for src in src_files:
                    src.close()

            output_paths[variable] = out_path
            self.logger.info(f"MERIT-Hydro {variable} saved: {out_path}")

        # Return the elevation path if available, otherwise first variable
        if 'elv' in output_paths:
            return output_paths['elv']
        return next(iter(output_paths.values()))
