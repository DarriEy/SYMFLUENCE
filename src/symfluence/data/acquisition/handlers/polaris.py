"""POLARIS Soil Properties Acquisition Handler

Cloud-based acquisition of POLARIS (Probabilistic Remapping of SSURGO)
continuous soil properties at 30m resolution for CONUS.

POLARIS Overview:
    Data Type: Continuous soil properties (probabilistic ensemble)
    Resolution: 1 arc-second (~30m)
    Coverage: CONUS only
    Variables: clay, sand, silt, bd, ksat, theta_s, theta_r, om, ph, alpha, n, lambda, hb
    Depths: 0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm
    Statistics: mean, mode, p5, p50, p95
    Source: Chaney et al. (2019), Duke University

Tile Structure:
    1x1 degree GeoTIFF tiles
    URL pattern: {base}/{variable}/{statistic}/{depth}/lat{lat}{lat+1}_lon{lon}{lon+1}.tif

Data Access:
    HTTP: http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/
    No authentication required

References:
    Chaney, N. W., et al. (2019). POLARIS soil properties: 30-m probabilistic
    maps of soil properties over the contiguous United States. Water Resources
    Research, 55, 2916-2938.
"""

import math
from pathlib import Path

import rasterio
from rasterio.merge import merge as rio_merge

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ..mixins import RetryMixin
from ..utils import create_robust_session, download_file_streaming


# Base URL for POLARIS v1.0
_BASE_URL = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"

# Available variables
_ALL_VARIABLES = [
    'clay', 'sand', 'silt', 'bd', 'ksat', 'theta_s', 'theta_r',
    'om', 'ph', 'alpha', 'n', 'lambda', 'hb',
]

# Default subset
_DEFAULT_VARIABLES = ['sand', 'clay', 'silt', 'ksat', 'bd', 'theta_s']

# Available depth layers
_ALL_DEPTHS = ['0_5', '5_15', '15_30', '30_60', '60_100', '100_200']
_DEFAULT_DEPTHS = ['0_5', '5_15', '15_30', '30_60', '60_100']

# Available statistics
_ALL_STATISTICS = ['mean', 'mode', 'p5', 'p50', 'p95']


@AcquisitionRegistry.register('POLARIS')
class POLARISAcquirer(BaseAcquisitionHandler, RetryMixin):
    """POLARIS 30m soil property acquisition via HTTP tile download.

    Downloads continuous soil property data from the POLARIS dataset.
    Data is organized as 1x1 degree GeoTIFF tiles, one per variable+depth+statistic
    combination.

    Acquisition Strategy:
        1. Calculate which 1x1 degree tiles cover the domain
        2. For each variable+depth combo, download all tiles
        3. Mosaic tiles into domain-wide GeoTIFF
        4. One output file per variable+depth combination

    Configuration:
        POLARIS_VARIABLES: List of soil properties to download
            (default: ['sand', 'clay', 'silt', 'ksat', 'bd', 'theta_s'])
        POLARIS_DEPTHS: List of depth layers
            (default: ['0_5', '5_15', '15_30', '30_60', '60_100'])
        POLARIS_STATISTIC: Statistical measure
            (default: 'mean', options: mean, mode, p5, p50, p95)

    Output:
        Per-variable+depth GeoTIFF files in project_dir/attributes/soilclass/polaris/
        e.g., domain_{domain_name}_polaris_sand_0_5_mean.tif

    References:
        Chaney et al. (2019). POLARIS soil properties. WRR, 55, 2916-2938.
    """

    def download(self, output_dir: Path) -> Path:
        polaris_dir = self._attribute_dir("soilclass") / "polaris"
        polaris_dir.mkdir(parents=True, exist_ok=True)

        variables = self.config_dict.get('POLARIS_VARIABLES', _DEFAULT_VARIABLES)
        variables = [v for v in variables if v in _ALL_VARIABLES]
        depths = self.config_dict.get('POLARIS_DEPTHS', _DEFAULT_DEPTHS)
        depths = [d for d in depths if d in _ALL_DEPTHS]
        statistic = self.config_dict.get('POLARIS_STATISTIC', 'mean')

        if statistic not in _ALL_STATISTICS:
            self.logger.warning(
                f"Unknown POLARIS statistic '{statistic}', using 'mean'"
            )
            statistic = 'mean'

        if not variables:
            raise ValueError(f"No valid POLARIS variables. Choose from: {_ALL_VARIABLES}")
        if not depths:
            raise ValueError(f"No valid POLARIS depths. Choose from: {_ALL_DEPTHS}")

        # CONUS coverage check
        if (self.bbox['lat_min'] < 24 or self.bbox['lat_max'] > 50 or
                self.bbox['lon_min'] < -125 or self.bbox['lon_max'] > -66):
            self.logger.warning(
                "POLARIS covers CONUS only (24-50N, 125-66W). "
                "Tiles outside this range may not be available."
            )

        self.logger.info(
            f"Acquiring POLARIS soil data for bbox: {self.bbox}, "
            f"variables: {variables}, depths: {depths}, statistic: {statistic}"
        )

        # Calculate 1x1 tile coverage
        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        session = create_robust_session(max_retries=5, backoff_factor=2.0)
        cache_dir = polaris_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        total_combos = len(variables) * len(depths)
        completed = 0

        for variable in variables:
            for depth in depths:
                completed += 1
                combo_key = f"{variable}_{depth}"
                out_path = polaris_dir / (
                    f"domain_{self.domain_name}_polaris_{variable}_{depth}_{statistic}.tif"
                )

                if self._skip_if_exists(out_path):
                    output_paths[combo_key] = out_path
                    continue

                self.logger.info(
                    f"[{completed}/{total_combos}] Downloading POLARIS {variable} {depth}cm {statistic}"
                )

                tile_paths = []
                for lat in range(lat_min, lat_max):
                    for lon in range(lon_min, lon_max):
                        tile_name = f"lat{lat}{lat + 1}_lon{lon}{lon + 1}.tif"
                        url = f"{_BASE_URL}/{variable}/{statistic}/{depth}/{tile_name}"
                        local_tile = cache_dir / f"polaris_{variable}_{depth}_{statistic}_{tile_name}"

                        if local_tile.exists() and local_tile.stat().st_size > 0:
                            tile_paths.append(local_tile)
                            continue

                        try:
                            self.execute_with_retry(
                                lambda u=url, t=local_tile: download_file_streaming(
                                    u, t, session=session, timeout=300
                                ),
                                max_retries=3,
                                base_delay=5.0,
                                backoff_factor=2.0,
                            )
                            tile_paths.append(local_tile)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to download POLARIS tile {tile_name} "
                                f"for {variable}/{depth}: {e}"
                            )
                            continue

                if not tile_paths:
                    self.logger.warning(
                        f"No tiles found for POLARIS {variable} {depth}cm"
                    )
                    continue

                # Mosaic tiles
                if len(tile_paths) == 1:
                    import shutil
                    shutil.copy2(tile_paths[0], out_path)
                else:
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

                output_paths[combo_key] = out_path
                self.logger.info(f"Saved: {out_path}")

        if not output_paths:
            raise RuntimeError("No POLARIS data could be downloaded")

        self.logger.info(
            f"POLARIS acquisition complete: {len(output_paths)} variable+depth files"
        )

        # Return the directory containing all outputs
        return polaris_dir
