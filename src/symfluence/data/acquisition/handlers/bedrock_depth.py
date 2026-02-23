"""
SoilGrids Depth to Bedrock (BDTICM) Acquisition Handler

Cloud-based acquisition of depth to bedrock from SoilGrids 2017.

Provides absolute depth to bedrock (cm) at 250m resolution globally.
Note: This layer is from SoilGrids 2017, not the current SoilGrids 2.0 release.

Primary method uses GDAL /vsicurl/ to remotely subset the global GeoTIFF
(8.5 GB) without downloading the entire file (supports HTTP range requests).
Falls back to full download + local subset if /vsicurl/ fails.

Data Source:
    https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif

References:
    Shangguan, W., Hengl, T., Mendes de Jesus, J., Yuan, H., & Dai, Y. (2017).
    Mapping the global depth to bedrock for land surface modeling.
    J. Adv. Model. Earth Syst., 9, 65-88.

Configuration:
    BDTICM_CONVERT_UNITS: Convert from cm to m (default: True)
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session

# Global GeoTIFF URL (supports HTTP range requests for remote subsetting)
_GLOBAL_URL = (
    "https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif"
)
_VSICURL_URL = f"/vsicurl/{_GLOBAL_URL}"


@AcquisitionRegistry.register('BEDROCK_DEPTH')
@AcquisitionRegistry.register('BDTICM')
@AcquisitionRegistry.register('DEPTH_TO_BEDROCK')
class BedrockDepthAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    SoilGrids 2017 depth to bedrock acquisition.

    Uses GDAL /vsicurl/ to remotely subset the 8.5 GB global GeoTIFF,
    downloading only the tiles needed for the domain bounding box.
    Falls back to full download if /vsicurl/ is not available.

    Output:
        {project_dir}/attributes/soilclass/bedrock/
            domain_{name}_bedrock_depth.tif  (units: meters or cm)
    """

    def download(self, output_dir: Path) -> Path:
        bedrock_dir = self._attribute_dir("soilclass") / "bedrock"
        bedrock_dir.mkdir(parents=True, exist_ok=True)

        convert_units = self._get_config_value(lambda: None, default=True, dict_key='BDTICM_CONVERT_UNITS')
        out_path = bedrock_dir / f"domain_{self.domain_name}_bedrock_depth.tif"

        if self._skip_if_exists(out_path):
            return bedrock_dir

        self.logger.info(
            f"Acquiring SoilGrids depth to bedrock (BDTICM) for bbox: {self.bbox}"
        )

        try:
            self._download_vsicurl(out_path, convert_units)
        except Exception as e:  # noqa: BLE001 — fallback to full download
            self.logger.warning(
                f"/vsicurl/ remote subset failed: {e}. "
                "Falling back to full download..."
            )
            self._download_full(out_path, convert_units)

        self._log_summary(out_path)
        self.logger.info(f"Bedrock depth acquisition complete: {out_path}")
        return bedrock_dir

    def _download_vsicurl(self, out_path: Path, convert_units: bool):
        """Remote subset via GDAL /vsicurl/ — downloads only needed tiles."""
        self.logger.info("Using /vsicurl/ for remote bbox subsetting...")

        with rasterio.open(_VSICURL_URL) as src:
            window = from_bounds(
                self.bbox['lon_min'], self.bbox['lat_min'],
                self.bbox['lon_max'], self.bbox['lat_max'],
                transform=src.transform
            )
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )

            data = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            nodata = src.nodata

            if convert_units:
                if nodata is not None:
                    mask = data == nodata
                    data = data / 100.0  # cm -> m
                    data[mask] = -9999.0
                    nodata = -9999.0
                else:
                    data = data / 100.0

            meta = src.meta.copy()
            meta.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform,
                'dtype': 'float32',
                'compress': 'lzw',
                'nodata': nodata,
            })

            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data, 1)

    def _download_full(self, out_path: Path, convert_units: bool):
        """Fallback: download full global GeoTIFF and subset locally."""
        cache_dir = out_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "BDTICM_M_250m_ll.tif"

        if not cache_path.exists():
            self.logger.info("Downloading global BDTICM GeoTIFF (~8.5 GB)...")
            session = create_robust_session(max_retries=3, backoff_factor=2.0)
            from ..utils import download_file_streaming
            download_file_streaming(
                _GLOBAL_URL, cache_path, session=session, timeout=3600
            )

        with rasterio.open(cache_path) as src:
            window = from_bounds(
                self.bbox['lon_min'], self.bbox['lat_min'],
                self.bbox['lon_max'], self.bbox['lat_max'],
                transform=src.transform
            )
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            data = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            nodata = src.nodata

            if convert_units:
                if nodata is not None:
                    mask = data == nodata
                    data = data / 100.0
                    data[mask] = -9999.0
                    nodata = -9999.0
                else:
                    data = data / 100.0

            meta = src.meta.copy()
            meta.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform,
                'dtype': 'float32',
                'compress': 'lzw',
                'nodata': nodata,
            })

            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data, 1)

    def _log_summary(self, path: Path):
        """Log summary statistics for the acquired data."""
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                nodata = src.nodata
                if nodata is not None:
                    valid = data[(data != nodata) & np.isfinite(data)]
                else:
                    valid = data[np.isfinite(data)]

                if len(valid) > 0:
                    unit = 'm' if self._get_config_value(lambda: None, default=True, dict_key='BDTICM_CONVERT_UNITS') else 'cm'
                    self.logger.info(
                        f"  Bedrock depth stats: "
                        f"min={valid.min():.1f} {unit}, "
                        f"mean={valid.mean():.1f} {unit}, "
                        f"max={valid.max():.1f} {unit}, "
                        f"pixels={len(valid)}"
                    )
        except Exception as e:  # noqa: BLE001 — summary logging is non-critical
            self.logger.debug(f"Could not log summary: {e}")
