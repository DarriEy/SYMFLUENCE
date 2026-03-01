# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Global Aridity Index and PET Acquisition Handler

Cloud-based acquisition of the Global Aridity Index and Potential
Evapotranspiration (ET0) Climate Database v3 from CGIAR-CSI.

Provides climatological (1970-2000) gridded data at ~1km resolution:
- Aridity Index (AI = P/ET0, dimensionless)
- Reference Evapotranspiration (ET0, mm/year)

Data Source:
    Figshare: https://doi.org/10.6084/m9.figshare.7504448
    No authentication required, CC-BY 4.0 license

References:
    Zomer, R.J., Xu, J., & Trabucco, A. (2022). Version 3 of the Global
    Aridity Index and Potential Evapotranspiration Database. Scientific
    Data, 9, 409.

Configuration:
    ARIDITY_INDEX_VARIABLES: ['ai', 'et0'] (default: both)
"""

import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session, download_file_streaming

# Figshare direct file download URLs (v3.1)
# Use the Figshare API ndownloader endpoint with file IDs for reliable downloads
# (article-level URLs are blocked by WAF bot protection)
_ARCHIVE_URL = "https://ndownloader.figshare.com/files/56300327"  # Global-AI_ET0__annual_v3_1.zip (646 MB)

_VARIABLES = {
    'ai': {
        'description': 'Global Aridity Index (P/ET0)',
        'units': 'dimensionless',
        'scale_factor': 0.0001,  # Raw values are scaled by 10000
        'search_patterns': ['*ai*yr.tif', '*ai*annual*.tif', '*AI*annual*.tif', '*aridity*.tif'],
    },
    'et0': {
        'description': 'Reference Evapotranspiration (Penman-Monteith)',
        'units': 'mm/year',
        'scale_factor': 0.1,  # Raw values are scaled by 10
        'search_patterns': ['*et0*yr.tif', '*et0*annual*.tif', '*ET0*annual*.tif', '*evapotranspiration*.tif'],
    },
}

_DEFAULT_VARIABLES = ['ai', 'et0']


@AcquisitionRegistry.register('ARIDITY_INDEX')
@AcquisitionRegistry.register('GLOBAL_ARIDITY')
@AcquisitionRegistry.register('CGIAR_ARIDITY')
class AridityIndexAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    CGIAR-CSI Global Aridity Index and PET acquisition.

    Downloads aridity index and/or reference ET from Figshare,
    subsets to domain bounding box, and applies scale factors.

    Output:
        {project_dir}/attributes/climate/aridity/
            domain_{name}_aridity_index.tif
            domain_{name}_reference_et0.tif
    """

    def download(self, output_dir: Path) -> Path:
        aridity_dir = self._attribute_dir("climate") / "aridity"
        aridity_dir.mkdir(parents=True, exist_ok=True)

        variables = self._get_config_value(lambda: None, default=_DEFAULT_VARIABLES, dict_key='ARIDITY_INDEX_VARIABLES')
        variables = [v.lower() for v in variables if v.lower() in _VARIABLES]

        if not variables:
            raise ValueError(f"No valid aridity variables. Choose from: {list(_VARIABLES.keys())}")

        self.logger.info(
            f"Acquiring Global Aridity Index/PET for bbox: {self.bbox}, "
            f"variables: {variables}"
        )

        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        cache_dir = aridity_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download archive
        self._ensure_archive_downloaded(session, cache_dir)

        output_paths = {}
        for var in variables:
            info = _VARIABLES[var]
            suffix = 'aridity_index' if var == 'ai' else 'reference_et0'
            out_path = aridity_dir / f"domain_{self.domain_name}_{suffix}.tif"

            if self._skip_if_exists(out_path):
                output_paths[var] = out_path
                continue

            # Find the source GeoTIFF in cache
            src_tif = self._find_source_tif(cache_dir, info['search_patterns'])
            if src_tif is None:
                self.logger.warning(f"Could not find {var} GeoTIFF in cache")
                continue

            self.logger.info(f"Subsetting {var} from {src_tif.name}")
            self._subset_and_scale(src_tif, out_path, info['scale_factor'])
            output_paths[var] = out_path

            # Log summary
            self._log_summary(out_path, var, info['units'])

        if not output_paths:
            raise RuntimeError("No aridity data could be processed")

        self.logger.info(f"Aridity index acquisition complete: {len(output_paths)} files")
        return aridity_dir

    def _ensure_archive_downloaded(self, session, cache_dir: Path):
        """Download and extract the Figshare archive if not cached."""
        # Check if TIFs already extracted
        existing_tifs = list(cache_dir.rglob("*.tif"))
        if existing_tifs:
            return

        zip_path = cache_dir / "Global_AI_ET0_v3.zip"

        if not zip_path.exists():
            self.logger.info("Downloading Global Aridity Index v3 from Figshare...")
            download_file_streaming(
                _ARCHIVE_URL, zip_path, session=session, timeout=600
            )

        self.logger.info("Extracting archive...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)
            # Also extract any nested zips
            for nested_zip in cache_dir.rglob("*.zip"):
                if nested_zip != zip_path:
                    try:
                        with zipfile.ZipFile(nested_zip, 'r') as zf:
                            zf.extractall(nested_zip.parent)
                    except zipfile.BadZipFile:
                        pass
        except zipfile.BadZipFile:
            self.logger.error("Downloaded file is not a valid zip archive")
            zip_path.unlink(missing_ok=True)
            raise

    def _find_source_tif(self, cache_dir: Path, patterns: List[str]) -> Optional[Path]:
        """Find a GeoTIFF matching one of the search patterns."""
        for pattern in patterns:
            matches = list(cache_dir.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def _subset_and_scale(self, src_path: Path, dst_path: Path, scale_factor: float):
        """Subset GeoTIFF to bbox and apply scale factor."""
        with rasterio.open(src_path) as src:
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

            # Apply scale factor
            if nodata is not None:
                mask = data == nodata
                data = data * scale_factor
                data[mask] = -9999.0
                nodata_out = -9999.0
            else:
                data = data * scale_factor
                nodata_out = None

            meta = src.meta.copy()
            meta.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform,
                'dtype': 'float32',
                'compress': 'lzw',
                'nodata': nodata_out,
            })

            with rasterio.open(dst_path, 'w', **meta) as dst:
                dst.write(data, 1)

    def _log_summary(self, path: Path, var: str, units: str):
        """Log summary statistics."""
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                nodata = src.nodata
                if nodata is not None:
                    valid = data[data != nodata]
                else:
                    valid = data[np.isfinite(data)]

                if len(valid) > 0:
                    self.logger.info(
                        f"  {var}: min={valid.min():.3f}, mean={valid.mean():.3f}, "
                        f"max={valid.max():.3f} {units}, pixels={len(valid)}"
                    )
        except Exception as e:  # noqa: BLE001 — summary logging is non-critical
            self.logger.debug(f"Could not log summary for {var}: {e}")
