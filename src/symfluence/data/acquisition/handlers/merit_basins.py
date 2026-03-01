# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MERIT-Basins Vector Data Acquisition Handler

Downloads catchment and river network vector data from the MERIT-Basins
dataset (Lin et al., 2019). Data is organized by Pfafstetter Level-1
hydrological regions.

Workflow:
    1. Map pour point to Pfafstetter Level-1 code(s) using hardcoded bboxes
    2. Download catchment + river shapefiles for matching region(s)
    3. The subsetter's spatial join handles precise selection after download

Source:
    reachhydro.org — Pfafstetter-organized shapefiles
    http://hydrology.princeton.edu/data/mpan/MERIT_Basins/

Column Convention (matches subsetter MERIT type):
    COMID, NextDownID, up1, up2, up3

References:
    - Lin et al. (2019). Global reconstruction of naturalized river flows at
      2.94 million reaches. Water Resources Research, 55, 6499-6516
    - Yamazaki et al. (2019). MERIT Hydro. Water Resources Research, 55, 5053-5073
"""

import zipfile
from pathlib import Path
from typing import List

import requests

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session

# MERIT-Basins data URLs
_MERIT_BASE_URL = "http://hydrology.princeton.edu/data/mpan/MERIT_Basins"
_CATCHMENTS_URL = f"{_MERIT_BASE_URL}/MERIT_Catchments/pfaf_{{code}}_MERIT_Hydro_v07_Basins_v01.zip"
_RIVERS_URL = f"{_MERIT_BASE_URL}/MERIT_Rivernet/pfaf_{{code}}_MERIT_Hydro_v07_Basins_v01_rivernet.zip"

# Pfafstetter Level-1 continental regions with approximate bounding boxes
# Format: {code: (lat_min, lon_min, lat_max, lon_max)}
_PFAFSTETTER_L1_BBOXES = {
    1: (-35.0, -82.0, 15.0, -34.0),   # South America — Amazon, eastern
    2: (-56.0, -82.0, 15.0, -34.0),   # South America — Parana, southern
    3: (5.0, -130.0, 62.0, -52.0),    # North America — Atlantic/Gulf
    4: (5.0, -170.0, 72.0, -100.0),   # North America — Pacific/Arctic
    5: (35.0, -15.0, 72.0, 60.0),     # Europe
    6: (-36.0, -20.0, 38.0, 55.0),    # Africa
    7: (0.0, 55.0, 78.0, 180.0),      # Asia — North/Central
    8: (-12.0, 90.0, 55.0, 155.0),    # Asia — Southeast / Oceania
    9: (-50.0, 110.0, -10.0, 180.0),  # Australia / Oceania
}


@AcquisitionRegistry.register('MERIT_BASINS')
class MERITBasinsAcquirer(BaseAcquisitionHandler, RetryMixin):
    """MERIT-Basins vector data acquisition from reachhydro.org.

    Downloads Pfafstetter-organized catchment and river network shapefiles.
    Uses coarse continental bounding boxes for region selection — the
    subsetter handles precise spatial selection after download.

    Output Files:
        merit_cat_pfaf_{code}.shp — catchment polygons
        merit_riv_pfaf_{code}.shp — river network lines
    """

    def download(self, output_dir: Path) -> Path:
        """Download MERIT-Basins data for the domain.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing downloaded geofabric files
        """
        geofabric_dir = self._attribute_dir("geofabric") / "merit_basins"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Downloading MERIT-Basins data for pour point ({lat:.4f}, {lon:.4f})"
        )

        # Find matching Pfafstetter region(s)
        pfaf_codes = self._find_pfafstetter_regions(lat, lon)
        if not pfaf_codes:
            raise ValueError(
                f"Pour point ({lat}, {lon}) does not fall within any "
                "Pfafstetter Level-1 region"
            )
        self.logger.info(f"Matched Pfafstetter L1 region(s): {pfaf_codes}")

        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        for code in pfaf_codes:
            # Download catchments
            cat_marker = geofabric_dir / f"merit_cat_pfaf_{code}.shp"
            if not self._skip_if_exists(cat_marker):
                cat_url = _CATCHMENTS_URL.format(code=code)
                self._download_and_extract(
                    session, cat_url, geofabric_dir,
                    f"MERIT catchments pfaf_{code}"
                )

            # Download rivers
            riv_marker = geofabric_dir / f"merit_riv_pfaf_{code}.shp"
            if not self._skip_if_exists(riv_marker):
                riv_url = _RIVERS_URL.format(code=code)
                self._download_and_extract(
                    session, riv_url, geofabric_dir,
                    f"MERIT rivers pfaf_{code}"
                )

        self.logger.info(f"MERIT-Basins data downloaded to: {geofabric_dir}")
        return geofabric_dir

    def _get_pour_point_coords(self) -> tuple:
        """Extract pour point lat/lon from config.

        Returns:
            Tuple of (lat, lon)
        """
        pour_point_str = self._get_config_value(lambda: self.config.domain.pour_point_coords, default=None)
        if pour_point_str:
            parts = str(pour_point_str).replace('/', ',').split(',')
            return float(parts[0].strip()), float(parts[1].strip())

        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
        return lat, lon

    def _find_pfafstetter_regions(self, lat: float, lon: float) -> List[int]:
        """Find Pfafstetter L1 region(s) containing the pour point.

        Uses coarse bounding boxes — may return multiple regions for
        points near boundaries. Also checks the full bounding box if
        available.

        Args:
            lat: Latitude of pour point
            lon: Longitude of pour point

        Returns:
            List of matching Pfafstetter L1 codes
        """
        matching = []
        for code, (lat_min, lon_min, lat_max, lon_max) in _PFAFSTETTER_L1_BBOXES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                matching.append(code)

        # Also check full domain bbox for large domains spanning regions
        if self.bbox:
            for code, (lat_min, lon_min, lat_max, lon_max) in _PFAFSTETTER_L1_BBOXES.items():
                if code not in matching:
                    # Check if any corner of domain bbox falls in this region
                    if (self.bbox['lat_min'] <= lat_max and
                            self.bbox['lat_max'] >= lat_min and
                            self.bbox['lon_min'] <= lon_max and
                            self.bbox['lon_max'] >= lon_min):
                        matching.append(code)

        return sorted(set(matching))

    def _download_and_extract(
        self, session, url: str, output_dir: Path, description: str
    ):
        """Download a zip file and extract its contents.

        Args:
            session: requests.Session
            url: URL to download
            output_dir: Directory to extract files into
            description: Human-readable description for logging
        """
        def do_download():
            self.logger.info(f"Downloading {description}")
            zip_path = output_dir / "temp_download.zip"
            try:
                with session.get(url, stream=True, timeout=600) as resp:
                    resp.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)

                self.logger.info(f"Extracting {description}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(output_dir)

                self.logger.info(f"Downloaded and extracted {description}")
            finally:
                if zip_path.exists():
                    zip_path.unlink()

        self.execute_with_retry(
            do_download,
            max_retries=3,
            base_delay=5,
            backoff_factor=2.0,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                IOError,
            ),
        )
