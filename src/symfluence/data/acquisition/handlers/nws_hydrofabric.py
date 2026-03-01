# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NWS NextGen Hydrofabric Acquisition Handler

Downloads CONUS river network and catchment data from the Lynker-Spatial
NextGen Hydrofabric on S3. Data is organized by HUC-2 based VPUs in
GeoPackage format.

Workflow:
    1. Map pour point to VPU code(s) using hardcoded HUC-2 bounding boxes
    2. Download GeoPackage for matching VPU(s)
    3. Extract catchments and flowpaths layers

Source:
    Lynker-Spatial S3:
    https://nextgen-hydrofabric.s3.amazonaws.com/{version}/nextgen_{VPU}.gpkg

Column Convention (matches subsetter NWS type):
    COMID, toCOMID

References:
    - Johnson et al. (2023). National Hydrologic Geospatial Fabric (hydrofabric)
      for the Next Generation (NextGen) Hydrologic Modeling Framework
    - https://www.lynker-spatial.com/
"""

from pathlib import Path
from typing import List

import requests

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session

# Lynker-Spatial S3 URL template
_NWS_S3_TEMPLATE = (
    "https://nextgen-hydrofabric.s3.amazonaws.com/{version}/nextgen_{vpu}.gpkg"
)

# CONUS VPU (HUC-2) approximate bounding boxes
# Format: {vpu_code: (lat_min, lon_min, lat_max, lon_max)}
_NWS_VPU_BBOXES = {
    "01": (40.0, -80.5, 48.0, -66.5),     # New England
    "02": (36.0, -81.0, 45.5, -72.0),     # Mid-Atlantic
    "03N": (29.0, -91.0, 37.0, -75.5),    # South Atlantic-Gulf (North)
    "03S": (24.0, -90.0, 31.0, -79.5),    # South Atlantic-Gulf (South)
    "03W": (28.5, -91.5, 35.0, -84.0),    # South Atlantic-Gulf (West)
    "04": (40.5, -93.0, 49.5, -74.0),     # Great Lakes
    "05": (34.5, -90.5, 43.5, -77.5),     # Ohio
    "06": (35.0, -91.5, 43.5, -83.0),     # Tennessee
    "07": (37.0, -98.0, 49.5, -84.0),     # Upper Mississippi
    "08": (28.0, -98.0, 37.0, -88.0),     # Lower Mississippi
    "09": (43.5, -99.0, 49.5, -88.0),     # Souris-Red-Rainy
    "10L": (36.0, -105.0, 49.5, -95.5),   # Missouri (Lower)
    "10U": (37.0, -114.0, 49.5, -96.0),   # Missouri (Upper)
    "11": (27.5, -106.5, 40.0, -93.5),    # Arkansas-White-Red
    "12": (25.5, -107.0, 37.0, -93.0),    # Texas-Gulf
    "13": (31.0, -109.5, 38.0, -103.0),   # Rio Grande
    "14": (35.5, -113.0, 43.5, -105.5),   # Upper Colorado
    "15": (31.0, -115.0, 38.0, -106.5),   # Lower Colorado
    "16": (34.0, -120.5, 44.0, -109.0),   # Great Basin
    "17": (42.0, -125.0, 49.5, -110.5),   # Pacific Northwest
    "18": (32.0, -125.0, 43.0, -114.0),   # California
}


@AcquisitionRegistry.register('NWS_HYDROFABRIC')
@AcquisitionRegistry.register('NWS')
class NWSHydrofabricAcquirer(BaseAcquisitionHandler, RetryMixin):
    """NWS NextGen Hydrofabric acquisition from Lynker-Spatial S3.

    Downloads CONUS hydrofabric GeoPackages organized by HUC-2 VPUs.
    Extracts catchments and flowpaths layers for use by the subsetter.

    Config Keys:
        NWS_HYDROFABRIC_VERSION: Version string (default: 'v2.2')

    Output Files:
        nws_catchments_{vpu}.gpkg — catchment polygons (from GeoPackage layer)
        nws_flowpaths_{vpu}.gpkg — river network lines (from GeoPackage layer)
    """

    def download(self, output_dir: Path) -> Path:
        """Download NWS Hydrofabric data for the domain.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing downloaded geofabric files
        """

        version = self._get_config_value(lambda: None, default='v2.2', dict_key='NWS_HYDROFABRIC_VERSION')

        geofabric_dir = self._attribute_dir("geofabric") / "nws_hydrofabric"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Downloading NWS Hydrofabric ({version}) for "
            f"pour point ({lat:.4f}, {lon:.4f})"
        )

        # Find matching VPU(s)
        vpu_codes = self._find_vpu_regions(lat, lon)
        if not vpu_codes:
            raise ValueError(
                f"Pour point ({lat}, {lon}) does not fall within any CONUS VPU. "
                "NWS Hydrofabric is CONUS-only."
            )
        self.logger.info(f"Matched VPU(s): {vpu_codes}")

        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        for vpu in vpu_codes:
            cat_path = geofabric_dir / f"nws_catchments_{vpu}.gpkg"
            riv_path = geofabric_dir / f"nws_flowpaths_{vpu}.gpkg"

            if self._skip_if_exists(cat_path) and self._skip_if_exists(riv_path):
                continue

            # Download full GeoPackage
            gpkg_path = geofabric_dir / f"nextgen_{vpu}.gpkg"
            if not gpkg_path.exists():
                url = _NWS_S3_TEMPLATE.format(version=version, vpu=vpu)
                self._download_gpkg(session, url, gpkg_path, f"NWS VPU {vpu}")

            # Extract layers
            self._extract_layers(gpkg_path, cat_path, riv_path, vpu)

        self.logger.info(f"NWS Hydrofabric data downloaded to: {geofabric_dir}")
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

    def _find_vpu_regions(self, lat: float, lon: float) -> List[str]:
        """Find NWS VPU(s) containing the pour point.

        Args:
            lat: Latitude of pour point
            lon: Longitude of pour point

        Returns:
            List of matching VPU codes
        """
        matching = []
        for vpu, (lat_min, lon_min, lat_max, lon_max) in _NWS_VPU_BBOXES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                matching.append(vpu)
        return matching

    def _download_gpkg(
        self, session, url: str, output_path: Path, description: str
    ):
        """Download a GeoPackage file with retry logic.

        Args:
            session: requests.Session
            url: URL to download from
            output_path: Local path to save to
            description: Human-readable description for logging
        """
        def do_download():
            self.logger.info(f"Downloading {description} from {url}")
            part_path = output_path.with_suffix('.part')
            with session.get(url, stream=True, timeout=900) as resp:
                resp.raise_for_status()
                with open(part_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            part_path.rename(output_path)
            self.logger.info(f"Downloaded {description}: {output_path}")

        self.execute_with_retry(
            do_download,
            max_retries=3,
            base_delay=10,
            backoff_factor=2.0,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                IOError,
            ),
        )

    def _extract_layers(
        self, gpkg_path: Path, cat_path: Path, riv_path: Path, vpu: str
    ):
        """Extract catchments and flowpaths layers from GeoPackage.

        Args:
            gpkg_path: Path to source GeoPackage
            cat_path: Output path for catchments
            riv_path: Output path for flowpaths
            vpu: VPU code for logging
        """
        import fiona
        import geopandas as gpd

        available_layers = fiona.listlayers(gpkg_path)
        self.logger.info(f"VPU {vpu} GeoPackage layers: {available_layers}")

        # Extract catchments layer
        cat_layer = None
        for name in ['divides', 'catchments', 'cats']:
            if name in available_layers:
                cat_layer = name
                break
        if cat_layer is None:
            # Use first polygon layer as fallback
            for name in available_layers:
                with fiona.open(gpkg_path, layer=name) as src:
                    if src.schema['geometry'] in ('Polygon', 'MultiPolygon'):
                        cat_layer = name
                        break

        if cat_layer:
            if not cat_path.exists():
                cats = gpd.read_file(gpkg_path, layer=cat_layer)
                cats.to_file(cat_path, driver='GPKG')
                self.logger.info(
                    f"Extracted {len(cats)} catchments from layer '{cat_layer}'"
                )
        else:
            self.logger.warning(f"No catchment layer found in VPU {vpu} GeoPackage")

        # Extract flowpaths/rivers layer
        riv_layer = None
        for name in ['flowpaths', 'flowlines', 'rivers', 'streams']:
            if name in available_layers:
                riv_layer = name
                break
        if riv_layer is None:
            for name in available_layers:
                with fiona.open(gpkg_path, layer=name) as src:
                    if src.schema['geometry'] in ('LineString', 'MultiLineString'):
                        riv_layer = name
                        break

        if riv_layer:
            if not riv_path.exists():
                rivs = gpd.read_file(gpkg_path, layer=riv_layer)
                rivs.to_file(riv_path, driver='GPKG')
                self.logger.info(
                    f"Extracted {len(rivs)} flowpaths from layer '{riv_layer}'"
                )
        else:
            self.logger.warning(f"No flowpath layer found in VPU {vpu} GeoPackage")

        # Clean up full GeoPackage after extraction
        if cat_path.exists() and riv_path.exists() and gpkg_path.exists():
            gpkg_path.unlink()
            self.logger.info(f"Removed intermediate GeoPackage for VPU {vpu}")
