# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GLHYMPS 2.0 Data Acquisition Handler

Cloud-based acquisition of the Global Hydrogeology Maps of Permeability
and Porosity (GLHYMPS) v2.0 from Borealis Data (University of Victoria).

GLHYMPS provides polygon-based hydrogeological properties globally:
- Log permeability (m^2) for consolidated and unconsolidated layers
- Porosity (fraction) for consolidated and unconsolidated layers

Data Source:
    Borealis Data: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/TTJNIU
    No authentication required, CC-BY 4.0 license

References:
    Huscroft, J., Gleeson, T., Hartmann, J., & Borgers, J. (2018).
    Compiling and mapping global permeability of the unconsolidated
    and consolidated Earth. Geophys. Res. Lett., 45, 1897-1904.

Configuration:
    GLHYMPS_VERSION: '2.0' (default) or '1.0'
"""

import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import box

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session, download_file_streaming

# GLHYMPS 2.0 download URLs (Borealis Data)
_GLHYMPS_URLS = {
    '2.0': 'https://borealisdata.ca/api/access/datafile/71909',  # GLHYMPS.zip (2.4 GB)
    '1.0': 'https://borealisdata.ca/api/access/datafile/72026',  # GLHYMPS.zip (1.1 GB)
}

# Key attribute columns
_KEEP_COLUMNS = [
    'Porosity',        # Porosity (fraction)
    'logK_Ice',        # Log permeability (m^2) considering permafrost
    'logK_Ferr',       # Log permeability (m^2) no permafrost adjustment
    'Porosity_x',      # Unconsolidated porosity (v2.0)
    'logK_Ice_x',      # Unconsolidated log permeability (v2.0)
    'geometry',
]


@AcquisitionRegistry.register('GLHYMPS')
@AcquisitionRegistry.register('GLHYMPS_V2')
class GLHYMPSAcquirer(BaseAcquisitionHandler):
    """
    GLHYMPS v2.0 global hydrogeology acquisition.

    Downloads the global GLHYMPS shapefile from Borealis Data, clips
    to the domain bounding box, and saves as a GeoPackage.

    Output:
        {project_dir}/attributes/geology/glhymps/
            domain_{name}_glhymps.gpkg
    """

    def download(self, output_dir: Path) -> Path:
        glhymps_dir = self._attribute_dir("geology") / "glhymps"
        glhymps_dir.mkdir(parents=True, exist_ok=True)

        out_gpkg = glhymps_dir / f"domain_{self.domain_name}_glhymps.gpkg"

        if self._skip_if_exists(out_gpkg):
            return glhymps_dir

        self.logger.info("Starting GLHYMPS v2.0 acquisition")

        # Download or locate cached global shapefile
        global_shp = self._get_global_shapefile(glhymps_dir)
        if global_shp is None:
            self.logger.error(
                "Could not obtain GLHYMPS data. "
                "Download manually from https://borealisdata.ca/dataset.xhtml"
                "?persistentId=doi:10.5683/SP2/TTJNIU "
                f"and place the shapefile in {glhymps_dir / 'cache'}"
            )
            return glhymps_dir

        # Clip to domain bbox with buffer
        buf_deg = 0.1
        domain_box = box(
            self.bbox["lon_min"] - buf_deg,
            self.bbox["lat_min"] - buf_deg,
            self.bbox["lon_max"] + buf_deg,
            self.bbox["lat_max"] + buf_deg,
        )

        self.logger.info("Reading and clipping GLHYMPS to domain bbox...")

        # GLHYMPS uses Cylindrical Equal Area projection — transform bbox to source CRS
        src_crs = gpd.read_file(global_shp, rows=0).crs
        if src_crs and not src_crs.is_geographic:
            bbox_gdf = gpd.GeoDataFrame(geometry=[domain_box], crs="EPSG:4326")
            bbox_gdf = bbox_gdf.to_crs(src_crs)
            query_bbox = bbox_gdf.geometry.iloc[0]
        else:
            query_bbox = domain_box

        gdf = gpd.read_file(global_shp, bbox=query_bbox)

        if gdf is None or len(gdf) == 0:
            self.logger.warning("No GLHYMPS data found in domain bounding box")
            return glhymps_dir

        # Keep available columns
        available = [c for c in _KEEP_COLUMNS if c in gdf.columns]
        if not available:
            available = list(gdf.columns)
        gdf = gdf[available].copy()

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        gdf.to_file(out_gpkg, driver="GPKG")

        # Log summary
        n_polys = len(gdf)
        for col in ['Porosity', 'logK_Ice']:
            if col in gdf.columns:
                self.logger.info(
                    f"  {col}: mean={gdf[col].mean():.4f}, "
                    f"range=[{gdf[col].min():.4f}, {gdf[col].max():.4f}]"
                )

        self.logger.info(f"GLHYMPS clipped: {n_polys} polygons -> {out_gpkg}")
        return glhymps_dir

    def _get_global_shapefile(self, glhymps_dir: Path) -> Optional[Path]:
        """Download or locate cached global GLHYMPS shapefile."""
        cache_dir = glhymps_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing shapefile
        for shp in cache_dir.rglob("*.shp"):
            if 'glhymps' in shp.name.lower() or 'GLHYMPS' in shp.name:
                self.logger.info(f"Using cached GLHYMPS: {shp}")
                return shp

        for shp in glhymps_dir.rglob("*.shp"):
            if 'glhymps' in shp.name.lower() or 'GLHYMPS' in shp.name:
                self.logger.info(f"Found local GLHYMPS: {shp}")
                return shp

        # Download from Borealis Data
        version = self._get_config_value(lambda: None, default='2.0', dict_key='GLHYMPS_VERSION')
        url = _GLHYMPS_URLS.get(version, _GLHYMPS_URLS['2.0'])

        zip_path = cache_dir / "GLHYMPS.zip"
        self.logger.info(f"Downloading GLHYMPS v{version} from Borealis Data")
        self.logger.info("This is a ~2.4 GB download and may take several minutes...")

        session = create_robust_session(max_retries=3, backoff_factor=2.0)

        try:
            download_file_streaming(url, zip_path, session=session, timeout=1800)

            self.logger.info("Download complete, extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)

            zip_path.unlink(missing_ok=True)

            for shp in cache_dir.rglob("*.shp"):
                self.logger.info(f"Extracted GLHYMPS shapefile: {shp}")
                return shp

            self.logger.error("No shapefile found in extracted archive")
            return None

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Failed to download GLHYMPS: {e}")
            self.logger.info(
                "Manual download: visit https://borealisdata.ca/dataset.xhtml"
                "?persistentId=doi:10.5683/SP2/TTJNIU\n"
                f"  Place the extracted shapefile in: {cache_dir}"
            )
            return None
