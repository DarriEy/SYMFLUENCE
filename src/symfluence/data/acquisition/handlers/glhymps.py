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
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import box

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..utils import create_robust_session, download_file_streaming

# GLHYMPS download sources — tried in order until one succeeds
_GLHYMPS_SOURCES = [
    {
        'name': 'Borealis Data (primary)',
        'url': 'https://borealisdata.ca/api/access/datafile/71909',
        'size_hint': '~2.4 GB',
        'format': 'shapefile',
    },
]

# CONUS-only fallback (GeoPackage from Hugging Face / pygeoglim)
_GLHYMPS_CONUS_URL = (
    "https://huggingface.co/datasets/mgalib/GLIM_GLHYMPS/"
    "resolve/main/GLHYMP_CONUS.gpkg"
)

# Key attribute columns
_KEEP_COLUMNS = [
    'Porosity',        # Porosity (fraction)
    'logK_Ice',        # Log permeability (m^2) considering permafrost
    'logK_Ferr',       # Log permeability (m^2) no permafrost adjustment
    'Porosity_x',      # Unconsolidated porosity (v2.0)
    'logK_Ice_x',      # Unconsolidated log permeability (v2.0)
    'geometry',
]


@R.acquisition_handlers.add('GLHYMPS')
@R.acquisition_handlers.add('GLHYMPS_V2')
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

        self.logger.debug("Reading and clipping GLHYMPS to domain bbox")

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

        # Log summary (one INFO line; per-column stats at DEBUG)
        n_polys = len(gdf)
        for col in ['Porosity', 'logK_Ice']:
            if col in gdf.columns:
                self.logger.debug(
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
                self.logger.debug(f"Using cached GLHYMPS: {shp}")
                return shp

        for shp in glhymps_dir.rglob("*.shp"):
            if 'glhymps' in shp.name.lower() or 'GLHYMPS' in shp.name:
                self.logger.debug(f"Found local GLHYMPS: {shp}")
                return shp

        # Try each source until one works
        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        zip_path = cache_dir / "GLHYMPS.zip"

        for source in _GLHYMPS_SOURCES:
            self.logger.info(
                f"Downloading GLHYMPS from {source['name']} "
                f"({source['size_hint']})"
            )

            try:
                download_file_streaming(
                    source['url'], zip_path, session=session, timeout=1800
                )

                # Detect placeholder responses (Borealis maintenance mode)
                if zip_path.stat().st_size < 10000:
                    content = zip_path.read_bytes()
                    if b'"status"' in content or b'<html' in content.lower():
                        zip_path.unlink(missing_ok=True)
                        self.logger.warning(
                            f"{source['name']} returned a placeholder — "
                            f"service may be in maintenance"
                        )
                        continue

                self.logger.debug("Download complete, extracting")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    from symfluence.core.archive_extraction import safe_zip_extract
                    safe_zip_extract(zf, cache_dir)

                zip_path.unlink(missing_ok=True)

                for shp in cache_dir.rglob("*.shp"):
                    if 'glhymps' in shp.name.lower() or 'GLHYMPS' in shp.name:
                        self.logger.debug(f"Extracted GLHYMPS shapefile: {shp}")
                        return shp

                # Bundle may contain GLHYMPS in a subdirectory
                for shp in cache_dir.rglob("*.shp"):
                    self.logger.debug(f"Extracted shapefile: {shp}")
                    return shp

                self.logger.warning("No shapefile found in archive")
                continue

            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Failed from {source['name']}: {e}"
                )
                zip_path.unlink(missing_ok=True)
                continue

        # Fallback: CONUS-only GeoPackage from Hugging Face
        if (self.bbox['lat_min'] >= 24 and self.bbox['lat_max'] <= 50
                and self.bbox['lon_min'] >= -125 and self.bbox['lon_max'] <= -66):
            self.logger.info(
                "Trying CONUS-only GLHYMPS from Hugging Face (~546 MB)"
            )
            gpkg_path = cache_dir / "GLHYMP_CONUS.gpkg"
            try:
                download_file_streaming(
                    _GLHYMPS_CONUS_URL, gpkg_path,
                    session=session, timeout=600,
                )
                if gpkg_path.exists() and gpkg_path.stat().st_size > 10000:
                    self.logger.debug(f"Downloaded CONUS GLHYMPS: {gpkg_path}")
                    return gpkg_path
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"CONUS fallback failed: {e}")

        self.logger.error(
            "All GLHYMPS download sources failed. Borealis Data "
            "(the only global source) may be in maintenance. "
            "Download manually from "
            "https://borealisdata.ca/dataset.xhtml"
            "?persistentId=doi:10.5683/SP2/TTJNIU "
            f"and place the shapefile in {cache_dir}"
        )
        return None
