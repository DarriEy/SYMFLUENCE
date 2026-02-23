"""
World Karst Aquifer Map (WOKAM) Acquisition Handler

Cloud-based acquisition of the WOKAM global karst aquifer dataset
from BGR (German Federal Institute for Geosciences and Natural Resources).

Provides polygon-based classification of karstifiable rock outcrops globally:
- Carbonate rock polygons (limestone, dolomite, marble, etc.)
- Evaporite rock polygons (gypsum, anhydrite, halite)
- Aquifer classification (exposed vs. covered karst)

Data Source:
    BGR WHYMAP: https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_WOKAM_v1.zip
    No authentication required

References:
    Chen, Z., Auler, A.S., Bakalowicz, M., et al. (2017). The World
    Karst Aquifer Mapping project: concept, mapping procedure and map
    of Europe. Hydrogeology Journal, 25, 771-785.

    Goldscheider, N., Chen, Z., Auler, A.S., et al. (2020). Global
    distribution of carbonate rocks and karst water resources.
    Hydrogeology Journal, 28, 1661-1677.
"""

import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import box

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session, download_file_streaming

# WOKAM download URL
_WOKAM_URL = "https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_WOKAM_v1.zip"


@AcquisitionRegistry.register('WOKAM')
@AcquisitionRegistry.register('KARST')
@AcquisitionRegistry.register('KARST_AQUIFER')
class WOKAMAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    WOKAM (World Karst Aquifer Map) global karst acquisition.

    Downloads the WOKAM shapefile from BGR, clips to the domain
    bounding box, and saves as a GeoPackage.

    Output:
        {project_dir}/attributes/geology/karst/
            domain_{name}_wokam_karst.gpkg
    """

    def download(self, output_dir: Path) -> Path:
        karst_dir = self._attribute_dir("geology") / "karst"
        karst_dir.mkdir(parents=True, exist_ok=True)

        out_gpkg = karst_dir / f"domain_{self.domain_name}_wokam_karst.gpkg"

        if self._skip_if_exists(out_gpkg):
            return karst_dir

        self.logger.info("Starting WOKAM karst aquifer acquisition")

        # Download or locate cached shapefile
        global_shp = self._get_global_shapefile(karst_dir)
        if global_shp is None:
            self.logger.error(
                "Could not obtain WOKAM data. "
                "Download manually from "
                "https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_WOKAM_v1.zip "
                f"and place the shapefile in {karst_dir / 'cache'}"
            )
            return karst_dir

        # Clip to domain bbox with buffer
        buf_deg = 0.5  # Larger buffer for coarse karst polygons
        domain_box = box(
            self.bbox["lon_min"] - buf_deg,
            self.bbox["lat_min"] - buf_deg,
            self.bbox["lon_max"] + buf_deg,
            self.bbox["lat_max"] + buf_deg,
        )

        self.logger.info("Reading and clipping WOKAM to domain bbox...")
        try:
            gdf = gpd.read_file(global_shp, bbox=domain_box)
        except Exception as e:
            self.logger.error(f"Failed to read WOKAM shapefile: {e}")
            return karst_dir

        if gdf is None or len(gdf) == 0:
            self.logger.info("No karst formations found in domain bounding box")
            # Write empty gpkg to avoid re-downloading
            empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            empty.to_file(out_gpkg, driver="GPKG")
            return karst_dir

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        gdf.to_file(out_gpkg, driver="GPKG")

        # Log summary
        n_polys = len(gdf)
        if 'ROCK_TYPE' in gdf.columns:
            rock_counts = gdf['ROCK_TYPE'].value_counts()
            self.logger.info(f"WOKAM clipped: {n_polys} polygons")
            for rock_type, count in rock_counts.items():
                self.logger.info(f"  {rock_type}: {count} polygons")
        elif 'Type' in gdf.columns:
            type_counts = gdf['Type'].value_counts()
            self.logger.info(f"WOKAM clipped: {n_polys} polygons")
            for t, count in type_counts.items():
                self.logger.info(f"  {t}: {count} polygons")
        else:
            self.logger.info(
                f"WOKAM clipped: {n_polys} polygons -> {out_gpkg}"
            )

        return karst_dir

    def _get_global_shapefile(self, karst_dir: Path) -> Optional[Path]:
        """Download or locate cached global WOKAM shapefile."""
        cache_dir = karst_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing karst polygon shapefile (prefer *karst*poly* over cave/spring point files)
        for search_dir in [cache_dir, karst_dir]:
            # First pass: look specifically for the karst polygon shapefile
            for shp in search_dir.rglob("*karst*poly*.shp"):
                self.logger.info(f"Using cached WOKAM karst polygons: {shp}")
                return shp
            # Second pass: any WOKAM/WHYMAP shapefile
            for shp in search_dir.rglob("*.shp"):
                if 'wokam' in shp.name.lower() or 'karst' in shp.name.lower():
                    self.logger.info(f"Using cached WOKAM: {shp}")
                    return shp

        # Download from BGR (retry up to 3 times — BGR often drops connections)
        zip_path = cache_dir / "WHYMAP_WOKAM_v1.zip"
        self.logger.info(f"Downloading WOKAM from BGR: {_WOKAM_URL}")

        last_err = None
        for attempt in range(1, 4):
            try:
                session = create_robust_session(max_retries=3, backoff_factor=2.0)
                download_file_streaming(
                    _WOKAM_URL, zip_path, session=session, timeout=600
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                self.logger.warning(f"Download attempt {attempt}/3 failed: {e}")

        if last_err is not None:
            self.logger.error(f"Failed to download WOKAM after 3 attempts: {last_err}")
            return None

        self.logger.info("Download complete, extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)
        except zipfile.BadZipFile:
            self.logger.error("Downloaded file is not a valid zip archive")
            zip_path.unlink(missing_ok=True)
            return None

        zip_path.unlink(missing_ok=True)

        # Find extracted karst polygon shapefile
        for shp in cache_dir.rglob("*karst*poly*.shp"):
            self.logger.info(f"Extracted WOKAM karst polygons: {shp}")
            return shp
        # Fallback to any shapefile
        for shp in cache_dir.rglob("*.shp"):
            self.logger.info(f"Extracted WOKAM shapefile: {shp}")
            return shp

        self.logger.error("No shapefile found in extracted archive")
        return None
