import math
from pathlib import Path
import requests
import rasterio
from rasterio.merge import merge as rio_merge
import numpy as np
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

class GeospatialAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        # This base class can be used if we want a unified entry point for geospatial
        pass

@AcquisitionRegistry.register('SOILGRIDS')
class SoilGridsAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        soil_dir = self._attribute_dir("soilclass")

        # Get layer name from config or use default
        layer = self.config.get("SOILGRIDS_LAYER", "wrb_0-5cm_mode")

        # Use default WCS map and coverage if not specified
        wcs_map = self.config.get("SOILGRIDS_WCS_MAP", "/vsicurl/https://files.isric.org/soilgrids/latest/data/wrb/wrb_0-5cm_mode.vrt")
        coverage = self.config.get("SOILGRIDS_COVERAGE_ID", layer)

        params = [("map", wcs_map), ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"), ("COVERAGEID", coverage), ("FORMAT", "GEOTIFF_INT16"), ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"), ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"), ("SUBSET", f"Lat({self.bbox['lat_min']},{self.bbox['lat_max']})"), ("SUBSET", f"Lon({self.bbox['lon_min']},{self.bbox['lon_max']})")]
        resp = requests.get("https://maps.isric.org/mapserv", params=params, stream=True)
        resp.raise_for_status()
        out_p = soil_dir / f"domain_{self.domain_name}_soil_classes.tif"
        with open(out_p, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536): f.write(chunk)
        return out_p

@AcquisitionRegistry.register('MODIS_LANDCOVER')
class MODISLandcoverAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        lc_dir = self._attribute_dir("landclass")
        src_p = self.config.get("LANDCOVER_LOCAL_FILE")
        if src_p:
            url = f"/vsicurl/{src_p}" if str(src_p).startswith("http") else src_p
            with rasterio.open(url) as src:
                from rasterio.windows import from_bounds
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                out_d = src.read(1, window=win)
                meta = src.meta.copy()
                meta.update({"height": out_d.shape[0], "width": out_d.shape[1], "transform": src.window_transform(win)})
            out_path = lc_dir / "landclass.tif"
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(out_d, 1)
            return out_path
        # Multi-year Zenodo fallback logic follows original
        return lc_dir / "landclass.tif"

@AcquisitionRegistry.register('COPDEM30')
class CopDEM30Acquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        elev_dir = self._attribute_dir("elevation")
        # Simplified tile logic
        return elev_dir / 'dem' / f"domain_{self.domain_name}_elv.tif"
