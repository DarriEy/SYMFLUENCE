import math
from pathlib import Path
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds
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
    def _download_with_size_check(self, url: str, dest_path: Path) -> None:
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            expected_size = resp.headers.get("Content-Length")
            with open(tmp_path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    handle.write(chunk)

        if expected_size:
            expected_size = int(expected_size)
            actual_size = tmp_path.stat().st_size
            if actual_size != expected_size:
                tmp_path.unlink(missing_ok=True)
                raise IOError(
                    f"Downloaded size mismatch for {url}: "
                    f"{actual_size} != {expected_size}"
                )
        tmp_path.replace(dest_path)

    def download(self, output_dir: Path) -> Path:
        lc_dir = self._attribute_dir("landclass")
        land_name = self.config.get("LAND_CLASS_NAME", "default")
        if land_name == "default":
            land_name = f"domain_{self.domain_name}_land_classes.tif"
        out_path = lc_dir / land_name

        src_p = self.config.get("LANDCOVER_LOCAL_FILE")
        if src_p:
            url = f"/vsicurl/{src_p}" if str(src_p).startswith("http") else src_p
            with rasterio.open(url) as src:
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                out_d = src.read(1, window=win)
                meta = src.meta.copy()
                meta.update({"height": out_d.shape[0], "width": out_d.shape[1], "transform": src.window_transform(win)})
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(out_d, 1)
            return out_path
        # Multi-year Zenodo fallback logic follows Fire-Engine-Framework legacy fetcher
        self.logger.info("Fetching MODIS Land Cover (MCD12Q1 v061) from Zenodo")
        self.logger.info(f"MODIS landcover bbox: {self.bbox}")
        years = self.config.get("MODIS_LANDCOVER_YEARS")
        if isinstance(years, (list, tuple)):
            years = [int(y) for y in years]
        else:
            start_year = self.config.get("MODIS_LANDCOVER_START_YEAR")
            end_year = self.config.get("MODIS_LANDCOVER_END_YEAR")
            if start_year and end_year:
                years = list(range(int(start_year), int(end_year) + 1))
            else:
                landcover_year = self.config.get("LANDCOVER_YEAR")
                if landcover_year:
                    years = [int(landcover_year)]
                else:
                    years = [2019]

        base_url = self.config.get(
            "MODIS_LANDCOVER_BASE_URL",
            "https://zenodo.org/records/8367523/files",
        )
        cache_dir = Path(
            self.config.get(
                "MODIS_LANDCOVER_CACHE_DIR",
                self.domain_dir / "cache" / "modis_landcover",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = []
        out_meta = None

        for year in years:
            fname = (
                "lc_mcd12q1v061.t1_c_500m_s_"
                f"{year}0101_{year}1231_go_epsg.4326_v20230818.tif"
            )
            url = f"{base_url}/{fname}"
            local_tmp = cache_dir / fname

            for attempt in range(2):
                try:
                    if not local_tmp.exists():
                        self._download_with_size_check(url, local_tmp)

                    with rasterio.open(local_tmp) as src:
                        win = from_bounds(
                            self.bbox["lon_min"],
                            self.bbox["lat_min"],
                            self.bbox["lon_max"],
                            self.bbox["lat_max"],
                            src.transform,
                        )
                        data = src.read(1, window=win)
                        if out_meta is None:
                            out_transform = src.window_transform(win)
                            out_meta = src.meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": data.shape[0],
                                "width": data.shape[1],
                                "transform": out_transform,
                                "compress": "lzw",
                            })

                        if data.shape == (out_meta["height"], out_meta["width"]):
                            arrays.append(data)
                        else:
                            self.logger.warning(
                                f"MODIS landcover shape mismatch for {year}: {data.shape}"
                            )
                    break
                except rasterio.errors.RasterioIOError as exc:
                    self.logger.warning(
                        f"MODIS landcover read failed for {year} (attempt {attempt + 1}/2): {exc}"
                    )
                    local_tmp.unlink(missing_ok=True)
                except Exception as exc:
                    self.logger.warning(
                        f"MODIS landcover download failed for {year} (attempt {attempt + 1}/2): {exc}"
                    )
                    local_tmp.unlink(missing_ok=True)
                    if attempt == 1:
                        raise

        if not arrays:
            raise FileNotFoundError("No MODIS land cover data processed from Zenodo.")

        stack = np.stack(arrays, axis=0)

        def calc_mode(arr):
            valid = arr[arr != 255]
            if valid.size == 0:
                return 255
            vals, counts = np.unique(valid, return_counts=True)
            return vals[np.argmax(counts)]

        mode_data = np.apply_along_axis(calc_mode, 0, stack).astype("uint8")

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mode_data, 1)
        return out_path

@AcquisitionRegistry.register('COPDEM30')
class CopDEM30Acquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if out_path.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_path

        self.logger.info(f"Downloading Copernicus DEM GLO-30 for bbox: {self.bbox}")
        
        # AWS S3 Public Dataset: eu-central-1
        base_url = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
        
        # Tiles are 1x1 degree
        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
                    url = f"{base_url}/{tile_name}/{tile_name}.tif"
                    
                    local_tile = dem_dir / f"temp_{tile_name}.tif"
                    if not local_tile.exists():
                        self.logger.info(f"Fetching tile: {tile_name}")
                        with requests.get(url, stream=True, timeout=60) as r:
                            if r.status_code == 200:
                                with open(local_tile, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                                tile_paths.append(local_tile)
                            else:
                                self.logger.warning(f"Tile {tile_name} not found (status {r.status_code})")
                    else:
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No Copernicus DEM tiles found for bbox: {self.bbox}")

            if len(tile_paths) == 1:
                if out_path.exists(): out_path.unlink()
                tile_paths[0].rename(out_path)
            else:
                self.logger.info(f"Merging {len(tile_paths)} tiles into {out_path}")
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw"
                })
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for src in src_files: src.close()
                for p in tile_paths: p.unlink(missing_ok=True)

        except Exception as e:
            self.logger.error(f"Error downloading/processing Copernicus DEM: {e}")
            for p in tile_paths: 
                if p.exists() and p != out_path: p.unlink(missing_ok=True)
            raise

        return out_path

@AcquisitionRegistry.register('FABDEM')
class FABDEMAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        dem_dir.mkdir(parents=True, exist_ok=True)
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"

        if out_path.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_path

        self.logger.info(f"Downloading FABDEM for bbox: {self.bbox}")
        # Source Cooperative (AWS)
        base_url = "https://data.source.coop/c_6_6/fabdem/tiles"
        
        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        try:
            for lat in range(lat_min, lat_max):
                for lon in range(lon_min, lon_max):
                    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                    # FABDEM format: N46W122_FABDEM_V1-2.tif
                    tile_name = f"{lat_str}{lon_str}_FABDEM_V1-2"
                    url = f"{base_url}/{tile_name}.tif"
                    
                    local_tile = dem_dir / f"temp_fab_{tile_name}.tif"
                    if not local_tile.exists():
                        self.logger.info(f"Fetching FABDEM tile: {tile_name}")
                        with requests.get(url, stream=True, timeout=60) as r:
                            if r.status_code == 200:
                                with open(local_tile, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                                tile_paths.append(local_tile)
                    else:
                        tile_paths.append(local_tile)

            if not tile_paths:
                raise FileNotFoundError(f"No FABDEM tiles found for bbox: {self.bbox}")

            if len(tile_paths) == 1:
                if out_path.exists(): out_path.unlink()
                tile_paths[0].rename(out_path)
            else:
                src_files = [rasterio.open(p) for p in tile_paths]
                mosaic, out_trans = rio_merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
                with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
                for src in src_files: src.close()
                for p in tile_paths: p.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Error with FABDEM: {e}")
            raise
        return out_path

@AcquisitionRegistry.register('NASADEM_LOCAL')
class NASADEMLocalAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        local_src_dir = Path(self.config.get("NASADEM_LOCAL_DIR"))
        elev_dir = self._attribute_dir("elevation")
        dem_dir = elev_dir / 'dem'
        out_path = dem_dir / f"domain_{self.domain_name}_elv.tif"
        
        if not local_src_dir.exists():
            raise FileNotFoundError(f"NASADEM_LOCAL_DIR not found: {local_src_dir}")

        lat_min = math.floor(self.bbox['lat_min'])
        lat_max = math.ceil(self.bbox['lat_max'])
        lon_min = math.floor(self.bbox['lon_min'])
        lon_max = math.ceil(self.bbox['lon_max'])

        tile_paths = []
        for lat in range(lat_min, lat_max):
            for lon in range(lon_min, lon_max):
                lat_str = f"n{lat:02d}" if lat >= 0 else f"s{-lat:02d}"
                lon_str = f"e{lon:03d}" if lon >= 0 else f"w{-lon:03d}"
                # Common NASADEM format: n46w122.hgt or .tif
                pattern = f"{lat_str}{lon_str}*.tif"
                matches = list(local_src_dir.glob(pattern))
                if not matches:
                    pattern = f"{lat_str}{lon_str}*.hgt"
                    matches = list(local_src_dir.glob(pattern))
                
                if matches:
                    tile_paths.append(matches[0])

        if not tile_paths:
            raise FileNotFoundError(f"No NASADEM tiles found in {local_src_dir} for bbox {self.bbox}")

        if len(tile_paths) == 1:
            # We don't want to move original files, so we crop/copy
            with rasterio.open(tile_paths[0]) as src:
                win = from_bounds(self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max'], self.bbox['lat_max'], src.transform)
                data = src.read(1, window=win)
                meta = src.meta.copy()
                meta.update({"height": data.shape[0], "width": data.shape[1], "transform": src.window_transform(win)})
            with rasterio.open(out_path, "w", **meta) as dst: dst.write(data, 1)
        else:
            src_files = [rasterio.open(p) for p in tile_paths]
            mosaic, out_trans = rio_merge(src_files)
            out_meta = src_files[0].meta.copy()
            out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
            with rasterio.open(out_path, "w", **out_meta) as dest: dest.write(mosaic)
            for src in src_files: src.close()
            
        return out_path
