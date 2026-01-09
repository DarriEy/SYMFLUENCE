"""
Observation Data Acquisition Handlers

Provides cloud acquisition for various observational datasets:
- SMAP Soil Moisture (NSIDC THREDDS/NCSS)
- ESA CCI Soil Moisture (CDS API)
- FLUXCOM Evapotranspiration
"""
import logging
import re
import requests
import pandas as pd
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import xarray as xr
import os

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('SMAP')
class SMAPAcquirer(BaseAcquisitionHandler):
    """
    Acquires SMAP Soil Moisture data via NSIDC THREDDS NCSS.
    Requires Earthdata Login credentials (via ~/.netrc or env vars).
    """

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Starting SMAP Soil Moisture acquisition via NSIDC THREDDS")
        
        # NSIDC THREDDS NCSS endpoint
        thredds_base = self.config.get('SMAP_THREDDS_BASE', "https://n5eil01u.ecs.nsidc.org/thredds/ncss/grid")
        product = self.config.get('SMAP_PRODUCT', 'SMAP_L4_SM_gph_v4')
        if isinstance(product, str) and product.upper() == 'SPL4SMGP':
            product = 'SMAP_L4_SM_gph_v4'
        
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
        
        start_date = self.start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date = self.end_date.strftime("%Y-%m-%dT23:59:59Z")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        out_nc = output_dir / f"{self.domain_name}_SMAP_raw.nc"
        
        if out_nc.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_nc

        params = {
            "var": "sm_surface",
            "north": lat_max,
            "south": lat_min,
            "west": lon_min,
            "east": lon_max,
            "time_start": start_date,
            "time_end": end_date,
            "accept": "netcdf4"
        }
        
        # Construct URL for the specific product/version/date
        # Note: NSIDC structure is complex (YYYY.MM.DD directories).
        # For NCSS Grid, we often need the exact file path or an aggregation.
        # This implementation assumes an aggregated NCML exists or the user provides a direct OPeNDAP/NCSS URL.
        # If 'SMAP_THREDDS_URL' is provided, use it directly.
        override_url = self.config.get('SMAP_THREDDS_URL')
        if override_url:
            candidate_urls = [override_url]
        else:
            candidate_products = [product]
            if isinstance(product, str) and product == 'SMAP_L4_SM_gph_v4':
                candidate_products.extend(['SMAP_L4_SM_gph', 'SMAP_L4_SM_gph_v5'])
            elif isinstance(product, str) and product != 'SMAP_L4_SM_gph_v4':
                candidate_products.append('SMAP_L4_SM_gph_v4')
            candidate_urls = [f"{thredds_base}/{p}/aggregated.ncml" for p in candidate_products]

        # Setup session with Earthdata Auth
        session = requests.Session()
        user = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        
        if user and password:
            session.auth = (user, password)
        else:
            # Check for .netrc
            pass # requests handles .netrc automatically
            
        last_error = None
        for url in candidate_urls:
            self.logger.info(f"Querying SMAP THREDDS: {url}")
            try:
                response = session.get(url, params=params, stream=True, timeout=600)
                
                # Handle redirects for auth
                if response.status_code == 401:
                    self.logger.error("Authentication failed. Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD or use a .netrc file.")
                    raise PermissionError("Earthdata Login required")

                if response.status_code == 404:
                    self.logger.warning(f"SMAP THREDDS URL not found: {url}")
                    last_error = requests.HTTPError(f"404 Client Error: Not Found for url: {response.url}")
                    continue
                    
                response.raise_for_status()
                
                with open(out_nc, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                
                self.logger.info(f"Successfully downloaded SMAP data to {out_nc}")
                return out_nc
            except Exception as e:
                last_error = e

        self.logger.warning(f"SMAP THREDDS acquisition failed: {last_error}")
        return self._download_via_cmr(output_dir)

    def _download_via_cmr(self, output_dir: Path) -> Path:
        """Fallback: download SMAP granules via CMR HTTPS links."""
        cmr_short_name = self.config.get('SMAP_CMR_SHORT_NAME', self.config.get('SMAP_PRODUCT', 'SPL4SMGP'))
        if isinstance(cmr_short_name, str) and cmr_short_name.upper().startswith('SMAP_L4_SM_GPH'):
            cmr_short_name = 'SPL4SMGP'
        cmr_version = str(self.config.get('SMAP_CMR_VERSION', '008')).zfill(3)
        max_granules = self.config.get('SMAP_MAX_GRANULES')
        use_opendap = bool(self.config.get('SMAP_USE_OPENDAP', False))

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
        temporal = f"{self.start_date.strftime('%Y-%m-%dT%H:%M:%SZ')},{self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        params = {
            "short_name": cmr_short_name,
            "version": cmr_version,
            "temporal": temporal,
            "bounding_box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "page_size": 2000,
            "page_num": 1,
        }

        self.logger.info(
            "Falling back to CMR granule search "
            f"(short_name={cmr_short_name}, version={cmr_version})"
        )

        session = requests.Session()
        user = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if user and password:
            session.auth = (user, password)

        downloaded = 0
        attempts = 0
        while True:
            resp = session.get("https://cmr.earthdata.nasa.gov/search/granules.json", params=params, timeout=600)
            resp.raise_for_status()
            entries = resp.json().get("feed", {}).get("entry", [])
            if not entries:
                break

            for entry in entries:
                if max_granules and attempts >= int(max_granules):
                    self.logger.warning("Reached SMAP_MAX_GRANULES limit; stopping downloads")
                    return output_dir
                links = entry.get("links", [])
                if use_opendap:
                    opendap_links = [
                        l.get("href") for l in links
                        if "service#" in l.get("rel", "") and "opendap" in l.get("href", "")
                    ]
                    if opendap_links:
                        attempts += 1
                        if self._download_subset_from_opendap(
                            opendap_links[0],
                            output_dir,
                            lat_min,
                            lat_max,
                            lon_min,
                            lon_max,
                        ):
                            downloaded += 1
                        continue

                data_links = []
                for link in links:
                    href = link.get("href")
                    if not href or "data#" not in link.get("rel", ""):
                        continue
                    if href.endswith((".h5", ".hdf5", ".nc")):
                        data_links.append(href)
                for href in data_links:
                    attempts += 1
                    if max_granules and downloaded >= int(max_granules):
                        self.logger.warning("Reached SMAP_MAX_GRANULES limit; stopping downloads")
                        return output_dir
                    filename = href.split("/")[-1]
                    out_file = output_dir / filename
                    if out_file.exists() and not self.config.get('FORCE_DOWNLOAD', False):
                        downloaded += 1
                        continue
                    self.logger.info(f"Downloading SMAP granule: {filename}")
                    tmp_file = out_file.with_suffix(out_file.suffix + ".part")
                    if tmp_file.exists():
                        tmp_file.unlink()
                    with session.get(href, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        with open(tmp_file, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)
                    tmp_file.replace(out_file)
                    downloaded += 1

            params["page_num"] += 1

        if downloaded == 0:
            raise RuntimeError("No SMAP granules found via CMR for the requested bounds/time range.")

        self.logger.info(f"Downloaded {downloaded} SMAP granules to {output_dir}")
        return output_dir

    def _download_subset_from_opendap(
        self,
        url: str,
        output_dir: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> bool:
        """Download a spatial subset from OPeNDAP and write as NetCDF."""
        try:
            import netrc
            from pydap.cas.urs import setup_session
            from pydap.client import open_url

            auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
            if not auth:
                auth = netrc.netrc().authenticators("opendap.earthdata.nasa.gov")
            if not auth:
                self.logger.warning("OPeNDAP access denied; missing Earthdata credentials in ~/.netrc")
                return False

            user, _, password = auth
            session = setup_session(user, password, check_url=url)
            dap_url = url.replace("https://", "dap4://")
            dataset = open_url(dap_url, session=session)
        except Exception as exc:
            if "Access denied" in str(exc):
                self.logger.warning(
                    "OPeNDAP access denied; check ~/.netrc or Earthdata login"
                )
            exc_msg = re.sub(r"https://[^@]+@", "https://<redacted>@", str(exc))
            self.logger.warning(f"Failed to open OPeNDAP dataset {url}: {exc_msg}")
            return False

        geo_group = dataset.get("Geophysical_Data")
        if geo_group is None:
            self.logger.warning(f"OPeNDAP dataset missing Geophysical_Data group: {url}")
            return False

        def _find_group_vars(group, candidates):
            matches = []
            for name in group.keys():
                for candidate in candidates:
                    if candidate in name.lower():
                        matches.append(name)
                        break
            return matches

        sm_names = _find_group_vars(geo_group, ["sm_surface", "soil_moisture", "rootzone"])
        if not sm_names:
            self.logger.warning(f"OPeNDAP dataset missing SMAP soil moisture variable: {url}")
            return False

        if "cell_lat" in dataset and "cell_lon" in dataset:
            lat_grid = np.asarray(dataset["cell_lat"][:])
            lon_grid = np.asarray(dataset["cell_lon"][:])
            mask = (
                (lat_grid >= lat_min) & (lat_grid <= lat_max) &
                (lon_grid >= lon_min) & (lon_grid <= lon_max)
            )
            if not np.any(mask):
                self.logger.warning(f"No SMAP grid cells intersect bbox for {url}")
                return False
            rows, cols = np.where(mask)
            y_slice = slice(int(rows.min()), int(rows.max()) + 1)
            x_slice = slice(int(cols.min()), int(cols.max()) + 1)
            lat_vals = lat_grid[y_slice, x_slice]
            lon_vals = lon_grid[y_slice, x_slice]
        else:
            self.logger.warning(f"OPeNDAP dataset missing cell_lat/cell_lon: {url}")
            return False

        out_dims = ["y", "x"]
        coords = {
            "y": np.asarray(dataset["y"][y_slice]),
            "x": np.asarray(dataset["x"][x_slice]),
        }
        data_vars = {}
        for sm_name in sm_names:
            var = geo_group[sm_name]
            dims = [d.lstrip("/") for d in list(getattr(var, "dimensions", ()))]
            if not dims:
                self.logger.warning(f"Unknown dimension layout for {sm_name} in {url}")
                return False
            data_vars[sm_name] = np.asarray(var[y_slice, x_slice])
        if "time" in dataset:
            time_vals = np.asarray(dataset["time"][:])
            if time_vals.size:
                time_val = time_vals.flat[0]
                time_units = dataset["time"].attributes.get("units")
                time_calendar = dataset["time"].attributes.get("calendar", "standard")
                if time_units:
                    try:
                        time_val = nc.num2date(time_val, time_units, calendar=time_calendar)
                    except Exception:
                        pass
                for sm_name, data in data_vars.items():
                    data_vars[sm_name] = np.expand_dims(data, axis=0)
                out_dims = ["time"] + out_dims
                coords["time"] = [time_val]

        granule_id = url.rstrip("/").split("/")[-1].replace(".h5", "")
        out_file = output_dir / f"{granule_id}_subset.nc"
        if out_file.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return True
        self.logger.info(f"Writing SMAP subset: {out_file.name}")
        ds_out = xr.Dataset(
            {sm_name: xr.DataArray(data, dims=out_dims, coords=coords) for sm_name, data in data_vars.items()}
        )
        ds_out.to_netcdf(out_file)
        return True


@AcquisitionRegistry.register('ISMN')
class ISMNAcquirer(BaseAcquisitionHandler):
    """
    Acquires ISMN soil moisture data via an API or metadata URL.
    """

    def download(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        force_download = bool(self.config.get('FORCE_DOWNLOAD', False))
        if any(output_dir.glob("*.csv")) and not force_download:
            return output_dir

        api_base = self.config.get('ISMN_API_BASE', 'https://ismn.geo.tuwien.ac.at/api/v1').rstrip("/")
        metadata_url = self.config.get('ISMN_METADATA_URL') or f"{api_base}/stations"
        data_template = self.config.get('ISMN_DATA_URL_TEMPLATE')
        if not data_template:
            data_template = f"{api_base}/stations/{{station_id}}/download?format=csv&start_date={{start_date}}&end_date={{end_date}}"

        session = requests.Session()
        auth = self._resolve_auth(metadata_url)
        if auth:
            session.auth = auth

        stations = self._load_station_metadata(session, metadata_url)
        if stations is None or stations.empty:
            raise RuntimeError("ISMN station metadata could not be loaded.")

        stations = self._select_stations(stations)
        if stations.empty:
            raise RuntimeError("No ISMN stations found for the requested domain.")

        selection_file = output_dir / "ismn_station_selection.csv"
        stations.to_csv(selection_file, index=False)

        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = self.end_date.strftime("%Y-%m-%d")

        downloaded = 0
        for _, row in stations.iterrows():
            station_id = str(row["station_id"])
            station_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", station_id).strip("_")
            out_file = output_dir / f"{station_slug}.csv"
            if out_file.exists() and not force_download:
                downloaded += 1
                continue

            data_url = data_template.format(
                station_id=station_id,
                start_date=start_date,
                end_date=end_date,
            )
            auth = self._resolve_auth(data_url)
            if auth:
                session.auth = auth

            self.logger.info(f"Downloading ISMN station data: {station_id}")
            try:
                resp = session.get(data_url, timeout=600)
                resp.raise_for_status()
                out_file.write_bytes(resp.content)
                downloaded += 1
            except Exception as exc:
                self.logger.warning(f"Failed to download ISMN station {station_id}: {exc}")

        if downloaded == 0:
            raise RuntimeError("No ISMN station data downloaded.")

        return output_dir

    def _resolve_auth(self, url: str):
        user = self.config.get("ISMN_USERNAME") or os.environ.get("ISMN_USERNAME")
        password = self.config.get("ISMN_PASSWORD") or os.environ.get("ISMN_PASSWORD")
        if user and password:
            return (user, password)

        try:
            import netrc
            from urllib.parse import urlparse

            host = urlparse(url).hostname
            if not host:
                return None
            auth = netrc.netrc().authenticators(host)
            if auth:
                user, _, password = auth
                return (user, password)
        except Exception:
            return None
        return None

    def _load_station_metadata(self, session: requests.Session, metadata_url: str) -> Optional[pd.DataFrame]:
        try:
            resp = session.get(metadata_url, timeout=600)
            resp.raise_for_status()
        except Exception as exc:
            self.logger.warning(f"Failed to fetch ISMN station metadata: {exc}")
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type or metadata_url.endswith(".json"):
            try:
                payload = resp.json()
            except ValueError as exc:
                self.logger.warning(f"Failed to parse ISMN metadata JSON: {exc}")
                return None
            records = payload
            if isinstance(payload, dict):
                for key in ("data", "stations", "items"):
                    if key in payload:
                        records = payload[key]
                        break
            if not isinstance(records, list):
                self.logger.warning("Unexpected ISMN metadata JSON structure.")
                return None
            df = pd.DataFrame(records)
        else:
            from io import StringIO

            df = pd.read_csv(StringIO(resp.text))

        if df.empty:
            return None

        col_map = self._normalize_station_columns(df.columns)
        df = df.rename(columns=col_map)
        required = {"station_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            self.logger.warning("ISMN metadata missing station_id/latitude/longitude columns.")
            return None

        cols = ["station_id", "latitude", "longitude"]
        if "network" in df.columns:
            cols.append("network")
        return df[cols].copy()

    def _normalize_station_columns(self, columns):
        col_map = {}
        for col in columns:
            lower = col.lower()
            if lower in ("station_id", "station", "stationid", "id"):
                col_map[col] = "station_id"
            elif "lat" in lower:
                col_map[col] = "latitude"
            elif "lon" in lower or "lng" in lower:
                col_map[col] = "longitude"
            elif "network" in lower:
                col_map[col] = "network"
        return col_map

    def _select_stations(self, stations: pd.DataFrame) -> pd.DataFrame:
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        all_stations = stations.copy()
        stations = all_stations[
            (stations["latitude"] >= lat_min) & (stations["latitude"] <= lat_max) &
            (stations["longitude"] >= lon_min) & (stations["longitude"] <= lon_max)
        ]

        if stations.empty:
            stations = all_stations.copy()

        stations["distance_km"] = self._haversine_km(
            stations["latitude"].astype(float),
            stations["longitude"].astype(float),
            lat_center,
            lon_center,
        )

        radius_km = self.config.get("ISMN_SEARCH_RADIUS_KM")
        if radius_km:
            stations = stations[stations["distance_km"] <= float(radius_km)]

        max_stations = int(self.config.get("ISMN_MAX_STATIONS", 3))
        stations = stations.sort_values("distance_km").head(max_stations)
        return stations

    def _haversine_km(self, lat_series, lon_series, lat0, lon0):
        lat1 = np.radians(lat_series.astype(float))
        lon1 = np.radians(lon_series.astype(float))
        lat2 = np.radians(float(lat0))
        lon2 = np.radians(float(lon0))

        dlat = lat1 - lat2
        dlon = lon1 - lon2

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

@AcquisitionRegistry.register('ESA_CCI_SM')
class ESACCISMAcquirer(BaseAcquisitionHandler):
    """
    Acquires ESA CCI Soil Moisture data via Copernicus CDS.
    """

    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required for ESA CCI SM acquisition")
            
        self.logger.info("Starting ESA CCI Soil Moisture acquisition via CDS")
        
        c = cdsapi.Client()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        # CDS returns a tar/zip for multiple files
        out_file = output_dir / f"{self.domain_name}_ESA_CCI_SM_raw.tar.gz"
        
        if out_file.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_file

        years = sorted(list(set([str(y) for y in range(self.start_date.year, self.end_date.year + 1)])))
        
        request = {
            'variable': 'volumetric_surface_soil_moisture',
            'type_of_sensor': 'combined',
            'time_aggregation': 'day_average',
            'year': years,
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'area': [self.bbox['lat_max'], self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max']],
            'format': 'tgz',
            'version': 'v07.1' # Specify version to ensure consistency
        }
        
        c.retrieve('satellite-soil-moisture', request, str(out_file))
        
        self.logger.info(f"ESA CCI SM data downloaded to {out_file}")
        
        # Auto-extract
        import tarfile
        extract_dir = output_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(out_file, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            
        self.logger.info(f"Extracted ESA CCI SM data to {extract_dir}")
        return extract_dir

@AcquisitionRegistry.register('FLUXCOM_ET')
class FLUXCOMETAcquirer(BaseAcquisitionHandler):
    """
    Acquires FLUXCOM Evapotranspiration data.
    Since FLUXCOM does not have a public API, this handler supports:
    1. Direct download from a user-provided URL (e.g., internal server, S3 presigned).
    2. Local file discovery (if data is manually placed).
    """

    def download(self, output_dir: Path) -> Path:
        et_dir = output_dir / "et" / "fluxcom"
        et_dir.mkdir(parents=True, exist_ok=True)
        
        # Option 1: Check for existing local files
        local_pattern = self.config.get('FLUXCOM_FILE_PATTERN', "*.nc")
        existing_files = list(et_dir.glob(local_pattern))
        if existing_files and not self.config.get('FORCE_DOWNLOAD', False):
            self.logger.info(f"Found existing FLUXCOM files in {et_dir}")
            return et_dir

        # Option 2: Download from URL
        download_url = self.config.get('FLUXCOM_DOWNLOAD_URL')
        if download_url:
            self.logger.info(f"Downloading FLUXCOM data from {download_url}")
            out_file = et_dir / "fluxcom_downloaded.nc" # Assuming single file or archive
            try:
                response = requests.get(download_url, stream=True, timeout=600)
                response.raise_for_status()
                with open(out_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                return et_dir
            except Exception as e:
                self.logger.error(f"Failed to download FLUXCOM data: {e}")
                raise

        # Fail if no method works
        raise ValueError(
            "FLUXCOM acquisition failed: No local files found and 'FLUXCOM_DOWNLOAD_URL' not set. "
            "Please manually download FLUXCOM data to: " + str(et_dir)
        )
