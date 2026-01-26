"""SMAP Soil Moisture Acquisition Handler

Provides cloud acquisition for NASA SMAP (Soil Moisture Active Passive) data:
- Primary: NSIDC THREDDS/NCSS aggregated access
- Fallback: CMR granule-by-granule download

SMAP Overview:
    Data Type: Satellite-derived soil moisture
    Resolution: ~9km (L4 assimilated product)
    Coverage: Global (except polar regions)
    Source: NASA NSIDC DAAC

Authentication:
    Requires NASA Earthdata Login credentials:
    - ~/.netrc file with machine urs.earthdata.nasa.gov entry
    - Or EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables
"""

import re
import requests
import numpy as np
import netCDF4 as nc
from pathlib import Path
import xarray as xr

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
        thredds_base = self.config_dict.get('SMAP_THREDDS_BASE', "https://n5eil01u.ecs.nsidc.org/thredds/ncss/grid")
        product = self._get_config_value(lambda: self.config.evaluation.smap.product, default='SMAP_L4_SM_gph_v4', dict_key='SMAP_PRODUCT')
        if isinstance(product, str) and product.upper() == 'SPL4SMGP':
            product = 'SMAP_L4_SM_gph_v4'

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        start_date = self.start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date = self.end_date.strftime("%Y-%m-%dT23:59:59Z")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_nc = output_dir / f"{self.domain_name}_SMAP_raw.nc"

        if self._skip_if_exists(out_nc):
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
        override_url = self.config_dict.get('SMAP_THREDDS_URL')
        if override_url:
            candidate_urls = [override_url]
        else:
            candidate_products = [product]
            if isinstance(product, str) and product == 'SMAP_L4_SM_gph_v4':
                candidate_products.extend(['SMAP_L4_SM_gph', 'SMAP_L4_SM_gph_v5'])
            elif isinstance(product, str) and product != 'SMAP_L4_SM_gph_v4':
                candidate_products.append('SMAP_L4_SM_gph_v4')
            candidate_urls = [f"{thredds_base}/{p}/aggregated.ncml" for p in candidate_products]

        # Setup session with Earthdata Auth (check .netrc first, then env vars)
        session = requests.Session()
        user, password = self._get_earthdata_credentials()
        if user and password:
            session.auth = (user, password)

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
        cmr_short_name = self.config_dict.get('SMAP_CMR_SHORT_NAME', self._get_config_value(lambda: self.config.evaluation.smap.product, default='SPL4SMGP', dict_key='SMAP_PRODUCT'))
        if isinstance(cmr_short_name, str) and cmr_short_name.upper().startswith('SMAP_L4_SM_GPH'):
            cmr_short_name = 'SPL4SMGP'
        cmr_version = str(self.config_dict.get('SMAP_CMR_VERSION', '008')).zfill(3)
        max_granules = self._get_config_value(lambda: self.config.evaluation.smap.max_granules, dict_key='SMAP_MAX_GRANULES')
        use_opendap = bool(self._get_config_value(lambda: self.config.evaluation.smap.use_opendap, default=False, dict_key='SMAP_USE_OPENDAP'))

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
        user, password = self._get_earthdata_credentials()
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
                        link.get("href") for link in links
                        if "service#" in link.get("rel", "") and "opendap" in link.get("href", "")
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
                    if out_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
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
            lat_grid[y_slice, x_slice]
            lon_grid[y_slice, x_slice]
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
                    except Exception as e:
                        self.logger.debug(f"Could not convert time value {time_val} with units '{time_units}': {e}")
                for sm_name, data in data_vars.items():
                    data_vars[sm_name] = np.expand_dims(data, axis=0)
                out_dims = ["time"] + out_dims
                coords["time"] = [time_val]

        granule_id = url.rstrip("/").split("/")[-1].replace(".h5", "")
        out_file = output_dir / f"{granule_id}_subset.nc"
        if out_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            return True
        self.logger.info(f"Writing SMAP subset: {out_file.name}")
        ds_out = xr.Dataset(
            {sm_name: xr.DataArray(data, dims=out_dims, coords=coords) for sm_name, data in data_vars.items()}
        )
        ds_out.to_netcdf(out_file)
        return True
