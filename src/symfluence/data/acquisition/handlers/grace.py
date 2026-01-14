"""
GRACE Data Acquisition Handler

Provides cloud acquisition for GRACE/GRACE-FO Terrestrial Water Storage anomaly data.
Retrieves data from NASA PO.DAAC or similar cloud-hosted repositories.
"""
import os
import netrc
import requests
import xarray as xr
from pathlib import Path
from typing import Any, Optional, Tuple
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('GRACE')
class GRACEAcquirer(BaseAcquisitionHandler):
    """
    Handles GRACE/GRACE-FO data acquisition.
    Currently focuses on the JPL/CSR/GSFC Mascon solutions.
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download GRACE data (JPL, CSR, and GSFC Mascon RL06v02).
        """
        self.logger.info("Starting GRACE data acquisition (JPL, CSR, GSFC)")
        output_dir.mkdir(parents=True, exist_ok=True)

        subset_enabled = self._parse_bool(self.config.get('GRACE_SUBSET', False))
        force_download = self._parse_bool(self.config.get('FORCE_DOWNLOAD', False))

        datasets = {
            'jpl': {
                'filename': 'GRCTellus.JPL.200204_202211.GLO.RL06M.MSCNv02CRI.nc',
                'url': 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/tellus/L3/grace/nasajpl/RL06_v02/GRCTellus.JPL.200204_202211.GLO.RL06M.MSCNv02CRI.nc'
            },
            'csr': {
                'filename': 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc',
                'url': 'https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
            },
            'gsfc': {
                'filename': 'gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc',
                'url': 'https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc'
            }
        }

        success_count = 0
        earthdata_auth = self._get_earthdata_auth()

        for center, info in datasets.items():
            target_file = output_dir / info['filename']
            subset_file = target_file.with_name(f"{target_file.stem}_subset.nc")
            url = info['url']

            if subset_enabled and center == 'jpl' and subset_file.exists() and not force_download:
                self.logger.info(f"GRACE {center.upper()} subset already exists: {subset_file}")
                success_count += 1
                continue
            if target_file.exists() and not force_download:
                self.logger.info(f"GRACE {center.upper()} file already exists: {target_file}")
                success_count += 1
                continue

            self.logger.info(f"Downloading {center.upper()} from {url}")
            try:
                if subset_enabled and center == 'jpl':
                    if self._download_jpl_subset(url, subset_file):
                        self.logger.info(f"Successfully downloaded GRACE {center.upper()} subset to {subset_file}")
                        success_count += 1
                        continue
                    self.logger.warning("JPL subset download failed; falling back to full file download.")

                if center == 'jpl':
                    cmr_path = self._download_jpl_from_cmr(output_dir, earthdata_auth, force_download)
                    if cmr_path is not None:
                        self.logger.info(f"Successfully downloaded GRACE {center.upper()} data to {cmr_path}")
                        success_count += 1
                        continue

                session = requests.Session()
                if earthdata_auth and center == 'jpl':
                    session.auth = earthdata_auth
                # Use verify=False to bypass SSL errors (e.g. CSR)
                with session.get(url, stream=True, timeout=120, verify=False) as r:
                    r.raise_for_status()
                    with open(target_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                self.logger.info(f"Successfully downloaded GRACE {center.upper()} data to {target_file}")
                success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to download GRACE {center.upper()} data: {e}")
                self.logger.warning(f"Please manually download the {center.upper()} Mascon NetCDF file and place it in the observation directory if automatic download fails.")

        if success_count == 0:
            raise RuntimeError("Failed to acquire any GRACE data.")

        return output_dir

    def _parse_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {'true', '1', 'yes', 'y'}
        return bool(value)

    def _get_earthdata_auth(self) -> Optional[Tuple[str, str]]:
        user = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        if user and password:
            return (user, password)

        try:
            auth_info = netrc.netrc()
        except (FileNotFoundError, netrc.NetrcParseError):
            return None

        for host in ("urs.earthdata.nasa.gov", "podaac-opendap.jpl.nasa.gov", "opendap.earthdata.nasa.gov"):
            auth = auth_info.authenticators(host)
            if auth:
                return (auth[0], auth[2])

        return None

    def _download_jpl_subset(self, url: str, target_file: Path) -> bool:
        if not self.bbox:
            self.logger.warning("GRACE_SUBSET requested but BOUNDING_BOX_COORDS not set.")
            return False

        try:
            ds = xr.open_dataset(url)
        except Exception as exc:
            self.logger.warning(f"Failed to open JPL OPeNDAP dataset: {exc}")
            return False

        try:
            subset = self._subset_grace_dataset(ds)
            subset.to_netcdf(target_file)
            return True
        except Exception as exc:
            self.logger.warning(f"Failed to subset JPL dataset: {exc}")
            return False
        finally:
            try:
                ds.close()
            except Exception:
                pass

    def _subset_grace_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        lat_name = self._get_coord_name(ds, ("lat", "latitude"))
        lon_name = self._get_coord_name(ds, ("lon", "longitude"))
        if not lat_name or not lon_name:
            raise ValueError("GRACE dataset missing expected lat/lon coordinates")

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        lon_vals = ds[lon_name].values
        if lon_vals.max() > 180 and (lon_min < 0 or lon_max < 0):
            lon_min = lon_min % 360
            lon_max = lon_max % 360

        if lon_min <= lon_max:
            lon_subset = ds.sel({lon_name: slice(lon_min, lon_max)})
        else:
            lon_subset = xr.concat(
                [
                    ds.sel({lon_name: slice(lon_min, lon_vals.max())}),
                    ds.sel({lon_name: slice(lon_vals.min(), lon_max)}),
                ],
                dim=lon_name,
            )

        lat_vals = lon_subset[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        subset = lon_subset.sel({lat_name: lat_slice})
        if "time" in subset.coords:
            subset = subset.sel(time=slice(self.start_date, self.end_date))

        return subset

    @staticmethod
    def _get_coord_name(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
        for name in candidates:
            if name in ds.coords:
                return name
        return None

    def _download_jpl_from_cmr(
        self,
        output_dir: Path,
        earthdata_auth: Optional[Tuple[str, str]],
        force_download: bool,
    ) -> Optional[Path]:
        collection_id = self.config.get(
            'GRACE_JPL_COLLECTION_ID',
            'C3195527175-POCLOUD',
        )

        self.logger.info(
            "Querying CMR for JPL GRACE mascon (collection %s)",
            collection_id,
        )

        session = requests.Session()
        if earthdata_auth:
            session.auth = earthdata_auth

        try:
            resp = session.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params={
                    "collection_concept_id": collection_id,
                    "page_size": 1,
                    "sort_key": "-start_date",
                },
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:
            self.logger.warning(f"CMR query failed: {exc}")
            return None

        entries = resp.json().get("feed", {}).get("entry", [])
        if not entries:
            self.logger.warning("CMR query returned no granules for JPL collection.")
            return None

        links = entries[0].get("links", [])
        data_links = [
            link.get("href")
            for link in links
            if link.get("rel", "").endswith("data#")
            and link.get("href", "").endswith(".nc")
        ]
        if not data_links:
            self.logger.warning("CMR granule did not include a NetCDF data link.")
            return None

        download_url = next(
            (href for href in data_links if "archive.podaac.earthdata.nasa.gov" in href),
            data_links[0],
        )
        target_file = output_dir / Path(download_url).name
        if target_file.exists() and not force_download:
            return target_file

        try:
            with session.get(download_url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(target_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return target_file
        except Exception as exc:
            self.logger.warning(f"CMR download failed: {exc}")
            return None
