"""
Daymet Climate Data Acquisition Handler

Provides acquisition for Daymet daily surface weather data for North America.
Daymet provides gridded estimates of daily weather parameters interpolated
from ground-based observations.

Daymet features:
- 1 km spatial resolution
- Daily temporal resolution
- Coverage: North America (US, Canada, Mexico)
- Period: 1980-present
- Variables: Tmin, Tmax, Precip, SWE, VP, SRAD, Dayl

Data access via ORNL DAAC:
https://daymet.ornl.gov/
"""
import netrc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


class _EarthdataSession(requests.Session):
    """Session that re-applies credentials during NASA URS OAuth redirect flow.

    ORNL DAAC THREDDS redirects to urs.earthdata.nasa.gov for authentication,
    then back to opendap.earthdata.nasa.gov with an OAuth code. The standard
    requests library strips Authorization headers on cross-host redirects, so
    credentials never reach URS. This subclass overrides ``rebuild_auth`` to
    re-apply credentials only when the redirect target is URS.
    """

    def __init__(self, username: str, password: str):
        super().__init__()
        self._earthdata_auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        if urlparse(prepared_request.url).hostname == 'urs.earthdata.nasa.gov':
            prepared_request.prepare_auth(self._earthdata_auth)


class _EarthdataTokenSession(requests.Session):
    """Session that uses a NASA Earthdata Bearer token for authentication.

    NASA Earthdata supports token-based access as an alternative to the
    username/password OAuth redirect flow.  The token is sent as a Bearer
    ``Authorization`` header and re-applied on redirects to URS so it
    survives the same cross-host redirect that strips default headers.
    """

    def __init__(self, token: str):
        super().__init__()
        self.headers.update({'Authorization': f'Bearer {token}'})
        self._token = token

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        host = urlparse(prepared_request.url).hostname or ''
        if host.endswith('.earthdata.nasa.gov') or host.endswith('.ornl.gov'):
            prepared_request.headers['Authorization'] = f'Bearer {self._token}'


@AcquisitionRegistry.register('DAYMET')
class DaymetAcquirer(BaseAcquisitionHandler):
    """
    Handles Daymet climate data acquisition from ORNL DAAC.

    Downloads daily gridded climate data for specified region
    and time period.

    Configuration:
        DAYMET_VARIABLES: Variables to download (default: all)
        DAYMET_FORMAT: Output format ('netcdf', 'csv') (default: netcdf)
    """

    # ORNL DAAC Daymet endpoints
    SINGLE_PIXEL_URL = "https://daymet.ornl.gov/single-pixel/api/data"
    GRIDDED_URL = "https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data"

    # Cloud OPeNDAP endpoint (NASA Hyrax) — supports server-side subsetting
    OPENDAP_URL = (
        "https://opendap.earthdata.nasa.gov/collections/"
        "C2532426483-ORNL_CLOUD/granules/"
        "Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc"
    )

    # Daymet Lambert Conformal Conic projection (native grid CRS)
    DAYMET_CRS = (
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
        "+lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )

    AVAILABLE_VARIABLES = [
        'tmax',  # Maximum temperature (°C)
        'tmin',  # Minimum temperature (°C)
        'prcp',  # Precipitation (mm)
        'swe',   # Snow water equivalent (mm)
        'vp',    # Vapor pressure (Pa)
        'srad',  # Shortwave radiation (W/m²)
        'dayl',  # Day length (seconds)
    ]

    def download(self, output_dir: Path) -> Path:
        """
        Download Daymet climate data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting Daymet climate data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve config values that may be missing when the typed config
        # could not be built (e.g. standalone ``data download`` CLI).
        self._resolve_flat_config()

        # Get configuration
        variables = self._get_variables()
        force_download = self._get_config_value(lambda: self.config.data.force_download, default=False)

        # Check if bbox is in North America
        if self.bbox:
            if (self.bbox['lat_min'] < 14 or self.bbox['lat_max'] > 83 or
                self.bbox['lon_min'] < -179 or self.bbox['lon_max'] > -52):
                self.logger.warning(
                    "Daymet coverage is limited to North America. "
                    "Data may not be available for this region."
                )

        # Build output filename
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        var_str = '_'.join(variables[:3])  # First 3 vars in filename
        output_file = output_dir / f"daymet_{var_str}_{start_str}_{end_str}.nc"

        if output_file.exists() and not force_download:
            self.logger.info(f"Daymet file already exists: {output_file}")
            return output_file

        # Download data
        if self._is_point_request():
            # Use single-pixel API for small areas
            self._download_single_pixel(output_dir, variables)
        else:
            # Use subset service for larger areas
            self._download_gridded(output_file, variables)

        return output_dir if not output_file.exists() else output_file

    def _get_variables(self) -> List[str]:
        """Get variables to download."""
        config_vars = self._get_config_value(lambda: None, default=None, dict_key='DAYMET_VARIABLES')

        if config_vars:
            if isinstance(config_vars, str):
                return [config_vars]
            return list(config_vars)

        # Default: core hydrological variables (temp + precip)
        return ['tmax', 'tmin', 'prcp']

    def _is_point_request(self) -> bool:
        """Check if this is a small enough area for single-pixel API."""
        if not self.bbox:
            return True

        lat_range = self.bbox['lat_max'] - self.bbox['lat_min']
        lon_range = self.bbox['lon_max'] - self.bbox['lon_min']

        # Use single-pixel for areas smaller than ~0.1 degrees
        return lat_range < 0.1 and lon_range < 0.1

    def _resolve_flat_config(self):
        """Populate bbox / dates from a flat config dict if the typed config
        coercion failed (e.g. standalone ``symfluence data download`` CLI).
        """
        cfg = self._config
        if not isinstance(cfg, dict):
            return

        if not self.bbox:
            raw = cfg.get('BOUNDING_BOX_COORDS')
            if raw:
                self.bbox = self._parse_bbox(raw)

        if self.start_date is None or pd.isna(self.start_date):
            raw = cfg.get('EXPERIMENT_TIME_START')
            if raw:
                self.start_date = pd.to_datetime(raw)

        if self.end_date is None or pd.isna(self.end_date):
            raw = cfg.get('EXPERIMENT_TIME_END')
            if raw:
                self.end_date = pd.to_datetime(raw)

    def _download_single_pixel(self, output_dir: Path, variables: List[str]):
        """Download using single-pixel API (for point locations)."""
        if not self.bbox:
            self.logger.error("Bounding box required for Daymet download")
            return

        # Use centroid for single-pixel request
        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2

        params = {
            'lat': lat,
            'lon': lon,
            'vars': ','.join(variables),
            'start': self.start_date.strftime('%Y-%m-%d'),
            'end': self.end_date.strftime('%Y-%m-%d'),
            'format': 'csv',
        }

        try:
            self.logger.info(f"Downloading Daymet single-pixel data at ({lat:.4f}, {lon:.4f})")

            response = requests.get(
                self.SINGLE_PIXEL_URL,
                params=params,
                timeout=300
            )
            response.raise_for_status()

            # Save CSV
            output_file = output_dir / f"daymet_{lat:.4f}_{lon:.4f}.csv"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            self.logger.info(f"Downloaded Daymet data: {output_file}")

        except Exception as e:  # noqa: BLE001 — data acquisition resilience
            self.logger.error(f"Daymet single-pixel download failed: {e}")

    def _get_earthdata_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Get NASA Earthdata credentials from .netrc, env vars, or config.

        Returns:
            (username, password) tuple, or (None, None) if not found.
        """
        # 1. Try .netrc (preferred — most secure)
        try:
            netrc_path = Path.home() / '.netrc'
            if netrc_path.exists():
                nrc = netrc.netrc(str(netrc_path))
                for host in ('urs.earthdata.nasa.gov', 'earthdata.nasa.gov'):
                    auth = nrc.authenticators(host)
                    if auth:
                        self.logger.debug(f"Using Earthdata credentials from ~/.netrc ({host})")
                        return auth[0], auth[2]
        except (OSError, netrc.NetrcParseError) as e:
            self.logger.debug(f"Could not read .netrc: {e}")

        # 2. Try environment variables
        username = os.environ.get('EARTHDATA_USERNAME')
        password = os.environ.get('EARTHDATA_PASSWORD')
        if username and password:
            self.logger.debug("Using Earthdata credentials from environment variables")
            return username, password

        # 3. Try config
        username = self._get_config_value(lambda: None, default=None, dict_key='EARTHDATA_USERNAME')
        password = self._get_config_value(lambda: None, default=None, dict_key='EARTHDATA_PASSWORD')
        if username and password:
            self.logger.debug("Using Earthdata credentials from config")
            return username, password

        return None, None

    def _get_earthdata_session(self) -> requests.Session:
        """Build an authenticated requests session for ORNL DAAC / Earthdata.

        Tries authentication methods in order of preference:

        1. **Bearer token** (``EARTHDATA_TOKEN`` env var or config key) —
           most reliable, avoids the URS OAuth redirect issues that can
           occur with username/password credentials.
        2. **Username/password** from ``~/.netrc``, env vars, or config —
           uses the ``_EarthdataSession`` redirect-aware session.

        Returns:
            A :class:`requests.Session` (authenticated when credentials exist).
        """
        # 1. Try Bearer token first (most reliable)
        token = self._get_earthdata_token()
        if token:
            self.logger.debug("Using Earthdata Bearer token for ORNL DAAC session")
            return _EarthdataTokenSession(token)

        # 2. Fall back to username/password
        username, password = self._get_earthdata_credentials()
        if username and password:
            return _EarthdataSession(username, password)

        self.logger.warning(
            "No NASA Earthdata credentials found. ORNL DAAC requires "
            "authentication for gridded Daymet downloads.\n"
            "To set up authentication, either:\n"
            "\n"
            "  1. Set a Bearer token (recommended):\n"
            "     export EARTHDATA_TOKEN=<your_token>\n"
            "     Generate at: https://urs.earthdata.nasa.gov → My Profile → Generate Token\n"
            "\n"
            "  2. Or create a ~/.netrc file with:\n"
            "     machine urs.earthdata.nasa.gov\n"
            "     login <your_earthdata_username>\n"
            "     password <your_earthdata_password>\n"
            "\n"
            "Register for a free account at https://urs.earthdata.nasa.gov/users/new"
        )
        return requests.Session()

    def _bbox_to_lcc(self) -> Dict[str, float]:
        """Convert lat/lon bounding box to Daymet LCC x/y coordinates.

        Transforms all four corners of the geographic bounding box to
        Lambert Conformal Conic projection and returns the enclosing
        rectangle in LCC space, with a 1 km buffer (one grid cell).
        """
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326", self.DAYMET_CRS, always_xy=True
        )

        corners_lon = [
            self.bbox['lon_min'], self.bbox['lon_max'],
            self.bbox['lon_min'], self.bbox['lon_max'],
        ]
        corners_lat = [
            self.bbox['lat_min'], self.bbox['lat_min'],
            self.bbox['lat_max'], self.bbox['lat_max'],
        ]

        x_coords, y_coords = transformer.transform(corners_lon, corners_lat)

        buffer = 1000.0  # 1 km = 1 Daymet grid cell
        return {
            'x_min': min(x_coords) - buffer,
            'x_max': max(x_coords) + buffer,
            'y_min': min(y_coords) - buffer,
            'y_max': max(y_coords) + buffer,
        }

    @staticmethod
    def _ensure_dodsrc():
        """Ensure ``~/.dodsrc`` exists so the netCDF4 C library can
        authenticate with NASA Earthdata via OPeNDAP.

        The C library does not read ``~/.netrc`` directly — it needs
        ``.dodsrc`` to point it to the cookie jar and netrc file.
        """
        dodsrc = Path.home() / '.dodsrc'
        if dodsrc.exists():
            return

        netrc_path = Path.home() / '.netrc'
        cookie_path = Path.home() / '.urs_cookies'
        if not netrc_path.exists():
            return  # nothing to point to

        dodsrc.write_text(
            f"HTTP.COOKIEJAR={cookie_path}\n"
            f"HTTP.NETRC={netrc_path}\n"
        )
        # Cookie jar must exist (even if empty) for the C library
        cookie_path.touch(exist_ok=True)

    def _suppress_stderr(self):
        """Context-manager-like helper that silences the netCDF4 C library's
        stderr output (HTML error pages, auth noise).

        Returns *(saved_fd, devnull_fd)* — caller must restore with
        ``_restore_stderr``.
        """
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        return saved, devnull

    @staticmethod
    def _restore_stderr(saved: int, devnull: int):
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

    def _download_opendap_subset(
        self, var: str, year: int, var_file: Path, lcc_bbox: Dict[str, float],
        max_retries: int = 3,
    ) -> bool:
        """Download a Daymet variable via OPeNDAP server-side subsetting.

        Uses a two-step approach to avoid the DAP2 "variable too large"
        error that occurs when opening the full continental dataset:

        1. Fetch only the 1-D coordinate arrays (``x``, ``y``, ``time``)
           via a DAP2 constraint expression — these are small (~100 KB).
        2. Compute array index ranges for the LCC bounding box, then
           request only that slice with a second constrained URL.

        Retries up to *max_retries* times with backoff for transient errors.
        Returns True on success, False on failure.
        """
        import time as _time

        import numpy as np
        import xarray as xr

        self._ensure_dodsrc()

        url = self.OPENDAP_URL.format(var=var, year=year)
        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    "OPeNDAP subsetting for Daymet %s %d (attempt %d/%d)",
                    var, year, attempt, max_retries,
                )

                # --- Step 1: fetch coordinate arrays only ---------------
                coord_url = f"{url}?x,y,time"
                s, d = self._suppress_stderr()
                try:
                    ds_coords = xr.open_dataset(coord_url, engine='netcdf4')
                finally:
                    self._restore_stderr(s, d)

                x_vals = ds_coords['x'].values
                y_vals = ds_coords['y'].values
                n_time = ds_coords.sizes['time']
                ds_coords.close()

                # --- Step 2: compute index ranges -----------------------
                x_idx = np.where(
                    (x_vals >= lcc_bbox['x_min']) & (x_vals <= lcc_bbox['x_max'])
                )[0]
                y_idx = np.where(
                    (y_vals >= lcc_bbox['y_min']) & (y_vals <= lcc_bbox['y_max'])
                )[0]

                if len(x_idx) == 0 or len(y_idx) == 0:
                    self.logger.warning(
                        "OPeNDAP subset returned empty data for %s %d",
                        var, year,
                    )
                    return False

                xi0, xi1 = int(x_idx[0]), int(x_idx[-1])
                yi0, yi1 = int(y_idx[0]), int(y_idx[-1])

                # --- Step 3: download subset as NetCDF4 via session -----
                # The netCDF4 C library can't reuse auth cookies for
                # constrained URLs, so download via the requests session
                # using the .nc4 extension (returns native NetCDF4).
                constraint = (
                    f"{var}[0:1:{n_time - 1}][{yi0}:1:{yi1}][{xi0}:1:{xi1}],"
                    f"x[{xi0}:1:{xi1}],"
                    f"y[{yi0}:1:{yi1}],"
                    f"time"
                )
                nc4_url = f"{url}.nc4?{constraint}"

                session = self._get_earthdata_session()
                response = session.get(nc4_url, timeout=120)
                response.raise_for_status()

                tmp_file = var_file.with_suffix('.tmp.nc')
                tmp_file.write_bytes(response.content)
                ds_sub = xr.open_dataset(tmp_file)
                ds_sub.load()
                for v in ds_sub.data_vars:
                    ds_sub[v].encoding.clear()
                for c in ds_sub.coords:
                    ds_sub[c].encoding.clear()
                ds_sub.to_netcdf(var_file, engine='h5netcdf')
                ds_sub.close()
                tmp_file.unlink(missing_ok=True)

                self.logger.info(
                    "Downloaded subsetted Daymet %s %d via OPeNDAP: %s",
                    var, year, var_file,
                )
                return True

            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries:
                    wait = 5 * attempt
                    self.logger.info(
                        "OPeNDAP attempt %d failed for %s %d: %s  "
                        "— retrying in %ds",
                        attempt, var, year, e, wait,
                    )
                    _time.sleep(wait)

        self.logger.error(
            "OPeNDAP subsetting failed for Daymet %s %d after %d attempts: %s",
            var, year, max_retries, last_err,
        )
        return False

    def _download_gridded(self, output_file: Path, variables: List[str]):
        """Download gridded Daymet data with server-side spatial subsetting.

        Converts the geographic bounding box to Daymet's native Lambert
        Conformal Conic projection and downloads each variable/year via
        OPeNDAP, which transfers only the subsetted region.
        """
        if not self.bbox:
            self.logger.error("Bounding box required for Daymet download")
            return

        lcc_bbox = self._bbox_to_lcc()
        self.logger.info(
            f"Daymet bbox in LCC: x=[{lcc_bbox['x_min']:.0f}, "
            f"{lcc_bbox['x_max']:.0f}], "
            f"y=[{lcc_bbox['y_min']:.0f}, {lcc_bbox['y_max']:.0f}]"
        )

        for year in range(self.start_date.year, self.end_date.year + 1):
            year_file = output_file.parent / f"daymet_{year}.nc"

            if year_file.exists():
                continue

            failed = []
            for var in variables:
                var_file = output_file.parent / f"daymet_{var}_{year}.nc"

                if var_file.exists():
                    continue

                if not self._download_opendap_subset(
                    var, year, var_file, lcc_bbox
                ):
                    failed.append(var)

            if failed:
                self.logger.error(
                    "Failed to download Daymet variables %s for %d. "
                    "Check that ~/.netrc has credentials for "
                    "urs.earthdata.nasa.gov and that ~/.dodsrc exists.",
                    failed, year,
                )

            # Merge per-variable files into a single year file
            self._merge_year_files(output_file.parent, variables, year, year_file)

    def _merge_year_files(
        self, output_dir: Path, variables: List[str], year: int, year_file: Path
    ):
        """Merge per-variable Daymet files into a single per-year file.

        Collects ``daymet_{var}_{year}.nc`` files, merges them with
        ``xr.merge``, writes to ``daymet_{year}.nc``, and removes the
        per-variable files.  Skips silently if no per-variable files exist
        (e.g. all downloads failed).
        """
        import xarray as xr

        var_files = [
            output_dir / f"daymet_{var}_{year}.nc"
            for var in variables
            if (output_dir / f"daymet_{var}_{year}.nc").exists()
        ]

        if not var_files:
            self.logger.warning(
                f"No per-variable files found for Daymet {year}; "
                "cannot create merged file"
            )
            return

        try:
            datasets = [xr.open_dataset(f) for f in var_files]
            merged = xr.merge(datasets, compat='override')
            merged.to_netcdf(year_file, engine='h5netcdf')

            for ds in datasets:
                ds.close()

            # Remove per-variable files now that the merged file exists
            for f in var_files:
                f.unlink()

            self.logger.info(
                f"Merged {len(var_files)} variable files into {year_file.name}"
            )

        except Exception as e:  # noqa: BLE001 — merge is best-effort
            self.logger.warning(
                f"Failed to merge per-variable files for {year}: {e}. "
                "Per-variable files retained."
            )
