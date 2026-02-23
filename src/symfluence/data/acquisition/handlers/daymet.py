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
from typing import List, Optional, Tuple
from urllib.parse import urlparse

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
    GRIDDED_URL = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129"

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

        # Default: core hydrological variables
        return ['tmax', 'tmin', 'prcp', 'swe']

    def _is_point_request(self) -> bool:
        """Check if this is a small enough area for single-pixel API."""
        if not self.bbox:
            return True

        lat_range = self.bbox['lat_max'] - self.bbox['lat_min']
        lon_range = self.bbox['lon_max'] - self.bbox['lon_min']

        # Use single-pixel for areas smaller than ~0.1 degrees
        return lat_range < 0.1 and lon_range < 0.1

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

        ORNL DAAC THREDDS redirects to ``urs.earthdata.nasa.gov`` for OAuth.
        A plain ``session.auth`` sends credentials to every host in the
        redirect chain (including the OAuth callback on opendap.earthdata),
        which breaks the flow.  :class:`_EarthdataSession` overrides
        ``rebuild_auth`` so credentials are only sent to URS itself.

        Returns:
            A :class:`requests.Session` (authenticated when credentials exist).
        """
        username, password = self._get_earthdata_credentials()
        if username and password:
            return _EarthdataSession(username, password)

        self.logger.warning(
            "No NASA Earthdata credentials found. ORNL DAAC requires "
            "authentication for gridded Daymet downloads.\n"
            "To set up credentials, create a ~/.netrc file with:\n"
            "\n"
            "  machine urs.earthdata.nasa.gov\n"
            "  login <your_earthdata_username>\n"
            "  password <your_earthdata_password>\n"
            "\n"
            "Register for a free account at https://urs.earthdata.nasa.gov/users/new\n"
            "Alternatively set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars."
        )
        return requests.Session()

    def _download_gridded(self, output_file: Path, variables: List[str]):
        """Download gridded data using THREDDS subset service."""
        if not self.bbox:
            self.logger.error("Bounding box required for Daymet download")
            return

        session = self._get_earthdata_session()

        # Download each year separately (THREDDS limitation)
        for year in range(self.start_date.year, self.end_date.year + 1):
            year_file = output_file.parent / f"daymet_{year}.nc"

            if year_file.exists():
                continue

            for var in variables:
                var_file = output_file.parent / f"daymet_{var}_{year}.nc"

                if var_file.exists():
                    continue

                # Build THREDDS NCSS URL
                url = f"{self.GRIDDED_URL}/daymet_v4_daily_na_{var}_{year}.nc"

                params = {
                    'var': var,
                    'north': self.bbox['lat_max'],
                    'south': self.bbox['lat_min'],
                    'east': self.bbox['lon_max'],
                    'west': self.bbox['lon_min'],
                    'time_start': f"{year}-01-01T00:00:00Z",
                    'time_end': f"{year}-12-31T23:59:59Z",
                    'accept': 'netcdf4',
                }

                try:
                    self.logger.info(f"Downloading Daymet {var} for {year}")

                    response = session.get(
                        url,
                        params=params,
                        stream=True,
                        timeout=600
                    )
                    response.raise_for_status()

                    with open(var_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Failed to download Daymet {var} {year}: {e}")
