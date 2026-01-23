"""
MSWEP Precipitation Data Acquisition Handler

Provides acquisition for Multi-Source Weighted-Ensemble Precipitation (MSWEP)
data. MSWEP is a global precipitation product that merges gauge, satellite,
and reanalysis data with optimal weighting.

MSWEP v2.8 features:
- 0.1° spatial resolution (approximately 10 km)
- 3-hourly temporal resolution (also daily/monthly available)
- Global land+ocean coverage
- Long record: 1979-present
- Real-time (NRT) updates

Data access requires registration at:
http://www.gloh2o.org/mswep/
"""
import os
import ftplib  # nosec B402 - FTP required for MSWEP data server access
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


@AcquisitionRegistry.register('MSWEP')
class MSWEPAcquirer(BaseAcquisitionHandler):
    """
    Handles MSWEP precipitation data acquisition.

    MSWEP data is distributed via FTP/HTTP from the GloH2O repository.
    Requires user registration and credentials for access.

    Configuration:
        MSWEP_USERNAME: GloH2O username (or env var MSWEP_USERNAME)
        MSWEP_PASSWORD: GloH2O password (or env var MSWEP_PASSWORD)
        MSWEP_VERSION: Version to download ('v2.8', 'nrt') (default: v2.8)
        MSWEP_RESOLUTION: Temporal resolution ('3hourly', 'daily', 'monthly')
        MSWEP_PRODUCT: Product type ('Past', 'NRT') (default: Past)
    """

    FTP_HOST = "data.gloh2o.org"
    BASE_PATH = "/MSWEP_V280"

    def download(self, output_dir: Path) -> Path:
        """
        Download MSWEP precipitation data.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data directory
        """
        self.logger.info("Starting MSWEP precipitation data acquisition")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get credentials
        username, password = self._get_credentials()
        if not username or not password:
            raise ValueError(
                "MSWEP credentials required. Set MSWEP_USERNAME and MSWEP_PASSWORD "
                "environment variables or config settings. Register at http://www.gloh2o.org/mswep/"
            )

        # Get configuration
        resolution = self.config_dict.get('MSWEP_RESOLUTION', 'daily')
        product = self.config_dict.get('MSWEP_PRODUCT', 'Past')
        force_download = self.config_dict.get('FORCE_DOWNLOAD', False)

        # Build file list
        file_list = self._generate_file_list(resolution, product)

        self.logger.info(
            f"Downloading MSWEP {resolution} data: {len(file_list)} files, "
            f"{self.start_date.date()} to {self.end_date.date()}"
        )

        # Download via FTP
        downloaded = self._download_ftp(
            output_dir, file_list, username, password, force_download
        )

        if not downloaded:
            # Try HTTP fallback
            self.logger.warning("FTP download failed, trying HTTP...")
            downloaded = self._download_http(
                output_dir, file_list, username, password, force_download
            )

        if not downloaded:
            raise RuntimeError("Failed to download any MSWEP data")

        self.logger.info(f"MSWEP download complete: {len(downloaded)} files")
        return output_dir

    def _get_credentials(self):
        """Get MSWEP credentials from environment or config."""
        username = (
            os.environ.get('MSWEP_USERNAME') or
            self.config_dict.get('MSWEP_USERNAME')
        )
        password = (
            os.environ.get('MSWEP_PASSWORD') or
            self.config_dict.get('MSWEP_PASSWORD')
        )
        return username, password

    def _generate_file_list(self, resolution: str, product: str) -> List[dict]:
        """Generate list of files to download based on date range."""
        files = []

        if resolution == '3hourly':
            # Files organized by year/day-of-year
            current = self.start_date
            while current <= self.end_date:
                year = current.year
                doy = current.timetuple().tm_yday

                for hour in range(0, 24, 3):
                    files.append({
                        'remote_path': f"{self.BASE_PATH}/{product}/3hourly/{year}/{doy:03d}{hour:02d}.nc",
                        'local_name': f"mswep_{year}{doy:03d}{hour:02d}.nc",
                        'date': current,
                    })

                current += pd.Timedelta(days=1)

        elif resolution == 'daily':
            # Daily files by year/day-of-year
            current = self.start_date
            while current <= self.end_date:
                year = current.year
                doy = current.timetuple().tm_yday

                files.append({
                    'remote_path': f"{self.BASE_PATH}/{product}/Daily/{year}/{doy:03d}.nc",
                    'local_name': f"mswep_{year}{doy:03d}.nc",
                    'date': current,
                })

                current += pd.Timedelta(days=1)

        elif resolution == 'monthly':
            # Monthly files by year
            current = datetime(self.start_date.year, self.start_date.month, 1)
            while current <= self.end_date:
                year = current.year
                month = current.month

                files.append({
                    'remote_path': f"{self.BASE_PATH}/{product}/Monthly/{year}{month:02d}.nc",
                    'local_name': f"mswep_{year}{month:02d}.nc",
                    'date': current,
                })

                # Next month
                if month == 12:
                    current = datetime(year + 1, 1, 1)
                else:
                    current = datetime(year, month + 1, 1)

        return files

    def _download_ftp(
        self,
        output_dir: Path,
        file_list: List[dict],
        username: str,
        password: str,
        force: bool
    ) -> List[Path]:
        """Download files via FTP."""
        downloaded = []

        try:
            ftp = ftplib.FTP(self.FTP_HOST)  # nosec B321 - FTP required for MSWEP data server
            ftp.login(username, password)
            self.logger.info("Connected to MSWEP FTP server")

            for file_info in file_list:
                local_path = output_dir / file_info['local_name']

                if local_path.exists() and not force:
                    downloaded.append(local_path)
                    continue

                try:
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {file_info['remote_path']}", f.write)
                    downloaded.append(local_path)
                    self.logger.debug(f"Downloaded: {file_info['local_name']}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {file_info['remote_path']}: {e}")

            ftp.quit()

        except Exception as e:
            self.logger.error(f"FTP connection failed: {e}")

        return downloaded

    def _download_http(
        self,
        output_dir: Path,
        file_list: List[dict],
        username: str,
        password: str,
        force: bool
    ) -> List[Path]:
        """Download files via HTTP (fallback)."""
        import requests
        from requests.auth import HTTPBasicAuth

        downloaded = []
        base_url = f"https://{self.FTP_HOST}"

        session = requests.Session()
        session.auth = HTTPBasicAuth(username, password)

        for file_info in file_list:
            local_path = output_dir / file_info['local_name']

            if local_path.exists() and not force:
                downloaded.append(local_path)
                continue

            url = f"{base_url}{file_info['remote_path']}"

            try:
                response = session.get(url, stream=True, timeout=60)
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded.append(local_path)
                self.logger.debug(f"Downloaded: {file_info['local_name']}")

            except Exception as e:
                self.logger.warning(f"Failed to download {url}: {e}")

        return downloaded
