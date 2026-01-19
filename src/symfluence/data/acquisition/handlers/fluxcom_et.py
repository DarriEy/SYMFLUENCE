"""FLUXCOM Evapotranspiration Acquisition Handler

Provides acquisition for FLUXCOM evapotranspiration data.

FLUXCOM Overview:
    Data Type: Machine-learning upscaled eddy covariance ET
    Resolution: Variable (0.5° typical)
    Coverage: Global
    Source: Max Planck Institute for Biogeochemistry

Note:
    FLUXCOM does not have a public API. This handler supports:
    1. Direct download from a user-provided URL (e.g., internal server, S3 presigned)
    2. Local file discovery (if data is manually placed)
"""

from pathlib import Path
import requests

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


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
        local_pattern = self.config_dict.get('FLUXCOM_FILE_PATTERN', "*.nc")
        existing_files = list(et_dir.glob(local_pattern))
        if existing_files and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Found existing FLUXCOM files in {et_dir}")
            return et_dir

        # Option 2: Download from URL
        download_url = self.config_dict.get('FLUXCOM_DOWNLOAD_URL')
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
