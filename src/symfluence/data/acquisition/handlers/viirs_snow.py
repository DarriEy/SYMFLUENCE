"""
VIIRS Snow Cover Data Acquisition Handler

Provides acquisition for VIIRS (Visible Infrared Imaging Radiometer Suite)
snow cover products. VIIRS is the successor to MODIS, providing improved
snow detection capabilities.

VIIRS Snow features:
- 375m (I-band) to 750m (M-band) spatial resolution
- Daily temporal resolution
- Improved cloud masking compared to MODIS
- Available from Suomi NPP (2012) and NOAA-20 (2018)
- Products: VNP10A1 (daily), VNP10A2F (8-day composite)

Data access via NASA AppEEARS or NSIDC DAAC.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@AcquisitionRegistry.register('VIIRS_SNOW')
@AcquisitionRegistry.register('VNP10')
class VIIRSSnowAcquirer(BaseAcquisitionHandler):
    """
    Handles VIIRS snow cover data acquisition via NASA AppEEARS.

    Downloads daily or 8-day composite snow cover data for specified
    spatial extent and time period.

    Configuration:
        VIIRS_SNOW_PRODUCT: Product ID ('VNP10A1F', 'VNP10A2F')
        VIIRS_SNOW_LAYERS: Layers to download (default: SCA, QA)
        VIIRS_SNOW_PLATFORM: 'NPP' or 'NOAA20' (default: NPP)
    """

    PRODUCTS = {
        'VNP10A1F': 'VNP10A1F.002',   # Daily fractional snow cover (NPP)
        'VNP10A2F': 'VNP10A2F.002',   # 8-day fractional snow cover (NPP)
        'VJ110A1F': 'VJ110A1F.002',   # Daily (NOAA-20)
        'VJ110A2F': 'VJ110A2F.002',   # 8-day (NOAA-20)
    }

    DEFAULT_LAYERS = [
        'CGF_NDSI_Snow_Cover',  # NDSI-based snow cover (0-100)
        'Snow_Albedo_Daily_Tile',
        'Basic_QA',
    ]

    def download(self, output_dir: Path) -> Path:
        """
        Download VIIRS snow cover data via AppEEARS.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting VIIRS snow cover data acquisition via AppEEARS")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data
        force_download = self.config_dict.get('FORCE_DOWNLOAD', False)
        existing_files = list(output_dir.glob("*VNP10*.nc")) + list(output_dir.glob("*Snow*.nc"))
        if existing_files and not force_download:
            self.logger.info(f"VIIRS snow data already exists: {len(existing_files)} files")
            return output_dir

        # Get credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise ValueError(
                "NASA Earthdata credentials required. Set via environment variables "
                "(EARTHDATA_USERNAME, EARTHDATA_PASSWORD) or ~/.netrc"
            )

        # Get configuration
        product = self.config_dict.get('VIIRS_SNOW_PRODUCT', 'VNP10A1F')
        product_id = self.PRODUCTS.get(product, f"{product}.002")
        layers = self._get_layers()

        # Authenticate
        token = self._get_appeears_token(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        # Submit task
        task_id = self._submit_task(token, product_id, layers)
        if not task_id:
            raise RuntimeError("Failed to submit AppEEARS task")

        # Wait for completion and download
        self._wait_and_download(token, task_id, output_dir)

        self.logger.info(f"VIIRS snow download complete: {output_dir}")
        return output_dir

    def _get_layers(self) -> List[str]:
        """Get layers to download."""
        config_layers = self.config_dict.get('VIIRS_SNOW_LAYERS')
        if config_layers:
            if isinstance(config_layers, str):
                return [config_layers]
            return list(config_layers)

        return ['CGF_NDSI_Snow_Cover', 'Basic_QA']

    def _get_appeears_token(self, username: str, password: str) -> Optional[str]:
        """Authenticate with AppEEARS and get token."""
        try:
            response = requests.post(
                f"{APPEEARS_BASE}/login",
                auth=(username, password),
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('token')
        except Exception as e:
            self.logger.error(f"AppEEARS authentication failed: {e}")
            return None

    def _submit_task(
        self,
        token: str,
        product_id: str,
        layers: List[str]
    ) -> Optional[str]:
        """Submit AppEEARS task request."""
        headers = {'Authorization': f'Bearer {token}'}

        layer_specs = [{'product': product_id, 'layer': layer} for layer in layers]

        task_name = f"VIIRS_Snow_{self.domain_name}_{self.start_date.strftime('%Y%m%d')}"

        task_request = {
            'task_type': 'area',
            'task_name': task_name,
            'params': {
                'dates': [{
                    'startDate': self.start_date.strftime('%m-%d-%Y'),
                    'endDate': self.end_date.strftime('%m-%d-%Y'),
                }],
                'layers': layer_specs,
                'output': {
                    'format': {'type': 'netcdf4'},
                    'projection': 'geographic',
                },
                'geo': self._build_geo_spec(),
            }
        }

        try:
            response = requests.post(
                f"{APPEEARS_BASE}/task",
                headers=headers,
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            task_id = result.get('task_id')
            self.logger.info(f"AppEEARS task submitted: {task_id}")
            return task_id
        except Exception as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            return None

    def _build_geo_spec(self) -> Dict[str, Any]:
        """Build geographic specification for task."""
        if self.bbox:
            return {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [self.bbox['lon_min'], self.bbox['lat_min']],
                            [self.bbox['lon_max'], self.bbox['lat_min']],
                            [self.bbox['lon_max'], self.bbox['lat_max']],
                            [self.bbox['lon_min'], self.bbox['lat_max']],
                            [self.bbox['lon_min'], self.bbox['lat_min']],
                        ]]
                    },
                    'properties': {}
                }]
            }
        else:
            raise ValueError("Bounding box required for VIIRS snow acquisition")

    def _wait_and_download(
        self,
        token: str,
        task_id: str,
        output_dir: Path,
        max_wait: int = 7200,
        poll_interval: int = 30
    ):
        """Wait for task completion and download results."""
        headers = {'Authorization': f'Bearer {token}'}
        elapsed = 0

        while elapsed < max_wait:
            try:
                response = requests.get(
                    f"{APPEEARS_BASE}/task/{task_id}",
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                status = response.json()

                task_status = status.get('status')
                self.logger.info(f"AppEEARS task status: {task_status}")

                if task_status == 'done':
                    self._download_results(token, task_id, output_dir)
                    return
                elif task_status in ['error', 'expired']:
                    raise RuntimeError(f"AppEEARS task failed: {status.get('error', 'Unknown error')}")

                time.sleep(poll_interval)
                elapsed += poll_interval

            except requests.RequestException as e:
                self.logger.warning(f"Error checking task status: {e}")
                time.sleep(poll_interval)
                elapsed += poll_interval

        raise RuntimeError(f"AppEEARS task timed out after {max_wait} seconds")

    def _download_results(self, token: str, task_id: str, output_dir: Path):
        """Download completed task results."""
        headers = {'Authorization': f'Bearer {token}'}

        try:
            response = requests.get(
                f"{APPEEARS_BASE}/bundle/{task_id}",
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            bundle = response.json()
        except Exception as e:
            self.logger.error(f"Failed to get bundle info: {e}")
            return

        files = bundle.get('files', [])
        self.logger.info(f"Downloading {len(files)} files from AppEEARS")

        for file_info in files:
            file_id = file_info.get('file_id')
            file_name = file_info.get('file_name')

            if not file_name.endswith(('.nc', '.nc4', '.tif')):
                continue

            output_path = output_dir / file_name

            try:
                response = requests.get(
                    f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                    headers=headers,
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.debug(f"Downloaded: {file_name}")

            except Exception as e:
                self.logger.warning(f"Failed to download {file_name}: {e}")
