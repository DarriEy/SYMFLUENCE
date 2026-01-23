"""
MODIS LAI/FPAR Data Acquisition Handler

Provides acquisition for MODIS MCD15A2H (combined Terra+Aqua) and
MOD15A2H/MYD15A2H Leaf Area Index (LAI) and Fraction of Photosynthetically
Active Radiation (FPAR) products via NASA AppEEARS API.

MODIS LAI features:
- 500m spatial resolution
- 8-day composite temporal resolution
- LAI and FPAR products
- Global coverage
- Combined Terra+Aqua (MCD) provides better coverage
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"


@AcquisitionRegistry.register('MODIS_LAI')
@AcquisitionRegistry.register('MCD15')
class MODISLAIAcquirer(BaseAcquisitionHandler):
    """
    Handles MODIS LAI/FPAR data acquisition via NASA AppEEARS.

    Downloads 8-day composite LAI and FPAR data for specified
    spatial extent and time period.

    Configuration:
        MODIS_LAI_PRODUCT: Product ID ('MCD15A2H', 'MOD15A2H', 'MYD15A2H')
        MODIS_LAI_LAYERS: Layers to download (default: Lai_500m, Fpar_500m)
        MODIS_LAI_QC: Include QC layers (default: True)
    """

    PRODUCTS = {
        'MCD15A2H': 'MCD15A2H.061',  # Combined Terra+Aqua
        'MOD15A2H': 'MOD15A2H.061',  # Terra only
        'MYD15A2H': 'MYD15A2H.061',  # Aqua only
    }

    DEFAULT_LAYERS = ['Lai_500m', 'Fpar_500m', 'FparLai_QC']

    def download(self, output_dir: Path) -> Path:
        """
        Download MODIS LAI/FPAR data via AppEEARS.

        Args:
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded data
        """
        self.logger.info("Starting MODIS LAI/FPAR data acquisition via AppEEARS")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data
        force_download = self.config_dict.get('FORCE_DOWNLOAD', False)
        existing_files = list(output_dir.glob("*LAI*.nc")) + list(output_dir.glob("*Lai*.nc"))
        if existing_files and not force_download:
            self.logger.info(f"MODIS LAI data already exists: {len(existing_files)} files")
            return output_dir

        # Get credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise ValueError(
                "NASA Earthdata credentials required. Set via environment variables "
                "(EARTHDATA_USERNAME, EARTHDATA_PASSWORD) or ~/.netrc"
            )

        # Get configuration
        product = self.config_dict.get('MODIS_LAI_PRODUCT', 'MCD15A2H')
        product_id = self.PRODUCTS.get(product, f"{product}.061")
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

        self.logger.info(f"MODIS LAI download complete: {output_dir}")
        return output_dir

    def _get_layers(self) -> List[str]:
        """Get layers to download."""
        config_layers = self.config_dict.get('MODIS_LAI_LAYERS')
        if config_layers:
            if isinstance(config_layers, str):
                return [config_layers]
            return list(config_layers)

        layers = ['Lai_500m', 'Fpar_500m']
        if self.config_dict.get('MODIS_LAI_QC', True):
            layers.append('FparLai_QC')

        return layers

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

        task_name = f"MODIS_LAI_{self.domain_name}_{self.start_date.strftime('%Y%m%d')}"

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
            raise ValueError("Bounding box required for MODIS LAI acquisition")

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
