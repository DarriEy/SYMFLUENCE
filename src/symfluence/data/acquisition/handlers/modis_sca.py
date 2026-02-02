"""
MODIS Snow Cover Area (SCA) Acquisition Handler

Acquires MOD10A1 (Terra) and MYD10A1 (Aqua) snow cover products via NASA AppEEARS API
and merges them into a combined daily product with improved spatial/temporal coverage.

References:
- MOD10A1: https://nsidc.org/data/mod10a1
- MYD10A1: https://nsidc.org/data/myd10a1
- AppEEARS: https://appeears.earthdatacloud.nasa.gov/
"""
import requests
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from ..registry import AcquisitionRegistry
from .appeears_base import BaseAppEEARSAcquirer



@AcquisitionRegistry.register('MODIS_SCA')
@AcquisitionRegistry.register('MODIS_SNOW_MERGED')
class MODISSCAAcquirer(BaseAppEEARSAcquirer):
    """
    Acquires MODIS Snow Cover Area data from both Terra (MOD10A1) and Aqua (MYD10A1)
    satellites via NASA AppEEARS API, then merges products for improved daily coverage.

    The merge strategy prioritizes:
    1. Cloud-free pixels from either satellite
    2. When both have valid data, uses maximum SCA (conservative for snow detection)
    3. Quality flags for filtering unreliable observations

    Configuration:
        MODIS_SCA_PRODUCTS: List of products, default ['MOD10A1.061', 'MYD10A1.061']
        MODIS_SCA_MERGE_STRATEGY: 'max' (default), 'mean', 'terra_priority', 'aqua_priority'
        MODIS_SCA_CLOUD_FILTER: True (default) - filter cloud-covered pixels
        MODIS_SCA_QA_FILTER: True (default) - filter by quality flags
        MODIS_SCA_MIN_VALID_RATIO: 0.1 (default) - minimum fraction of valid pixels
        EARTHDATA_USERNAME/EARTHDATA_PASSWORD: NASA Earthdata credentials
    """

    # NDSI Snow Cover value interpretation (MOD10A1/MYD10A1)
    # 0-100: NDSI snow cover percentage
    # 200: missing data
    # 201: no decision
    # 211: night
    # 237: inland water
    # 239: ocean
    # 250: cloud
    # 254: detector saturated
    # 255: fill value
    VALID_SNOW_RANGE = (0, 100)
    CLOUD_VALUE = 250
    MISSING_VALUES = {200, 201, 211, 237, 239, 250, 254, 255}

    def download(self, output_dir: Path) -> Path:
        """Download and merge MOD10A1/MYD10A1 snow cover products."""
        self.logger.info("Starting MODIS SCA acquisition (Terra + Aqua merge)")

        output_dir.mkdir(parents=True, exist_ok=True)
        merged_file = output_dir / f"{self.domain_name}_MODIS_SCA_merged.nc"

        if merged_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing merged SCA file: {merged_file}")
            return merged_file

        # Get products to download
        products = self._get_config_value(lambda: self.config.evaluation.modis_snow.products, default=['MOD10A1.061', 'MYD10A1.061'], dict_key='MODIS_SCA_PRODUCTS')
        if isinstance(products, str):
            products = [p.strip() for p in products.split(',')]

        # Check for Earthdata credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            self.logger.warning(
                "Earthdata credentials not found. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD "
                "environment variables or add to ~/.netrc"
            )
            # Fall back to legacy THREDDS method if available
            return self._download_via_thredds(output_dir, products[0])

        # Download each product via AppEEARS
        product_files = {}
        for product in products:
            try:
                product_file = self._download_product_appeears(
                    output_dir, product, username, password
                )
                if product_file and product_file.exists():
                    product_files[product] = product_file
            except Exception as e:
                self.logger.warning(f"Failed to download {product}: {e}")

        if not product_files:
            raise RuntimeError("No MODIS SCA products could be downloaded")

        # Merge products if we have multiple
        if len(product_files) > 1:
            self._merge_products(product_files, merged_file)
        else:
            # Just copy/rename the single product
            single_file = list(product_files.values())[0]
            if single_file != merged_file:
                import shutil
                shutil.copy(single_file, merged_file)

        self.logger.info(f"MODIS SCA acquisition complete: {merged_file}")
        return merged_file

    def _generate_date_chunks(self, chunk_years: int = 4) -> List[tuple]:
        """Split the full date range into smaller chunks for AppEEARS."""
        chunks = []
        chunk_start = self.start_date
        while chunk_start < self.end_date:
            chunk_end = min(
                chunk_start.replace(year=chunk_start.year + chunk_years) - pd.Timedelta(days=1),
                self.end_date
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + pd.Timedelta(days=1)
        return chunks

    def _download_product_appeears(
        self,
        output_dir: Path,
        product: str,
        username: str,
        password: str
    ) -> Optional[Path]:
        """Download a single MODIS product via AppEEARS API with time chunking."""
        self.logger.info(f"Downloading {product} via AppEEARS")

        # Parse product name (e.g., 'MOD10A1.061' -> product='MOD10A1', version='061')
        parts = product.split('.')
        product_name = parts[0]
        version = parts[1] if len(parts) > 1 else '061'

        output_file = output_dir / f"{self.domain_name}_{product_name}_raw.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing file: {output_file}")
            return output_file

        # Split into time chunks to avoid AppEEARS file limits
        chunks = self._generate_date_chunks(chunk_years=4)
        self.logger.info(f"Splitting {product} request into {len(chunks)} time chunks")

        # Login to AppEEARS
        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_label = f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"
                self.logger.info(
                    f"Chunk {i+1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}"
                )

                # Submit task for this chunk
                task_id = self._submit_appeears_task(
                    token, product_name, version,
                    start_date=chunk_start, end_date=chunk_end
                )

                if not task_id:
                    self.logger.warning(f"Failed to submit chunk {i+1}, skipping")
                    continue

                # Wait for task completion
                if not self._wait_for_task(token, task_id):
                    self.logger.warning(f"Chunk {i+1} task did not complete, skipping")
                    continue

                # Download results
                self._download_task_results(
                    token, task_id, output_dir, f"{product_name}_{chunk_label}"
                )

            # Consolidate all chunk files into single NetCDF
            self._consolidate_appeears_output(output_dir, product_name, output_file)

            return output_file

        finally:
            self._appeears_logout(token)

    def _submit_appeears_task(
        self,
        token: str,
        product: str,
        version: str,
        start_date=None,
        end_date=None
    ) -> Optional[str]:
        """Submit an AppEEARS area request task."""
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        # Create GeoJSON polygon for the bounding box
        coordinates = [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min]
        ]]

        # Full product name with version (e.g., MOD10A1.061)
        product_full = f"{product}.{version}"

        task_name = f"SYMFLUENCE_{self.domain_name}_{product}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Format dates for AppEEARS (use overrides if provided)
        req_start = start_date if start_date is not None else self.start_date
        req_end = end_date if end_date is not None else self.end_date
        start_date = req_start.strftime("%m-%d-%Y")
        end_date = req_end.strftime("%m-%d-%Y")

        # Build task request
        task_request = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date,
                    "endDate": end_date
                }],
                "layers": [{
                    "product": product_full,
                    "layer": "NDSI_Snow_Cover"
                }],
                "geo": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates
                        },
                        "properties": {}
                    }]
                },
                "output": {
                    "format": {
                        "type": "netcdf4"
                    },
                    "projection": "geographic"
                }
            }
        }

        try:
            response = requests.post(
                f"{self.APPEEARS_BASE}/task",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=task_request,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            task_id = result.get('task_id')
            self.logger.info(f"Submitted AppEEARS task: {task_id}")
            return task_id
        except Exception as e:
            self.logger.error(f"Failed to submit AppEEARS task: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response: {e.response.text[:500]}")
            return None

    def _convert_time_to_dates(self, time_values) -> List:
        """Convert time values to date objects, handling cftime."""
        dates = []
        for t in time_values:
            try:
                # Try standard datetime conversion
                if hasattr(t, 'date'):
                    dates.append(t.date() if callable(t.date) else t.date)
                elif hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
                    # cftime object
                    from datetime import date
                    dates.append(date(t.year, t.month, t.day))
                else:
                    # Try pandas conversion
                    dates.append(pd.to_datetime(t).date())
            except (ValueError, TypeError, AttributeError):
                # Last resort: string parsing
                dates.append(pd.to_datetime(str(t)[:10]).date())
        return dates

    def _merge_products(self, product_files: Dict[str, Path], output_file: Path):
        """Merge Terra and Aqua products into combined daily SCA."""
        self.logger.info("Merging MOD10A1 and MYD10A1 products")

        merge_strategy = self._get_config_value(lambda: self.config.evaluation.modis_snow.merge_strategy, default='max', dict_key='MODIS_SCA_MERGE_STRATEGY').lower()
        cloud_filter = self._get_config_value(lambda: self.config.evaluation.modis_snow.cloud_filter, default=True, dict_key='MODIS_SCA_CLOUD_FILTER')

        datasets = {}
        for product, path in product_files.items():
            try:
                ds = xr.open_dataset(path)
                # Identify the snow cover variable
                sca_var = None
                for var in ds.data_vars:
                    if 'snow' in var.lower() or 'ndsi' in var.lower():
                        sca_var = var
                        break
                if sca_var:
                    datasets[product] = ds[sca_var]
            except Exception as e:
                self.logger.warning(f"Failed to open {path}: {e}")

        if not datasets:
            raise RuntimeError("No valid datasets to merge")

        # Get common time range
        all_times = set()
        for da in datasets.values():
            if 'time' in da.dims:
                time_dates = self._convert_time_to_dates(da.time.values)
                all_times.update(time_dates)

        if not all_times:
            # No time dimension - just use first dataset
            first_da = list(datasets.values())[0]
            ds_out = xr.Dataset({'NDSI_Snow_Cover': first_da})
            ds_out.to_netcdf(output_file)
            return

        all_times = sorted(all_times)

        # Prepare merged array
        merged_data = []

        for date in all_times:
            day_data = []
            for product, da in datasets.items():
                if 'time' not in da.dims:
                    continue
                # Select this date
                time_dates = self._convert_time_to_dates(da.time.values)
                day_mask = [d == date for d in time_dates]
                if not any(day_mask):
                    continue
                day_slice = da.isel(time=day_mask)
                if day_slice.size > 0:
                    day_data.append(day_slice.values)

            if not day_data:
                continue

            # Stack and merge
            stacked = np.stack([d.squeeze() if d.ndim > 2 else d for d in day_data], axis=0)

            # Apply cloud filtering
            if cloud_filter:
                # Mask cloud values
                stacked = np.where(stacked == self.CLOUD_VALUE, np.nan, stacked)

            # Mask other invalid values
            for mv in self.MISSING_VALUES:
                stacked = np.where(stacked == mv, np.nan, stacked.astype(float))

            # Apply merge strategy
            if merge_strategy == 'max':
                merged = np.nanmax(stacked, axis=0)
            elif merge_strategy == 'mean':
                merged = np.nanmean(stacked, axis=0)
            elif merge_strategy == 'terra_priority':
                # Use Terra if available, else Aqua
                merged = stacked[0] if len(stacked) > 0 else stacked[-1]
            elif merge_strategy == 'aqua_priority':
                merged = stacked[-1] if len(stacked) > 1 else stacked[0]
            else:
                merged = np.nanmax(stacked, axis=0)

            merged_data.append((date, merged))

        if not merged_data:
            raise RuntimeError("No data after merging")

        # Create output dataset
        times = [datetime.combine(d, datetime.min.time()) for d, _ in merged_data]
        data_stack = np.stack([d for _, d in merged_data], axis=0)

        # Get spatial coordinates from first dataset
        first_da = list(datasets.values())[0]
        lat_dim = 'lat' if 'lat' in first_da.dims else 'y'
        lon_dim = 'lon' if 'lon' in first_da.dims else 'x'

        coords = {'time': times}
        if lat_dim in first_da.coords:
            coords[lat_dim] = first_da.coords[lat_dim].values
        if lon_dim in first_da.coords:
            coords[lon_dim] = first_da.coords[lon_dim].values

        dims = ['time', lat_dim, lon_dim] if data_stack.ndim == 3 else ['time']

        da_merged = xr.DataArray(
            data_stack,
            dims=dims,
            coords=coords,
            name='NDSI_Snow_Cover',
            attrs={
                'long_name': 'NDSI Snow Cover (Merged Terra+Aqua)',
                'units': 'percent',
                'valid_range': [0, 100],
                'merge_strategy': merge_strategy,
                'source_products': list(product_files.keys())
            }
        )

        ds_out = xr.Dataset({'NDSI_Snow_Cover': da_merged})
        ds_out.attrs['title'] = 'Merged MODIS Snow Cover Area'
        ds_out.attrs['source'] = 'MOD10A1 + MYD10A1 via AppEEARS'
        ds_out.attrs['created'] = datetime.now().isoformat()

        ds_out.to_netcdf(output_file)

        # Cleanup
        for da in datasets.values():
            da.close()

        self.logger.info(f"Merged SCA product saved: {output_file}")

    def _download_via_thredds(self, output_dir: Path, product: str) -> Path:
        """Fallback to THREDDS download (legacy, single product)."""
        self.logger.warning("Falling back to THREDDS download (single product only)")

        from . import modis
        # Use existing MODIS snow acquirer as fallback
        legacy_acquirer = modis.MODISSnowAcquirer(self.config, self.logger)
        return legacy_acquirer.download(output_dir)
