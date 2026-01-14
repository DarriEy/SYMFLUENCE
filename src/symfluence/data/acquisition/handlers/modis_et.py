"""
MODIS MOD16A2 Evapotranspiration (ET) Acquisition Handler

Acquires MOD16A2 (Terra) and MYD16A2 (Aqua) 8-day composite ET products
via NASA AppEEARS API and processes them into daily-mean ET values.

References:
- MOD16A2: https://lpdaac.usgs.gov/products/mod16a2v061/
- MYD16A2: https://lpdaac.usgs.gov/products/myd16a2v061/
- AppEEARS: https://appeears.earthdatacloud.nasa.gov/

Products provide 8-day composite values at 500m resolution:
- ET_500m: Total Evapotranspiration (kg/m²/8day)
- LE_500m: Average Latent Heat Flux (J/m²/day)
- PET_500m: Total Potential ET (kg/m²/8day)
- ET_QC_500m: Quality Control flags
"""
import requests
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from ..registry import AcquisitionRegistry
from .appeears_base import BaseAppEEARSAcquirer



@AcquisitionRegistry.register('MOD16')
@AcquisitionRegistry.register('MODIS_ET')
@AcquisitionRegistry.register('MOD16A2')
class MOD16ETAcquirer(BaseAppEEARSAcquirer):
    """
    Acquires MODIS MOD16A2/MYD16A2 Evapotranspiration data via NASA AppEEARS API.

    The handler downloads 8-day composite ET products and optionally converts
    them to daily mean values for comparison with model outputs.

    Configuration:
        MOD16_PRODUCTS: List of products, default ['MOD16A2.061']
        MOD16_VARIABLE: 'ET_500m' (default), 'LE_500m', 'PET_500m'
        MOD16_MERGE_PRODUCTS: True to merge Terra+Aqua, False for Terra only
        MOD16_CONVERT_TO_DAILY: True (default) - convert 8-day to daily mean
        MOD16_QC_FILTER: True (default) - filter by quality flags
        MOD16_UNITS: 'mm_day' (default), 'kg_m2_8day' (raw)
        EARTHDATA_USERNAME/EARTHDATA_PASSWORD: NASA Earthdata credentials
    """

    # MOD16A2 QC flag interpretation (ET_QC_500m)
    # Bits 0-1: MODLAND_QC (00=Good, 01=Other, 10=Marginal, 11=Cloud/NoData)
    # Bit 2: Sensor (0=Terra, 1=Aqua)
    # Bit 3: DeadDetector (0=No, 1=Yes)
    # Bits 4-7: CloudState (0=Clear, 1=Cloudy, 2=Mixed, 3=NotSet)
    QC_GOOD_MASK = 0b00000011  # Bits 0-1 for quality
    QC_GOOD_VALUES = {0b00, 0b01}  # Good quality or Other quality
    FILL_VALUE = 32767  # Fill value for no data

    # Unit conversion: kg/m²/8day to mm/day (water density ~1000 kg/m³)
    # 1 kg/m² = 1 mm, so kg/m²/8day / 8 = mm/day
    SCALE_FACTOR = 0.1  # MOD16A2 scale factor
    DAYS_IN_COMPOSITE = 8

    def download(self, output_dir: Path) -> Path:
        """Download MOD16A2 ET products via AppEEARS."""
        self.logger.info("Starting MOD16 ET acquisition via AppEEARS")

        output_dir.mkdir(parents=True, exist_ok=True)
        processed_file = output_dir / f"{self.domain_name}_MOD16_ET.nc"

        if processed_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing MOD16 ET file: {processed_file}")
            return processed_file

        # Get products to download
        products = self.config.get('MOD16_PRODUCTS', ['MOD16A2.061'])
        if isinstance(products, str):
            products = [p.strip() for p in products.split(',')]

        # Add Aqua product if merging is enabled
        merge_products = self.config.get('MOD16_MERGE_PRODUCTS', False)
        if merge_products and 'MYD16A2.061' not in products:
            products.append('MYD16A2.061')

        # Check for Earthdata credentials
        username, password = self._get_earthdata_credentials()
        if not username or not password:
            raise RuntimeError(
                "Earthdata credentials required for MOD16 acquisition. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables "
                "or add to ~/.netrc (machine urs.earthdata.nasa.gov)"
            )

        # Get variable to download
        variable = self.config.get('MOD16_VARIABLE', 'ET_500m')

        # Download each product via AppEEARS
        product_files = {}
        for product in products:
            try:
                product_file = self._download_product_appeears(
                    output_dir, product, variable, username, password
                )
                if product_file and product_file.exists():
                    product_files[product] = product_file
            except Exception as e:
                self.logger.warning(f"Failed to download {product}: {e}")

        if not product_files:
            raise RuntimeError("No MOD16 ET products could be downloaded")

        # Process and optionally merge products
        self._process_products(product_files, processed_file, variable)

        self.logger.info(f"MOD16 ET acquisition complete: {processed_file}")
        return processed_file

    def _download_product_appeears(
        self,
        output_dir: Path,
        product: str,
        variable: str,
        username: str,
        password: str
    ) -> Optional[Path]:
        """Download a single MOD16 product via AppEEARS API."""
        self.logger.info(f"Downloading {product} ({variable}) via AppEEARS")

        # Parse product name (e.g., 'MOD16A2.061')
        parts = product.split('.')
        product_name = parts[0]
        version = parts[1] if len(parts) > 1 else '061'

        output_file = output_dir / f"{self.domain_name}_{product_name}_raw.nc"

        if output_file.exists() and not self._get_config_value(lambda: self.config.data.force_download, default=False, dict_key='FORCE_DOWNLOAD'):
            self.logger.info(f"Using existing file: {output_file}")
            return output_file

        # Login to AppEEARS
        token = self._appeears_login(username, password)
        if not token:
            raise RuntimeError("Failed to authenticate with AppEEARS")

        try:
            # Submit task
            task_id = self._submit_appeears_task(
                token, product_name, version, variable
            )

            if not task_id:
                raise RuntimeError(f"Failed to submit AppEEARS task for {product}")

            # Wait for task completion
            if not self._wait_for_task(token, task_id):
                raise RuntimeError(f"AppEEARS task {task_id} did not complete")

            # Download results
            self._download_task_results(token, task_id, output_dir, product_name)

            # Process downloaded files into single NetCDF
            self._consolidate_appeears_output(output_dir, product_name, output_file)

            return output_file

        finally:
            self._appeears_logout(token)

    def _submit_appeears_task(
        self,
        token: str,
        product: str,
        version: str,
        variable: str
    ) -> Optional[str]:
        """Submit an AppEEARS area request task for MOD16."""
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

        product_full = f"{product}.{version}"
        task_name = f"SYMFLUENCE_{self.domain_name}_{product}_ET_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Format dates for AppEEARS (MM-DD-YYYY)
        start_date = self.start_date.strftime("%m-%d-%Y")
        end_date = self.end_date.strftime("%m-%d-%Y")

        # Build layers - include QC layer for filtering
        layers = [
            {"product": product_full, "layer": variable}
        ]

        # Add QC layer if filtering is enabled
        if self.config.get('MOD16_QC_FILTER', True):
            layers.append({"product": product_full, "layer": "ET_QC_500m"})

        task_request = {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [{
                    "startDate": start_date,
                    "endDate": end_date
                }],
                "layers": layers,
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

    def _process_products(
        self,
        product_files: Dict[str, Path],
        output_file: Path,
        variable: str
    ):
        """Process and merge ET products, convert units."""
        self.logger.info("Processing MOD16 ET products")

        convert_to_daily = self.config.get('MOD16_CONVERT_TO_DAILY', True)
        qc_filter = self.config.get('MOD16_QC_FILTER', True)
        target_units = self.config.get('MOD16_UNITS', 'mm_day')

        datasets = {}
        for product, path in product_files.items():
            try:
                ds = xr.open_dataset(path)
                # Find the ET variable
                et_var = None
                for var in ds.data_vars:
                    if variable.lower().replace('_500m', '') in var.lower():
                        et_var = var
                        break
                    if 'et' in var.lower() and 'qc' not in var.lower():
                        et_var = var
                        break
                if et_var:
                    datasets[product] = {'data': ds[et_var], 'ds': ds}
                    # Also get QC if available
                    for qc_var in ds.data_vars:
                        if 'qc' in qc_var.lower():
                            datasets[product]['qc'] = ds[qc_var]
                            break
            except Exception as e:
                self.logger.warning(f"Failed to open {path}: {e}")

        if not datasets:
            raise RuntimeError("No valid ET datasets to process")

        # Process each dataset
        processed_das = []
        for product, data_dict in datasets.items():
            da = data_dict['data'].copy()

            # Apply scale factor if needed (MOD16A2 stores as scaled integers)
            if da.dtype in [np.int16, np.int32]:
                da = da.astype(float) * self.SCALE_FACTOR

            # Apply fill value mask
            da = da.where(da != self.FILL_VALUE * self.SCALE_FACTOR)

            # Apply QC filter
            if qc_filter and 'qc' in data_dict:
                qc = data_dict['qc']
                # Good quality: bits 0-1 are 00 or 01
                good_quality = (qc & self.QC_GOOD_MASK).isin(list(self.QC_GOOD_VALUES))
                da = da.where(good_quality)

            processed_das.append(da)

        # Merge if multiple products (Terra + Aqua)
        if len(processed_das) > 1:
            # Stack and take mean (or max for conservative estimate)
            stacked = xr.concat(processed_das, dim='product')
            et_merged = stacked.mean(dim='product', skipna=True)
        else:
            et_merged = processed_das[0]

        # Convert units: kg/m²/8day to mm/day
        # 1 kg/m² water = 1 mm
        if target_units == 'mm_day' and convert_to_daily:
            et_merged = et_merged / self.DAYS_IN_COMPOSITE
            units = 'mm/day'
            long_name = 'Evapotranspiration (daily mean from 8-day composite)'
        else:
            units = 'kg/m2/8day'
            long_name = 'Evapotranspiration (8-day composite)'

        # Compute spatial mean for basin-scale calibration
        lat_dim = 'lat' if 'lat' in et_merged.dims else 'y'
        lon_dim = 'lon' if 'lon' in et_merged.dims else 'x'

        if lat_dim in et_merged.dims and lon_dim in et_merged.dims:
            et_basin_mean = et_merged.mean(dim=[lat_dim, lon_dim], skipna=True)
        else:
            et_basin_mean = et_merged

        # Create output dataset
        ds_out = xr.Dataset({
            'ET': et_merged.rename('ET'),
            'ET_basin_mean': et_basin_mean.rename('ET_basin_mean')
        })

        ds_out['ET'].attrs = {
            'long_name': long_name,
            'units': units,
            'source': 'MODIS MOD16A2/MYD16A2',
            'variable': variable
        }
        ds_out['ET_basin_mean'].attrs = {
            'long_name': f'Basin-averaged {long_name}',
            'units': units
        }

        ds_out.attrs['title'] = 'MODIS MOD16 Evapotranspiration'
        ds_out.attrs['source_products'] = list(product_files.keys())
        ds_out.attrs['created'] = datetime.now().isoformat()
        ds_out.attrs['domain'] = self.domain_name

        ds_out.to_netcdf(output_file)

        # Cleanup
        for data_dict in datasets.values():
            data_dict['ds'].close()

        self.logger.info(f"Processed MOD16 ET saved: {output_file}")

        # Also create CSV for observation handler compatibility
        self._create_csv_output(et_basin_mean, output_file.parent)

    def _create_csv_output(self, et_series: xr.DataArray, output_dir: Path):
        """Create CSV output compatible with observation handler."""
        csv_file = output_dir / f"{self.domain_name}_MOD16_ET_timeseries.csv"

        df = et_series.to_dataframe().reset_index()
        df.columns = ['date' if 'time' in c.lower() else c for c in df.columns]

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Rename to match expected column names
        if 'ET_basin_mean' in df.columns:
            df = df.rename(columns={'ET_basin_mean': 'et_mm_day'})

        df.to_csv(csv_file)
        self.logger.info(f"Created CSV timeseries: {csv_file}")
