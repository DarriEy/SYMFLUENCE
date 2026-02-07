"""FLUXCOM Evapotranspiration Acquisition Handler

Provides acquisition for FLUXCOM machine learning upscaled ET products.

FLUXCOM Overview:
    Data Type: Machine-learning upscaled eddy covariance ET
    Resolution: 0.5° (FLUXCOM) or 0.05° (FLUXCOM-X)
    Coverage: Global
    Temporal: Daily/Monthly, 2001-2020+ depending on version
    Source: Max Planck Institute for Biogeochemistry / ICOS Carbon Portal

Products:
    FLUXCOM (original): 0.5° resolution, multiple ML methods
    FLUXCOM-X: 0.05° resolution, extended version with improved methods

Data Access Options:
    1. ICOS Carbon Portal (FLUXCOM-X): https://www.icos-cp.eu/data-products/2G60-ZHAK
       - NetCDF files available for download after registration
       - Daily/monthly LE (latent heat flux) which can be converted to ET

    2. Max Planck BGI Data Portal (original FLUXCOM):
       - Requires registration and data request
       - ftp://dataportal.bgc-jena.mpg.de/

    3. Google Earth Engine:
       - FLUXCOM available as ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
       - Contains derived ET variables

Configuration:
    FLUXCOM_DOWNLOAD_URL: Direct URL to download data file
    FLUXCOM_FILE_PATTERN: Pattern for local files (default: "*.nc")
    FLUXCOM_VARIABLE: Variable to extract (default: "LE" for latent heat)
    FLUXCOM_CONVERT_TO_ET: True (default) - convert LE to ET
    FLUXCOM_VERSION: 'original' or 'x' (default: 'original')

Unit Conversion:
    LE (W/m²) to ET (mm/day): ET = LE * 86400 / (2.45e6)
    where 2.45e6 J/kg is the latent heat of vaporization

References:
    - Jung et al. (2019): https://doi.org/10.1038/s41597-019-0076-8
    - FLUXCOM-X: https://www.icos-cp.eu/data-products/2G60-ZHAK
"""

import numpy as np
import xarray as xr
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


# Latent heat of vaporization (J/kg)
LATENT_HEAT_VAPORIZATION = 2.45e6
SECONDS_PER_DAY = 86400


@AcquisitionRegistry.register('FLUXCOM_ET')
@AcquisitionRegistry.register('FLUXCOM')
class FLUXCOMETAcquirer(BaseAcquisitionHandler):
    """
    Acquires FLUXCOM Evapotranspiration data.

    FLUXCOM is a machine learning approach that upscales eddy covariance
    flux measurements using satellite and meteorological data.

    Since FLUXCOM data requires registration, this handler supports:
    1. Direct download from a user-provided URL
    2. Local file discovery (if data is manually placed)
    3. Instructions for manual data access
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download/process FLUXCOM ET data.

        Args:
            output_dir: Directory to save processed files

        Returns:
            Path to output directory or processed file
        """
        self.logger.info("Starting FLUXCOM ET acquisition")

        et_dir = output_dir / "et" / "fluxcom"
        et_dir.mkdir(parents=True, exist_ok=True)

        processed_file = et_dir / f"{self.domain_name}_FLUXCOM_ET.nc"

        if processed_file.exists() and not self.config_dict.get('FORCE_DOWNLOAD', False):
            self.logger.info(f"Using existing FLUXCOM file: {processed_file}")
            return processed_file

        # Option 1: Check for existing local files
        local_pattern = self.config_dict.get('FLUXCOM_FILE_PATTERN', "*.nc")
        existing_files = list(et_dir.glob(local_pattern))

        if existing_files:
            self.logger.info(f"Found {len(existing_files)} local FLUXCOM files")
            return self._process_local_files(existing_files, processed_file)

        # Option 2: Download from URL
        download_url = self.config_dict.get('FLUXCOM_DOWNLOAD_URL')
        if download_url:
            self.logger.info("Downloading FLUXCOM data from URL")
            downloaded_file = self._download_from_url(download_url, et_dir)
            return self._process_local_files([downloaded_file], processed_file)

        # Option 3: Try ICOS Carbon Portal
        icos_result = self._try_icos_carbon_portal(et_dir)
        if icos_result:
            return self._process_local_files([icos_result], processed_file)

        # Fail with helpful instructions
        self._provide_data_access_instructions(et_dir)
        raise ValueError(
            f"FLUXCOM data not available. Please download manually to: {et_dir}\n"
            "See log output for data access instructions."
        )

    def _download_from_url(self, url: str, output_dir: Path) -> Path:
        """Download file from URL."""
        filename = url.split('/')[-1]
        if not filename.endswith('.nc'):
            filename = "fluxcom_downloaded.nc"

        out_file = output_dir / filename

        if out_file.exists() and not self.config_dict.get('FORCE_DOWNLOAD', False):
            return out_file

        self.logger.info(f"Downloading: {filename}")
        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()

            with open(out_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            self.logger.info(f"Downloaded: {out_file}")
            return out_file

        except Exception as e:
            self.logger.error(f"Failed to download FLUXCOM data: {e}")
            raise

    def _try_icos_carbon_portal(self, output_dir: Path) -> Optional[Path]:
        """
        Try to access FLUXCOM-X from ICOS Carbon Portal.

        Note: ICOS requires authentication for most data products.
        This method checks if data is available and provides guidance.
        """
        # ICOS Carbon Portal data objects for FLUXCOM-X
        # These would require authentication tokens to download directly
        self.logger.info("Checking ICOS Carbon Portal for FLUXCOM-X data...")

        # The ICOS CP API requires authentication for downloads
        # We can at least check if the data exists
        try:
            # Check ICOS data portal landing page
            icos_url = "https://meta.icos-cp.eu/collections/2G60-ZHAK"
            response = requests.get(icos_url, timeout=30)

            if response.status_code == 200:
                self.logger.info(
                    "FLUXCOM-X data available at ICOS Carbon Portal.\n"
                    "Please register at https://cpauth.icos-cp.eu/ and download:\n"
                    f"  https://www.icos-cp.eu/data-products/2G60-ZHAK\n"
                    f"Place downloaded files in: {output_dir}"
                )

        except Exception as e:
            self.logger.debug(f"Could not check ICOS portal: {e}")

        return None

    def _process_local_files(self, files: List[Path], output_file: Path) -> Path:
        """Process local FLUXCOM files."""
        self.logger.info(f"Processing {len(files)} FLUXCOM files...")

        variable = self.config_dict.get('FLUXCOM_VARIABLE', 'LE')
        convert_to_et = self.config_dict.get('FLUXCOM_CONVERT_TO_ET', True)

        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])

        datasets = []
        for f in files:
            try:
                ds = xr.open_dataset(f)

                # Find the variable (LE, ET, etc.)
                var_name = None
                for v in ds.data_vars:
                    if variable.lower() in v.lower():
                        var_name = v
                        break
                    if 'latent' in v.lower() or 'et' in v.lower():
                        var_name = v
                        break

                if not var_name:
                    self.logger.warning(f"Could not find {variable} in {f.name}")
                    continue

                da = ds[var_name]

                # Subset to domain
                lat_dim = 'lat' if 'lat' in da.dims else 'latitude'
                lon_dim = 'lon' if 'lon' in da.dims else 'longitude'

                if lat_dim in da.dims and lon_dim in da.dims:
                    # Handle coordinate ordering
                    lat_coords = da[lat_dim].values
                    if lat_coords[0] > lat_coords[-1]:
                        # Descending latitudes
                        da = da.sel(
                            **{lat_dim: slice(lat_max, lat_min),
                               lon_dim: slice(lon_min, lon_max)}
                        )
                    else:
                        da = da.sel(
                            **{lat_dim: slice(lat_min, lat_max),
                               lon_dim: slice(lon_min, lon_max)}
                        )

                datasets.append(da)

            except Exception as e:
                self.logger.warning(f"Error reading {f}: {e}")
                continue

        if not datasets:
            raise RuntimeError("No valid FLUXCOM data found in files")

        # Combine datasets
        if len(datasets) > 1:
            combined = xr.concat(datasets, dim='time')
        else:
            combined = datasets[0]

        combined = combined.sortby('time')

        # Convert LE to ET if needed
        if convert_to_et and 'le' in variable.lower():
            # LE (W/m²) to ET (mm/day)
            # ET = LE * seconds_per_day / latent_heat
            combined = combined * SECONDS_PER_DAY / LATENT_HEAT_VAPORIZATION
            units = 'mm/day'
            long_name = 'Evapotranspiration'
        else:
            units = combined.attrs.get('units', 'W/m2')
            long_name = combined.attrs.get('long_name', variable)

        # Compute spatial mean
        lat_dim = 'lat' if 'lat' in combined.dims else 'latitude'
        lon_dim = 'lon' if 'lon' in combined.dims else 'longitude'

        if lat_dim in combined.dims and lon_dim in combined.dims:
            # Weight by cos(lat) for proper area averaging
            weights = np.cos(np.deg2rad(combined[lat_dim]))
            weights = weights / weights.sum()
            basin_mean = combined.weighted(weights).mean(dim=[lat_dim, lon_dim])
        else:
            basin_mean = combined

        # Create output dataset
        ds_out = xr.Dataset({
            'ET': basin_mean.rename('ET'),
        })

        ds_out['ET'].attrs = {
            'long_name': long_name,
            'units': units,
            'source': 'FLUXCOM',
        }

        ds_out.attrs['title'] = 'FLUXCOM Evapotranspiration'
        ds_out.attrs['source'] = 'Max Planck Institute / ICOS Carbon Portal'
        ds_out.attrs['original_variable'] = variable
        ds_out.attrs['created'] = datetime.now().isoformat()
        ds_out.attrs['domain'] = self.domain_name
        ds_out.attrs['bbox'] = f"{lon_min},{lat_min},{lon_max},{lat_max}"

        ds_out.to_netcdf(output_file)

        self.logger.info(f"Saved FLUXCOM ET: {output_file}")

        # Create CSV
        self._create_csv_output(basin_mean, output_file.parent)

        return output_file

    def _create_csv_output(self, da: xr.DataArray, output_dir: Path):
        """Create CSV timeseries output."""
        csv_file = output_dir / f"{self.domain_name}_FLUXCOM_ET_timeseries.csv"

        df = da.to_dataframe()
        if 'ET' not in df.columns:
            df.columns = ['ET']

        df.index.name = 'date'
        df.columns = ['et_mm_day']

        df.to_csv(csv_file)
        self.logger.info(f"Created CSV: {csv_file}")

    def _provide_data_access_instructions(self, output_dir: Path):
        """Log instructions for accessing FLUXCOM data."""
        instructions = """
========================================================================
FLUXCOM Data Access Instructions
========================================================================

FLUXCOM data requires registration. Options:

1. FLUXCOM-X (Recommended - higher resolution 0.05°)
   - Register at: https://cpauth.icos-cp.eu/
   - Download from: https://www.icos-cp.eu/data-products/2G60-ZHAK
   - Product: FLUXCOM-X-BASE daily/monthly Latent Heat Flux
   - Coverage: 2001-2020, Global

2. Original FLUXCOM (0.5° resolution)
   - Request access at: https://www.bgc-jena.mpg.de/geodb/projects/Data.php
   - FTP: ftp://dataportal.bgc-jena.mpg.de/ (requires credentials)
   - Product: FLUXCOM RS/RS+METEO daily/monthly LE

3. Google Earth Engine (indirect access)
   - FLUXCOM variables available in TerraClimate collection
   - ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')

4. Direct URL (if you have a download link)
   - Set FLUXCOM_DOWNLOAD_URL in your config file

After downloading, place NetCDF files in:
  {output_dir}

Configuration example:
  FLUXCOM_VARIABLE: 'LE'           # Latent heat flux
  FLUXCOM_CONVERT_TO_ET: true      # Convert to ET (mm/day)
  FLUXCOM_FILE_PATTERN: '*.nc'     # File pattern to search

========================================================================
"""
        self.logger.info(instructions.format(output_dir=output_dir))
