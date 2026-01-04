"""
Observation Data Acquisition Handlers

Provides cloud acquisition for various observational datasets:
- SMAP Soil Moisture (NSIDC THREDDS/NCSS)
- ESA CCI Soil Moisture (CDS API)
- FLUXCOM Evapotranspiration
"""
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import xarray as xr
import os

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('SMAP')
class SMAPAcquirer(BaseAcquisitionHandler):
    """
    Acquires SMAP Soil Moisture data via NSIDC THREDDS NCSS.
    Requires Earthdata Login credentials (via ~/.netrc or env vars).
    """

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Starting SMAP Soil Moisture acquisition via NSIDC THREDDS")
        
        # NSIDC THREDDS NCSS endpoint
        thredds_base = self.config.get('SMAP_THREDDS_BASE', "https://n5eil01u.ecs.nsidc.org/thredds/ncss/grid")
        product = self.config.get('SMAP_PRODUCT', 'SMAP_L4_SM_gph_v4')
        
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
        
        start_date = self.start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date = self.end_date.strftime("%Y-%m-%dT23:59:59Z")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        out_nc = output_dir / f"{self.domain_name}_SMAP_raw.nc"
        
        if out_nc.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_nc

        params = {
            "var": "sm_surface",
            "north": lat_max,
            "south": lat_min,
            "west": lon_min,
            "east": lon_max,
            "time_start": start_date,
            "time_end": end_date,
            "accept": "netcdf4"
        }
        
        # Construct URL for the specific product/version/date
        # Note: NSIDC structure is complex (YYYY.MM.DD directories). 
        # For NCSS Grid, we often need the exact file path or an aggregation.
        # This implementation assumes an aggregated NCML exists or the user provides a direct OPeNDAP/NCSS URL.
        # If 'SMAP_THREDDS_URL' is provided, use it directly.
        
        url = self.config.get('SMAP_THREDDS_URL', f"{thredds_base}/{product}/aggregated.ncml")
        self.logger.info(f"Querying SMAP THREDDS: {url}")
        
        # Setup session with Earthdata Auth
        session = requests.Session()
        user = os.environ.get("EARTHDATA_USERNAME")
        password = os.environ.get("EARTHDATA_PASSWORD")
        
        if user and password:
            session.auth = (user, password)
        else:
            # Check for .netrc
            pass # requests handles .netrc automatically
            
        try:
            response = session.get(url, params=params, stream=True, timeout=600)
            
            # Handle redirects for auth
            if response.status_code == 401:
                self.logger.error("Authentication failed. Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD or use a .netrc file.")
                raise PermissionError("Earthdata Login required")
                
            response.raise_for_status()
            
            with open(out_nc, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            self.logger.info(f"Successfully downloaded SMAP data to {out_nc}")
            return out_nc
        except Exception as e:
            self.logger.error(f"SMAP acquisition failed: {e}")
            raise

@AcquisitionRegistry.register('ESA_CCI_SM')
class ESACCISMAcquirer(BaseAcquisitionHandler):
    """
    Acquires ESA CCI Soil Moisture data via Copernicus CDS.
    """

    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi required for ESA CCI SM acquisition")
            
        self.logger.info("Starting ESA CCI Soil Moisture acquisition via CDS")
        
        c = cdsapi.Client()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        # CDS returns a tar/zip for multiple files
        out_file = output_dir / f"{self.domain_name}_ESA_CCI_SM_raw.tar.gz"
        
        if out_file.exists() and not self.config.get('FORCE_DOWNLOAD', False):
            return out_file

        years = sorted(list(set([str(y) for y in range(self.start_date.year, self.end_date.year + 1)])))
        
        request = {
            'variable': 'volumetric_surface_soil_moisture',
            'type_of_sensor': 'combined',
            'time_aggregation': 'day_average',
            'year': years,
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            'area': [self.bbox['lat_max'], self.bbox['lon_min'], self.bbox['lat_min'], self.bbox['lon_max']],
            'format': 'tgz',
            'version': 'v07.1' # Specify version to ensure consistency
        }
        
        c.retrieve('satellite-soil-moisture', request, str(out_file))
        
        self.logger.info(f"ESA CCI SM data downloaded to {out_file}")
        
        # Auto-extract
        import tarfile
        extract_dir = output_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(out_file, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            
        self.logger.info(f"Extracted ESA CCI SM data to {extract_dir}")
        return extract_dir

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
        local_pattern = self.config.get('FLUXCOM_FILE_PATTERN', "*.nc")
        existing_files = list(et_dir.glob(local_pattern))
        if existing_files and not self.config.get('FORCE_DOWNLOAD', False):
            self.logger.info(f"Found existing FLUXCOM files in {et_dir}")
            return et_dir

        # Option 2: Download from URL
        download_url = self.config.get('FLUXCOM_DOWNLOAD_URL')
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

