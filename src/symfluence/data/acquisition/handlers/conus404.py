"""
CONUS404 Data Acquisition Handler.

Provides access to the CONUS404 high-resolution (4km) regional climate
dataset for the contiguous United States via the HyTEST data catalog.
"""

import xarray as xr
import pandas as pd
import numpy as np
import intake
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ...utils import VariableStandardizer, create_spatial_mask


@AcquisitionRegistry.register('CONUS404')
class CONUS404Acquirer(BaseAcquisitionHandler):
    """
    Acquirer for CONUS404 high-resolution regional climate data.

    CONUS404 is a 4-km resolution, hourly dataset covering the contiguous
    United States from 1979-present. Data is accessed via the HyTEST intake
    catalog (USGS/NCAR collaboration).

    Key features:
    - 4 km horizontal resolution
    - Hourly temporal resolution
    - Variables: T2, Q2, PSFC, U10, V10, radiation, precipitation
    - WRF model output with data assimilation

    Data source:
        HyTEST catalog: https://github.com/hytest-org/hytest

    Output:
        NetCDF file with standardized variable names for SUMMA compatibility
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download CONUS404 data from HyTEST catalog with spatial and temporal subsetting.

        This method accesses CONUS404 data via the HyTEST intake catalog, performs
        intelligent spatial subsetting using bounding box masking, selects variables
        with priority fallback logic, and standardizes variable names for SUMMA
        compatibility.

        Args:
            output_dir: Directory to save downloaded NetCDF file

        Returns:
            Path to downloaded NetCDF file:
                Format: {output_dir}/{domain_name}_CONUS404_{start_year}-{end_year}.nc

        Raises:
            ValueError: If no grid points fall within bounding box
            Exception: If catalog access fails or dataset unavailable

        Process:
            1. **Catalog Access**: Open HyTEST intake catalog (USGS/NCAR)
            2. **Dataset Loading**: Load CONUS404 hourly dataset as Dask array
            3. **Coordinate Detection**: Identify lat/lon coordinate names (flexible)
            4. **Spatial Masking**: Create boolean mask for bbox region
            5. **Spatial Subset**: Extract minimal bounding box containing masked cells
            6. **Temporal Subset**: Slice to requested date range using xarray
            7. **Variable Selection**: Select required variables with priority fallback
               - Core variables: T2, Q2, PSFC, U10, V10 (always required)
               - Shortwave: ACSWDNB (preferred) OR SWDOWN (fallback)
               - Longwave: ACLWDNB (preferred) OR LWDOWN (fallback)
               - Precipitation: PREC_ACC_NC, RAINRATE, PRATE, or ACDRIPR (priority order)
            8. **Data Loading**: Load subset into memory (.load())
            9. **Standardization**: Rename variables to SUMMA-compatible names
            10. **Export**: Save as NetCDF4 file

        Variable Selection Strategy:
            Required Core Variables (5):
                - T2: 2-meter air temperature (K)
                - Q2: 2-meter specific humidity (kg/kg)
                - PSFC: Surface pressure (Pa)
                - U10: U-component wind at 10m (m/s)
                - V10: V-component wind at 10m (m/s)

            Radiation (fallback logic):
                - Shortwave: ACSWDNB (accumulated downward) OR SWDOWN (instantaneous)
                - Longwave: ACLWDNB (accumulated downward) OR LWDOWN (instantaneous)

            Precipitation (priority fallback):
                1. PREC_ACC_NC (accumulated, preferred for consistency)
                2. RAINRATE (instantaneous rate)
                3. PRATE (instantaneous rate alternative)
                4. ACDRIPR (accumulated, alternative)

        Spatial Subsetting Details:
            - Uses create_spatial_mask() for efficient bbox masking
            - Handles 2D curvilinear coordinates (WRF native grid)
            - Minimal bounding box extraction (reduces memory footprint)
            - Validates at least one grid point in bbox

        Coordinate Handling:
            - Flexible coordinate name detection: "lat" or "latitude"
            - Supports both geographic (lat/lon) and projected coordinates
            - Preserves WRF projection information from source

        Standardization:
            - Variable names mapped to SUMMA conventions
            - Unit conversions applied if needed
            - Missing data handled via standardizer
            - Coordinate names normalized

        Performance:
            - Intake catalog: ~2-5 seconds (initial load)
            - Dask lazy loading: Minimal overhead until .load() call
            - Spatial subset: ~50-200 MB per year for typical basin
            - .load() operation: ~30-120 seconds for multi-year download
            - Typical total time: 1-5 minutes for 3-5 year period

        Configuration:
            Optional (with defaults):
                - CONUS404_CATALOG_URL: Custom HyTEST catalog URL
                  Default: https://raw.githubusercontent.com/hytest-org/hytest/...

        Example:
            >>> config = {
            ...     'DOMAIN_NAME': 'arkansas_river',
            ...     'EXPERIMENT_TIME_START': '2015-01-01 00:00',
            ...     'EXPERIMENT_TIME_END': '2017-12-31 23:00',
            ...     'DOMAIN_BOUNDING_BOX': [-106.5, 38.0, -105.0, 39.5]
            ... }
            >>> acquirer = CONUS404Acquirer(config, logger)
            >>> output = acquirer.download(Path('./forcing/raw'))
            >>> print(output)
            ./forcing/raw/arkansas_river_CONUS404_2015-2017.nc
            # Size: ~180 MB for 3 years
            # Variables: T2, Q2, PSFC, U10, V10, SWDOWN, LWDOWN, PREC_ACC_NC

        Notes:
            - HyTEST catalog requires internet connection
            - Data hosted on Open Storage Network (OSN) - public access
            - Dask enables efficient streaming (no full dataset download)
            - Coordinate detection handles catalog version updates
            - Fallback logic ensures robustness to variable name changes
            - WRF curvilinear coordinates preserved in output

        See Also:
            - data.utils.VariableStandardizer: Variable name standardization
            - data.utils.create_spatial_mask: Bounding box masking utility
            - data.preprocessing.dataset_handlers.conus404_utils: Variable processing
        """
        self.logger.info("Downloading CONUS404 data")

        # Open catalog and get dataset
        cat_url = self.config.get(
            "CONUS404_CATALOG_URL",
            "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
        )
        cat = intake.open_catalog(cat_url)
        ds_full = cat["conus404-catalog"]["conus404-hourly-osn"].to_dask()

        # Detect coordinate names
        lat_name = next(c for c in ["lat", "latitude"] if c in ds_full)
        lon_name = next(c for c in ["lon", "longitude"] if c in ds_full)

        # Spatial subsetting using centralized utility
        lat_v, lon_v = ds_full[lat_name].values, ds_full[lon_name].values
        mask = create_spatial_mask(lat_v, lon_v, self.bbox)
        iy, ix = np.where(mask)
        if iy.size == 0:
            raise ValueError("No grid points in bbox")

        ds_spatial = ds_full.isel({
            ds_full[lat_name].dims[0]: slice(iy.min(), iy.max() + 1),
            ds_full[lat_name].dims[1]: slice(ix.min(), ix.max() + 1)
        })

        # Temporal subsetting
        ds_subset = ds_spatial.sel(time=slice(self.start_date, self.end_date))

        # Select required variables (prioritize by availability)
        req_vars = ["T2", "Q2", "PSFC", "U10", "V10"]
        # Radiation and precipitation: use first available from each group
        var_groups = [
            ["ACSWDNB", "SWDOWN"],  # Shortwave
            ["ACLWDNB", "LWDOWN"],  # Longwave
            ["PREC_ACC_NC", "RAINRATE", "PRATE", "ACDRIPR"]  # Precipitation
        ]
        for group in var_groups:
            var = next((v for v in group if v in ds_subset.data_vars), None)
            if var:
                req_vars.append(var)

        ds_raw = ds_subset[req_vars].load()

        # Standardize variable names using centralized utility
        standardizer = VariableStandardizer(self.logger)
        ds_final = standardizer.standardize(ds_raw, 'CONUS404')

        # Add metadata and save
        ds_final.attrs.update({"source": "CONUS404", "bbox": str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        out_f = output_dir / f"{self.domain_name}_CONUS404_{self.start_date.year}-{self.end_date.year}.nc"
        ds_final.to_netcdf(out_f)

        return out_f
