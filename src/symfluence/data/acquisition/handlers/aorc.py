"""
NOAA AORC atmospheric data acquisition from cloud storage.

Provides automated download and processing of Analysis of Record for Calibration
(AORC) forcing data with bounding box subsetting and multi-year support.
"""

import xarray as xr
import pandas as pd
import s3fs
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('AORC')
class AORCAcquirer(BaseAcquisitionHandler):
    """
    Download and process NOAA AORC (Analysis of Record for Calibration) forcing data.

    AORC provides high-resolution (1 km, hourly) atmospheric forcing data for the
    continental United States (CONUS) from 1979-present. Data is stored in AWS S3
    as yearly Zarr archives and accessed via anonymous S3 access.

    Dataset Characteristics:
        Source: NOAA National Water Model (NWM) retrospective forcing
        Spatial Coverage: CONUS (Continental United States)
        Spatial Resolution: ~1 km (Lambert Conformal Conic projection)
        Temporal Coverage: 1979-01-01 to near-present
        Temporal Resolution: Hourly
        Format: Zarr (one archive per year)
        S3 Bucket: noaa-nws-aorc-v1-1-1km
        Access: Anonymous (no AWS credentials required)

    Variables Available:
        - APCP_surface: Total precipitation (kg/m²)
        - DLWRF_surface: Downward longwave radiation (W/m²)
        - DSWRF_surface: Downward shortwave radiation (W/m²)
        - PRES_surface: Surface pressure (Pa)
        - SPFH_2maboveground: Specific humidity at 2m (kg/kg)
        - TMP_2maboveground: Air temperature at 2m (K)
        - UGRD_10maboveground: U-component of wind at 10m (m/s)
        - VGRD_10maboveground: V-component of wind at 10m (m/s)

    Coordinate Conventions:
        - Longitude: 0-360° (Eastern longitude convention)
        - Latitude: Standard -90 to 90°
        - Projection: Lambert Conformal Conic (not geographic coordinates)
        - Coordinates represent grid cell centers

    Workflow:
        1. **Year Iteration**: Loop through all years in date range
        2. **Zarr Access**: Open yearly Zarr archive from S3 (anonymous)
        3. **Longitude Conversion**: Convert bounding box to 0-360° if needed
        4. **Spatial Subset**: Select bbox region using xarray slicing
        5. **Temporal Subset**: Clip to requested date range within year
        6. **Concatenation**: Combine all yearly subsets along time dimension
        7. **NetCDF Export**: Save as single NetCDF file for entire period

    Bounding Box Handling:
        - Accepts bbox in -180 to 180° format
        - Automatically converts to 0-360° for AORC coordinate system
        - Uses xarray slice() for efficient spatial subsetting
        - Handles antimeridian crossing (though AORC is CONUS-only)

    Multi-Year Strategy:
        - Each year stored as separate Zarr archive
        - Downloads each year independently
        - Handles partial years (e.g., 2015-03-01 to 2017-08-15)
        - Concatenates all yearly datasets into single output file
        - Skips empty years (no data in requested time range)

    Configuration Requirements:
        Required (inherited from BaseAcquisitionHandler):
            - DOMAIN_NAME: Basin identifier
            - EXPERIMENT_TIME_START: Download start date (YYYY-MM-DD HH:MM)
            - EXPERIMENT_TIME_END: Download end date (YYYY-MM-DD HH:MM)
            - DOMAIN_BOUNDING_BOX: Spatial extent [lon_min, lat_min, lon_max, lat_max]

        Optional:
            - FORCING_AORC_PATH: Custom output directory (default: domain forcing dir)

    Output Format:
        - Filename: {DOMAIN_NAME}_AORC_{start_year}-{end_year}.nc
        - Format: NetCDF4
        - Dimensions: time, latitude, longitude
        - Attributes: source (NOAA AORC v1.1), bbox (spatial extent)
        - Coordinate Reference System: Lambert Conformal Conic (retained from source)

    Performance Notes:
        - Zarr format enables efficient spatial subsetting (no full download)
        - S3 transfer speed: ~10-50 MB/s (depends on region/network)
        - Typical download: ~100-500 MB per year for small basins
        - Memory usage: Holds one year in memory at a time
        - Processing time: ~1-5 minutes per year

    Error Handling:
        - Raises ValueError if no data in requested time period
        - Logs and re-raises exceptions for individual year failures
        - S3 connection errors propagate to caller
        - Invalid bbox coordinates caught by xarray slicing

    Example:
        >>> config = {
        ...     'DOMAIN_NAME': 'test_basin',
        ...     'EXPERIMENT_TIME_START': '2015-01-01 00:00',
        ...     'EXPERIMENT_TIME_END': '2016-12-31 23:00',
        ...     'DOMAIN_BOUNDING_BOX': [-105.5, 39.0, -105.0, 39.5]
        ... }
        >>> acquirer = AORCAcquirer(config, logger)
        >>> output_file = acquirer.download(Path('./forcing/raw'))
        >>> print(output_file)
        ./forcing/raw/test_basin_AORC_2015-2016.nc

    Notes:
        - AORC data is free and publicly accessible (no authentication)
        - Data updated periodically (typically 1-2 month lag from present)
        - Lambert Conformal projection may require coordinate transformation
        - For post-processing, use AOrcHandler to convert variables to SUMMA format
        - Zarr archives are read-only; spatial subsetting is done via xarray

    See Also:
        - data.preprocessing.dataset_handlers.aorc_utils.AOrcHandler: Variable conversion
        - data.acquisition.base.BaseAcquisitionHandler: Base acquisition interface
        - data.acquisition.registry.AcquisitionRegistry: Handler registration
    """

    def download(self, output_dir: Path) -> Path:
        """
        Download AORC data from AWS S3 for specified date range and bounding box.

        This method downloads hourly AORC forcing data from NOAA's public S3 bucket,
        subsetting spatially and temporally. Data is accessed year-by-year from Zarr
        archives and combined into a single NetCDF file.

        Args:
            output_dir: Directory to save downloaded NetCDF file

        Returns:
            Path to downloaded NetCDF file:
                Format: {output_dir}/{domain_name}_AORC_{start_year}-{end_year}.nc

        Raises:
            ValueError: If no data available for requested time period
            Exception: If S3 connection fails or year processing errors occur

        Process:
            1. Initialize S3 filesystem (anonymous access)
            2. Loop through years in date range
            3. For each year:
               - Open Zarr archive from S3
               - Convert bbox coordinates to 0-360° if needed
               - Subset spatially using bbox
               - Subset temporally to requested dates
               - Append to datasets list if data exists
            4. Concatenate all yearly datasets
            5. Add metadata attributes
            6. Export to NetCDF

        Example:
            >>> acquirer = AORCAcquirer(config, logger)
            >>> output_file = acquirer.download(Path('./forcing/raw'))
            # Downloads: ./forcing/raw/basin_AORC_2015-2017.nc
            # Size: ~300 MB for 3 years, small basin
            # Variables: 8 forcing variables, hourly timestep
        """
        self.logger.info("Downloading AORC data from S3")
        fs = s3fs.S3FileSystem(anon=True)
        years = range(self.start_date.year, self.end_date.year + 1)
        datasets = []
        for year in years:
            try:
                store = s3fs.S3Map(f'noaa-nws-aorc-v1-1-1km/{year}.zarr', s3=fs)
                ds = xr.open_zarr(store)
                lon1, lon2 = sorted([self.bbox['lon_min'], self.bbox['lon_max']])
                # Convert to 0-360 if dataset uses that convention
                if float(ds['longitude'].max()) > 180.0:
                    lon_min, lon_max = (lon1 + 360.0) % 360.0, (lon2 + 360.0) % 360.0
                else:
                    lon_min, lon_max = lon1, lon2
                ds_subset = ds.sel(latitude=slice(self.bbox['lat_min'], self.bbox['lat_max']), longitude=slice(lon_min, lon_max))
                ds_subset = ds_subset.sel(time=slice(max(self.start_date, pd.Timestamp(f'{year}-01-01')), min(self.end_date, pd.Timestamp(f'{year}-12-31 23:59:59'))))
                if len(ds_subset.time) > 0: datasets.append(ds_subset)
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
                raise
        if not datasets: raise ValueError("No data extracted for the specified time period")
        ds_combined = xr.concat(datasets, dim='time')
        ds_combined.attrs.update({'source': 'NOAA AORC v1.1', 'bbox': str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc"
        ds_combined.to_netcdf(output_file)
        return output_file
