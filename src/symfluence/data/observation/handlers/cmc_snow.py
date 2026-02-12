"""
CMC Snow Depth Analysis Observation Handler.

Provides acquisition and preprocessing of CMC (Canadian Meteorological Centre)
Daily Snow Depth Analysis data (NSIDC-0447) for hydrological model calibration
and validation.

CMC Snow Overview:
    Data Type: Analyzed snow depth (station + satellite assimilation)
    Resolution: ~24 km (706x706 Northern Hemisphere grid)
    Coverage: Northern Hemisphere
    Variables: Snow depth (cm) -> converted to SWE (mm)
    Temporal: Daily (stored as yearly GeoTIFFs with 365/366 bands)
    Units: Snow depth in cm, converted to SWE in mm

Output Format:
    CSV with columns: datetime, swe_mm
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

# Default bulk snow density (kg/m^3) for depth-to-SWE conversion
DEFAULT_SNOW_DENSITY = 200.0


@ObservationRegistry.register('cmc_snow')
@ObservationRegistry.register('cmc_swe')
class CMCSnowHandler(BaseObservationHandler):
    """
    Handles CMC Snow Depth Analysis data acquisition and processing.

    Provides basin-averaged monthly snow water equivalent time series from
    the CMC Daily Snow Depth Analysis product for model calibration and
    validation.

    Converts snow depth (cm) to SWE (mm) using a configurable bulk snow
    density (default: 200 kg/m^3).
    """

    obs_type = "snow"
    source_name = "CMC"

    def acquire(self) -> Path:
        """
        Locate or download CMC snow depth data.

        Returns:
            Path to directory containing CMC GeoTIFF files
        """
        data_access = self._config_value(
            'DATA_ACCESS',
            typed_path=lambda: self.config.domain.data_access,
            default='local'
        )
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        cmc_path = self._config_value(
            'CMC_SNOW_PATH', 'CMC_PATH',
            typed_path=lambda: self.config.evaluation.cmc_snow.path,
            default='default'
        )
        if isinstance(cmc_path, str) and cmc_path.lower() == 'default':
            cmc_dir = self.project_dir / "observations" / "snow" / "cmc"
        else:
            cmc_dir = Path(cmc_path)

        cmc_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        force_download = self._config_value(
            'FORCE_DOWNLOAD',
            typed_path=lambda: self.config.data.force_download,
            default=False
        )

        existing_files = list(cmc_dir.glob("*.tif"))
        if existing_files and not force_download:
            self.logger.info(f"Using existing CMC data: {len(existing_files)} GeoTIFF files")
            return cmc_dir

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for CMC snow data")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('CMC_SNOW', self.config, self.logger)
            return acquirer.download(cmc_dir)

        return cmc_dir

    def process(self, input_path: Path) -> Path:
        """
        Process CMC GeoTIFFs to monthly basin-averaged SWE time series.

        Args:
            input_path: Path to directory containing CMC GeoTIFF files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing CMC snow data for domain: {self.domain_name}")

        # Configuration
        snow_density = float(self._config_value(
            'CMC_SNOW_DENSITY',
            default=DEFAULT_SNOW_DENSITY
        ))
        temporal_agg = self._config_value(
            'CMC_TEMPORAL_AGG',
            default='monthly'
        )

        # Find GeoTIFF files
        tif_files = sorted(input_path.glob("cmc_sdepth_dly_*.tif"))
        if not tif_files:
            tif_files = sorted(input_path.glob("*.tif"))
        if not tif_files:
            self.logger.warning("No CMC GeoTIFF files found")
            return input_path

        self.logger.info(f"Processing {len(tif_files)} CMC GeoTIFF files")

        # Get bounding box
        lat_min = lat_max = lon_min = lon_max = None
        if self.bbox:
            lat_min = self.bbox.get('lat_min')
            lat_max = self.bbox.get('lat_max')
            lon_min = self.bbox.get('lon_min')
            lon_max = self.bbox.get('lon_max')

        all_results: List[dict] = []

        for tif_file in tif_files:
            # Extract year from filename
            year = self._extract_year(tif_file.name)
            if year is None:
                continue

            # Filter by experiment time range
            if self.start_date and year < self.start_date.year:
                continue
            if self.end_date and year > self.end_date.year:
                continue

            year_results = self._process_geotiff(
                tif_file, year, snow_density, temporal_agg,
                lat_min, lat_max, lon_min, lon_max
            )
            all_results.extend(year_results)

        if not all_results:
            self.logger.warning("No CMC snow data could be extracted")
            return input_path

        # Create DataFrame
        df = pd.DataFrame(all_results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        # Filter to experiment time range
        if self.start_date is not None:
            df = df[df['datetime'] >= self.start_date]
        if self.end_date is not None:
            df = df[df['datetime'] <= self.end_date]

        # Ensure non-negative SWE
        df['swe_mm'] = df['swe_mm'].clip(lower=0)

        # Save output
        output_dir = self._get_observation_dir('snow')
        output_file = output_dir / f"{self.domain_name}_cmc_swe_processed.csv"
        df[['datetime', 'swe_mm']].to_csv(output_file, index=False)

        self.logger.info(f"CMC processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df)}")
        if len(df) > 0:
            self.logger.info(f"  Mean SWE: {df['swe_mm'].mean():.1f} mm")
            self.logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        return output_file

    def _extract_year(self, filename: str) -> Optional[int]:
        """Extract year from CMC filename like cmc_sdepth_dly_2005_v01.2.tif."""
        import re
        match = re.search(r'(\d{4})', filename)
        if match:
            return int(match.group(1))
        return None

    def _process_geotiff(
        self,
        filepath: Path,
        year: int,
        snow_density: float,
        temporal_agg: str,
        lat_min: Optional[float],
        lat_max: Optional[float],
        lon_min: Optional[float],
        lon_max: Optional[float],
    ) -> List[dict]:
        """Process a single yearly CMC GeoTIFF."""
        results: list[dict] = []

        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            self.logger.error("rasterio is required for CMC GeoTIFF processing")
            return results

        try:
            with rasterio.open(filepath) as src:
                # Determine pixel window for basin bbox
                transform = src.transform
                crs = src.crs

                if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
                    xmin, ymin, xmax, ymax = lon_min, lat_min, lon_max, lat_max

                    # Transform bbox if raster is in a projected CRS
                    if crs and not crs.is_geographic:
                        from rasterio.warp import transform_bounds
                        from rasterio.crs import CRS
                        xmin, ymin, xmax, ymax = transform_bounds(
                            CRS.from_epsg(4326), crs,
                            lon_min, lat_min, lon_max, lat_max
                        )

                    # Convert to pixel coordinates
                    inv_transform = ~transform
                    col_min_f, row_min_f = inv_transform * (xmin, ymax)
                    col_max_f, row_max_f = inv_transform * (xmax, ymin)

                    col_start = max(0, int(min(col_min_f, col_max_f)))
                    col_end = min(src.width, int(max(col_min_f, col_max_f)) + 1)
                    row_start = max(0, int(min(row_min_f, row_max_f)))
                    row_end = min(src.height, int(max(row_min_f, row_max_f)) + 1)
                else:
                    col_start, row_start = 0, 0
                    col_end, row_end = src.width, src.height

                if row_end <= row_start or col_end <= col_start:
                    self.logger.warning(f"Empty pixel window for {filepath.name}")
                    return results

                window = Window(
                    col_start, row_start,
                    col_end - col_start, row_end - row_start
                )
                nodata = src.nodata
                total_bands = src.count

                # Determine which bands to read
                if temporal_agg == 'monthly':
                    # Read 1st of each month
                    bands_to_read = {}
                    for month in range(1, 13):
                        doy = datetime(year, month, 1).timetuple().tm_yday
                        if doy <= total_bands:
                            bands_to_read[month] = doy
                else:
                    # Read all bands (daily)
                    bands_to_read = {doy: doy for doy in range(1, total_bands + 1)}

                for key, band_idx in bands_to_read.items():
                    if temporal_agg == 'monthly':
                        date = datetime(year, key, 1)
                    else:
                        date = datetime(year, 1, 1) + pd.Timedelta(days=key - 1)

                    data = src.read(band_idx, window=window).astype(np.float64)

                    # Mask nodata and unreasonable values
                    if nodata is not None:
                        data[data == nodata] = np.nan
                    data[(data < 0) | (data > 999)] = np.nan

                    n_valid = np.sum(~np.isnan(data))
                    if n_valid == 0:
                        continue

                    # Basin mean snow depth (cm) -> SWE (mm)
                    mean_depth_cm = np.nanmean(data)
                    swe_mm = mean_depth_cm * (snow_density / 100.0)

                    results.append({
                        'datetime': date.strftime('%Y-%m-%d'),
                        'swe_mm': swe_mm,
                    })

        except Exception as e:
            self.logger.warning(f"Error processing {filepath.name}: {e}")

        return results
