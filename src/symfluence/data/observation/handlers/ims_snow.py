"""
IMS Snow Cover Observation Handler.

Provides acquisition and preprocessing of NOAA IMS (Interactive Multisensor
Snow and Ice Mapping System) snow cover data for hydrological model calibration
and validation.

IMS Snow Overview:
    Data Type: Multi-sensor fusion snow/ice cover (binary classification)
    Resolution: 1km (2014+), 4km (2004+), 24km (1997+)
    Coverage: Northern Hemisphere
    Variables: Snow cover fraction (derived from binary snow/no-snow maps)
    Temporal: Daily
    Units: Fraction (0-1)

Output Format:
    CSV with columns: datetime, sca_fraction
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry


@ObservationRegistry.register('ims_snow')
@ObservationRegistry.register('ims_sca')
class IMSSnowHandler(BaseObservationHandler):
    """
    Handles IMS snow cover data acquisition and processing.

    Provides basin-averaged daily snow cover fraction time series from
    NOAA's IMS multi-sensor snow product for model calibration and
    validation.

    IMS provides binary snow/no-snow classification. The handler computes
    snow-covered area (SCA) fraction as the ratio of snow-covered land
    pixels to total land pixels within the basin bounding box.
    """

    obs_type = "snow"
    source_name = "IMS"

    def acquire(self) -> Path:
        """
        Locate or download IMS snow cover data.

        Returns:
            Path to directory or NetCDF file containing IMS data
        """
        data_access = self._config_value(
            'DATA_ACCESS',
            typed_path=lambda: self.config.domain.data_access,
            default='local'
        )
        if isinstance(data_access, str):
            data_access = data_access.lower()

        # Determine data directory
        ims_path = self._config_value(
            'IMS_SNOW_PATH', 'IMS_PATH',
            typed_path=lambda: self.config.evaluation.ims_snow.path,
            default='default'
        )
        if isinstance(ims_path, str) and ims_path.lower() == 'default':
            ims_dir = self.project_observations_dir / "snow" / "ims"
        else:
            ims_dir = Path(ims_path)

        ims_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing processed NetCDF
        force_download = self._config_value(
            'FORCE_DOWNLOAD',
            typed_path=lambda: self.config.data.force_download,
            default=False
        )

        existing_nc = list(ims_dir.glob("*IMS_snow*.nc"))
        if existing_nc and not force_download:
            self.logger.info(f"Using existing IMS data: {existing_nc[0].name}")
            return existing_nc[0]

        # Trigger cloud acquisition if enabled
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for IMS snow data")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('IMS_SNOW', self.config, self.logger)
            return acquirer.download(ims_dir)

        # Return directory for local data
        return ims_dir

    def process(self, input_path: Path) -> Path:
        """
        Process IMS snow cover data to daily basin-averaged SCA fraction.

        Args:
            input_path: Path to IMS NetCDF file or directory containing files

        Returns:
            Path to processed CSV file
        """
        self.logger.info(f"Processing IMS snow cover data for domain: {self.domain_name}")

        # Find NetCDF files
        if input_path.is_file():
            nc_files = [input_path]
        else:
            nc_files = sorted(input_path.glob("*IMS*.nc"))
            if not nc_files:
                nc_files = sorted(input_path.glob("*.nc"))

        if not nc_files:
            self.logger.warning("No IMS NetCDF files found")
            return input_path

        self.logger.info(f"Processing {len(nc_files)} IMS files")

        all_dfs = []

        for nc_file in nc_files:
            try:
                ds = self._open_dataset(nc_file)
            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                self.logger.warning(f"Failed to open {nc_file.name}: {e}")
                continue

            with ds:
                # Look for snow_fraction variable (output of IMS acquirer)
                if 'snow_fraction' in ds.data_vars:
                    sca = ds['snow_fraction']

                    # Convert to DataFrame
                    df = sca.to_dataframe().reset_index()

                    # Standardize time column
                    time_col = None
                    for col in ['time', 'date', 'datetime']:
                        if col in df.columns:
                            time_col = col
                            break

                    if time_col is None:
                        self.logger.warning(f"No time column found in {nc_file.name}")
                        continue

                    df['datetime'] = pd.to_datetime(df[time_col])
                    df = df[['datetime', 'snow_fraction']].rename(
                        columns={'snow_fraction': 'sca_fraction'}
                    )

                else:
                    # Try to compute from raw grid data
                    df = self._compute_sca_from_grid(ds)
                    if df is None:
                        continue

                all_dfs.append(df)

        if not all_dfs:
            self.logger.warning("No IMS snow data could be extracted")
            return input_path

        # Combine
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined = df_combined.sort_values('datetime')
        df_combined = df_combined.drop_duplicates(subset='datetime', keep='first')

        # Filter to experiment time range
        if self.start_date is not None:
            df_combined = df_combined[df_combined['datetime'] >= self.start_date]
        if self.end_date is not None:
            df_combined = df_combined[df_combined['datetime'] <= self.end_date]

        # Ensure valid range [0, 1]
        df_combined['sca_fraction'] = df_combined['sca_fraction'].clip(0, 1)

        # Save output
        output_dir = self._get_observation_dir('snow')
        output_file = output_dir / f"{self.domain_name}_ims_sca_processed.csv"
        df_combined.to_csv(output_file, index=False)

        self.logger.info(f"IMS processing complete: {output_file}")
        self.logger.info(f"  Records: {len(df_combined)}")
        if len(df_combined) > 0:
            self.logger.info(
                f"  SCA range: {df_combined['sca_fraction'].min():.3f} - "
                f"{df_combined['sca_fraction'].max():.3f}"
            )
            self.logger.info(
                f"  Date range: {df_combined['datetime'].min()} to "
                f"{df_combined['datetime'].max()}"
            )

        return output_file

    def _compute_sca_from_grid(self, ds: xr.Dataset) -> Optional[pd.DataFrame]:
        """
        Compute SCA fraction from raw IMS grid data.

        IMS value codes:
            0 = Outside NH, 1 = Water, 2 = Land (no snow),
            3 = Sea Ice, 4 = Snow-covered Land
        """
        # Find a suitable grid variable
        grid_var = None
        for var in ds.data_vars:
            if 'ims' in var.lower() or 'snow' in var.lower() or 'sca' in var.lower():
                grid_var = var
                break

        if grid_var is None and len(ds.data_vars) == 1:
            grid_var = list(ds.data_vars)[0]

        if grid_var is None:
            self.logger.warning("Could not identify IMS grid variable")
            return None

        data = ds[grid_var]

        # If the data has spatial dimensions, compute SCA fraction per timestep
        if 'time' not in data.dims:
            self.logger.warning("No time dimension in IMS grid data")
            return None

        results = []
        for t in data.time.values:
            snapshot = data.sel(time=t).values
            land_pixels = ((snapshot == 2) | (snapshot == 4)).sum()
            snow_pixels = (snapshot == 4).sum()
            sca = float(snow_pixels) / float(land_pixels) if land_pixels > 0 else float('nan')
            results.append({
                'datetime': pd.Timestamp(t),
                'sca_fraction': sca,
            })

        if results:
            return pd.DataFrame(results)
        return None
