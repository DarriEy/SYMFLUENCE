"""
Forcing data processing utilities for HYPE model.

Handles merging of forcing data from multiple NetCDF files and conversion
to HYPE-compatible daily observation formats.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Third-party imports
import cdo
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ..utilities import BaseForcingProcessor


class HYPEForcingProcessor(BaseForcingProcessor):
    """
    Processor for HYPE forcing data.

    Handles:
    - Merging hourly NetCDF forcing files
    - Rolling time for time zone offsets
    - Resampling hourly data to daily HYPE format (Pobs, Tobs, TMAXobs, TMINobs)
    - Unit conversions and HYPE-specific file formatting
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        forcing_input_dir: Path,
        output_path: Path,
        cache_path: Path,
        timeshift: int = 0,
        forcing_units: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HYPE forcing processor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            forcing_input_dir: Path to input basin-averaged NetCDF files
            output_path: Path to output HYPE settings directory
            cache_path: Path for temporary processing files
            timeshift: Hour offset for time zone correction
            forcing_units: Mapping of variables to units and names
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_input_dir,
            output_path=output_path,
            cache_path=cache_path
        )
        # Keep forcing_input_dir as alias for backward compatibility
        self.forcing_input_dir = self.input_path
        self.timeshift = timeshift
        self.forcing_units = forcing_units or {}

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "HYPE"

    def process_forcing(self) -> None:
        """Execute the full HYPE forcing processing workflow."""
        self.logger.info("Merging HYPE forcing files...")
        merged_forcing_path = self._merge_forcing_files()

        if not merged_forcing_path or not merged_forcing_path.exists():
            self.logger.error("Forcing merge failed, cannot proceed with daily conversion")
            return

        self.logger.info("Converting hourly forcing to HYPE daily observations...")
        self._convert_to_daily_obs(merged_forcing_path)

        # Cleanup
        if merged_forcing_path.exists():
            merged_forcing_path.unlink()

    def _merge_forcing_files(self) -> Optional[Path]:
        """Merge individual NetCDF files using CDO with xarray fallback."""
        easymore_nc_files = sorted(list(self.forcing_input_dir.glob('*.nc')))
        if not easymore_nc_files:
            self.logger.warning(f"No forcing files found in {self.forcing_input_dir}")
            return None

        merged_forcing_path = self.cache_path / 'merged_forcing.nc'

        # Try CDO first (faster for large datasets)
        try:
            cdo_obj = cdo.Cdo()
            # If initialization succeeded, try merging
            self.logger.info("Merging forcing files with CDO...")

            # split the files in batches as cdo cannot mergetime long list of file names
            batch_size = 20
            if len(easymore_nc_files) < batch_size:
                batch_size = len(easymore_nc_files)

            files_split: List[Any] = np.array_split(easymore_nc_files, batch_size)
            intermediate_files = []

            for i in tqdm(range(batch_size), desc="Merging forcing batches"):
                batch_files = [str(f) for f in files_split[i].tolist()]
                batch_output = self.cache_path / f"forcing_batch_{i}.nc"
                cdo_obj.mergetime(input=batch_files, output=str(batch_output))
                intermediate_files.append(batch_output)

            # Combine intermediate results
            cdo_obj.mergetime(input=[str(f) for f in intermediate_files], output=str(merged_forcing_path))

            # Clean up intermediate files
            for f in intermediate_files:
                if f.exists():
                    f.unlink()

            self.logger.info("CDO merge successful")

        except (AttributeError, Exception) as e:
            self.logger.warning(f"CDO merge failed or CDO not available: {e}. Falling back to xarray...")
            try:
                # Fallback to xarray (more portable but slower for huge files)
                with xr.open_mfdataset(easymore_nc_files, combine='nested', concat_dim='time', data_vars='all') as ds:
                    ds.sortby('time').to_netcdf(merged_forcing_path)
                self.logger.info("Xarray merge successful")
            except Exception as xe:
                self.logger.error(f"Xarray merge also failed: {xe}")
                return None

        # Handle time shift and calendar
        if not merged_forcing_path.exists():
            return None

        with xr.open_dataset(merged_forcing_path) as forcing:
            forcing = forcing.convert_calendar('standard')
            if self.timeshift != 0:
                forcing['time'] = forcing['time'] + pd.Timedelta(hours=self.timeshift)

            tmp_path = merged_forcing_path.with_suffix('.nc.tmp')
            forcing.to_netcdf(tmp_path)

        os.replace(tmp_path, merged_forcing_path)
        return merged_forcing_path

    def _convert_to_daily_obs(self, merged_forcing_path: Path) -> None:
        """Convert hourly merged data to HYPE daily observation files."""
        def get_in_var(key):
            return self.forcing_units[key]['in_varname']

        # Get temperature units for conversion (HYPE expects Celsius)
        temp_units = self.forcing_units.get('temperature', {}).get('in_units', 'K')

        # TMAX
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMAXobs',
            stat='max',
            output_file_name_txt=self.output_path / 'TMAXobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # TMIN
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMINobs',
            stat='min',
            output_file_name_txt=self.output_path / 'TMINobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # Tobs (Mean)
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'Tobs',
            stat='mean',
            output_file_name_txt=self.output_path / 'Tobs.txt',
            unit_conversion=temp_units  # Convert K to C if needed
        )

        # Pobs (Sum)
        # Get precipitation units for conversion
        precip_units = self.forcing_units.get('precipitation', {}).get('in_units', 'mm/s')
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('precipitation'),
            'Pobs',
            stat='sum',
            output_file_name_txt=self.output_path / 'Pobs.txt',
            unit_conversion=precip_units  # Pass units for conversion
        )

    def _convert_hourly_to_daily(
        self,
        input_file_name: Path,
        variable_in: str,
        variable_out: str,
        var_time: str = 'time',
        var_id: str = 'hruId',
        stat: str = 'max',
        output_file_name_txt: Optional[Path] = None,
        unit_conversion: Optional[str] = None
    ) -> xr.Dataset:
        """Helper to resample hourly NetCDF to daily text file.

        Args:
            input_file_name: Path to merged forcing NetCDF
            variable_in: Input variable name
            variable_out: Output variable name (for logging)
            var_time: Time dimension name
            var_id: HRU/subbasin ID variable name
            stat: Aggregation statistic ('max', 'min', 'mean', 'sum')
            output_file_name_txt: Output text file path
            unit_conversion: Input units for conversion. If 'mm/s' or 'kg/m²/s' or 'kg m-2 s-1',
                applies conversion factor of 3600 (seconds per hour) for hourly data.
        """
        with xr.open_dataset(input_file_name) as ds:
            ds = ds.copy()

            # Apply unit conversion
            if unit_conversion:
                unit_lower = unit_conversion.lower()

                # Precipitation: convert from rate (per second) to amount (per hour)
                if unit_lower in ['mm/s', 'mm s-1', 'kg/m²/s', 'kg m-2 s-1', 'kg/m2/s']:
                    # Multiply by 3600 seconds/hour to convert rate to hourly amount
                    self.logger.info(f"Converting {variable_in} from {unit_conversion} to mm/hour (multiplying by 3600)")
                    ds[variable_in] = ds[variable_in] * 3600.0

                # Temperature: convert from Kelvin to Celsius
                elif unit_lower in ['k', 'kelvin']:
                    self.logger.info(f"Converting {variable_in} from Kelvin to Celsius (subtracting 273.15)")
                    ds[variable_in] = ds[variable_in] - 273.15

            # Get the mapping from hru dimension index to actual hruId values
            # This is needed because hruId is often a data variable, not a coordinate
            hru_id_mapping = None
            if var_id in ds.data_vars and var_id not in ds.coords:
                # hruId is a data variable - get the mapping from hru index to actual IDs
                hru_id_da = ds[var_id]
                # Find the dimension name for hruId (typically 'hru')
                hru_dim = hru_id_da.dims[0] if hru_id_da.dims else None
                if hru_dim:
                    # Handle multi-dimensional case (e.g., if hruId has time dimension)
                    # Take the first time slice if multiple dimensions exist
                    if hru_id_da.ndim > 1:
                        # Select first index along all dimensions except the hru dimension
                        sel_dict = {str(dim): 0 for dim in hru_id_da.dims if dim != hru_dim}
                        hru_id_values = hru_id_da.isel(**sel_dict).values.flatten()
                    else:
                        hru_id_values = hru_id_da.values.flatten()
                    # Convert to integer and create mapping
                    hru_id_values = hru_id_values.astype(int)
                    hru_id_mapping = {i: int(hru_id_values[i]) for i in range(len(hru_id_values))}
            elif var_id in ds.coords:
                # hruId is already a coordinate - cast to int
                ds.coords[var_id] = ds.coords[var_id].astype(int)

            # Ensure time index is sorted
            ds = ds.sortby('time')

            # Resample to daily
            if stat == 'max':
                ds_daily = ds.resample(time='D').max()
            elif stat == 'min':
                ds_daily = ds.resample(time='D').min()
            elif stat == 'mean':
                ds_daily = ds.resample(time='D').mean()
            elif stat == 'sum':
                ds_daily = ds.resample(time='D').sum()
            else:
                raise ValueError(f"Unsupported stat: {stat}")

            # Extract variable and convert to dataframe
            # Use to_series().unstack() to get time as index and IDs as columns
            series = ds_daily[variable_in].to_series()

            # Dynamically determine the ID level name
            actual_id_level = var_id
            if var_id not in series.index.names:
                for fallback in ['id', 'hru', 'subid']:
                    if fallback in series.index.names:
                        actual_id_level = fallback
                        break

            df = series.unstack(level=actual_id_level)

            # Map column indices to actual hruId values if we have the mapping
            if hru_id_mapping is not None:
                # Columns are currently hru dimension indices (0, 1, 2, ...)
                # Map them to actual hruId values
                df.columns = [hru_id_mapping.get(int(c), int(c)) for c in df.columns]
            else:
                # Ensure columns (subids) are integers
                df.columns = df.columns.astype(int)
                # Shift 0-based IDs if needed (legacy behavior for backwards compatibility)
                if 0 in df.columns:
                    df.columns = [c + 1 if c == 0 else c for c in df.columns]

            df.columns.name = None
            df.index.name = 'time'

            # Ensure time index is formatted as YYYY-MM-DD for HYPE
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

            if output_file_name_txt:
                # HYPE observation files: header is 'time' then subids
                # Separated by tabs
                df.to_csv(output_file_name_txt, sep='\t', na_rep='-9999.0', index=True, float_format='%.3f')

            return ds_daily
