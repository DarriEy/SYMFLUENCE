"""
Forcing data processing utilities for HYPE model.

Handles merging of forcing data from multiple NetCDF files and conversion
to HYPE-compatible daily observation formats.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, Any, Optional

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
            
            files_split = np.array_split(easymore_nc_files, batch_size)
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
                with xr.open_mfdataset(easymore_nc_files, combine='nested', concat_dim='time') as ds:
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

        # TMAX
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMAXobs',
            stat='max',
            output_file_name_txt=self.output_path / 'TMAXobs.txt'
        )

        # TMIN
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMINobs',
            stat='min',
            output_file_name_txt=self.output_path / 'TMINobs.txt'
        )

        # Tobs (Mean)
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'Tobs',
            stat='mean',
            output_file_name_txt=self.output_path / 'Tobs.txt'
        )

        # Pobs (Sum)
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('precipitation'),
            'Pobs',
            stat='sum',
            output_file_name_txt=self.output_path / 'Pobs.txt'
        )

    def _convert_hourly_to_daily(
        self,
        input_file_name: Path,
        variable_in: str,
        variable_out: str,
        var_time: str = 'time',
        var_id: str = 'hruId',
        stat: str = 'max',
        output_file_name_txt: Optional[Path] = None
    ) -> xr.Dataset:
        """Helper to resample hourly NetCDF to daily text file."""
        with xr.open_dataset(input_file_name) as ds:
            ds = ds.copy()
            
            # Cast ID to integer
            if var_id in ds.coords:
                ds.coords[var_id] = ds.coords[var_id].astype(int)
            elif var_id in ds.data_vars:
                ds[var_id] = ds[var_id].astype(int)

            # Keep only required variables
            variables_to_keep = [variable_in, var_time]
            if var_id is not None:
                variables_to_keep.append(var_id)
            
            # Ensure monotonic time for resampling
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

            # Rename variable
            if variable_in in ds_daily:
                ds_daily = ds_daily.rename({variable_in: variable_out})

            # Transpose if needed
            if var_time in ds_daily[variable_out].dims and var_id in ds_daily[variable_out].dims:
                ds_daily[variable_out] = ds_daily[variable_out].transpose(var_id, var_time)

            # Convert to dataframe and unstack
            df = ds_daily[variable_out].to_dataframe()
            if var_id in df.index.names:
                df = df.unstack(level=var_id)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1)
                
            df.columns.name = None
            df.index.name = var_time
            
            if output_file_name_txt:
                df.to_csv(output_file_name_txt, sep='\t', na_rep='', index_label='time', float_format='%.3f')
            
            return ds_daily
