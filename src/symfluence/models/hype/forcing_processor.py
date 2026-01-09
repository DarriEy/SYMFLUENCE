"""
Forcing data processing utilities for HYPE model.

This module provides the HYPEForcingProcessor class for converting basin-averaged
forcing data (typically hourly NetCDF files from EASYMORE remapping) into HYPE's
daily observation file format.

HYPE requires forcing data as tab-separated text files with specific naming:
- **Pobs.txt**: Daily precipitation (mm/day)
- **Tobs.txt**: Daily mean temperature (°C)
- **TMAXobs.txt**: Daily maximum temperature (°C)
- **TMINobs.txt**: Daily minimum temperature (°C)

The processor handles:
- Merging multiple NetCDF files (using CDO or xarray fallback)
- Time zone correction via hour offset
- Unit conversion (e.g., K→°C, kg/m²/s→mm/day)
- Resampling from hourly to daily resolution

Example usage:
    >>> from symfluence.models.hype import HYPEForcingProcessor
    >>> processor = HYPEForcingProcessor(
    ...     config=config_dict,
    ...     logger=logger,
    ...     forcing_input_dir=Path('/project/forcing/basin_avg'),
    ...     output_path=Path('/project/settings/HYPE'),
    ...     cache_path=Path('/project/cache'),
    ...     timeshift=-6,  # UTC to local time
    ...     forcing_units=forcing_units_dict
    ... )
    >>> processor.process_forcing()
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cdo
import numpy as np
import pandas as pd
import pint
import xarray as xr

from ..utilities import BaseForcingProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Module-level unit registry for conversions
ureg = pint.UnitRegistry()


class HYPEForcingProcessor(BaseForcingProcessor):
    """
    Processor for converting forcing data to HYPE observation format.

    This class handles the complete workflow of converting hourly, gridded or
    basin-averaged forcing data into HYPE's daily observation file format.
    It inherits from BaseForcingProcessor and implements HYPE-specific logic.

    The processing workflow:
        1. Merge individual NetCDF files (one per timestep) into a single file
        2. Apply time zone correction if needed
        3. Resample from hourly to daily (using appropriate statistics)
        4. Convert units to HYPE requirements
        5. Write tab-separated observation files

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        forcing_input_dir: Path to input NetCDF files.
        output_path: Path for output observation files.
        cache_path: Path for temporary files during processing.
        timeshift: Hours to shift timestamps (for time zone correction).
        forcing_units: Dictionary mapping variable names to unit specifications.

    Note:
        The processor attempts to use CDO for merging (faster for large datasets)
        but falls back to xarray if CDO is not available or fails.
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | Any,
        forcing_input_dir: Path | str,
        output_path: Path | str,
        cache_path: Path | str,
        timeshift: int = 0,
        forcing_units: dict[str, dict[str, str]] | None = None
    ) -> None:
        """
        Initialize the HYPE forcing processor.

        Args:
            config: Configuration dictionary containing experiment settings.
            logger: Logger instance for status and warning messages.
            forcing_input_dir: Path to directory containing input basin-averaged
                NetCDF files (typically output from EASYMORE remapping).
            output_path: Path to HYPE settings directory where observation
                files (Pobs.txt, Tobs.txt, etc.) will be written.
            cache_path: Path for temporary files during merge operations.
                Intermediate files are cleaned up after processing.
            timeshift: Hours to add to timestamps for time zone correction.
                Positive values shift forward, negative values shift backward.
                Example: -6 converts UTC to Mountain Standard Time.
            forcing_units: Dictionary mapping forcing variables to their unit
                specifications. Expected structure::

                    {
                        'temperature': {
                            'in_varname': 'airtemp',
                            'in_units': 'K',
                            'out_units': 'degC'
                        },
                        'precipitation': {
                            'in_varname': 'pptrate',
                            'in_units': 'kg m^-2 s^-1',
                            'out_units': 'mm/day'
                        }
                    }
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

            for i in range(batch_size):
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
        
        def get_units(key):
            return self.forcing_units[key].get('in_units'), self.forcing_units[key].get('out_units')

        # TMAX
        in_u, out_u = get_units('temperature')
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMAXobs',
            stat='max',
            output_file_name_txt=self.output_path / 'TMAXobs.txt',
            in_units=in_u,
            out_units=out_u
        )

        # TMIN
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'TMINobs',
            stat='min',
            output_file_name_txt=self.output_path / 'TMINobs.txt',
            in_units=in_u,
            out_units=out_u
        )

        # Tobs (Mean)
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('temperature'),
            'Tobs',
            stat='mean',
            output_file_name_txt=self.output_path / 'Tobs.txt',
            in_units=in_u,
            out_units=out_u
        )

        # Pobs (Sum -> Mean, because we convert to mm/day rate first)
        in_u_p, out_u_p = get_units('precipitation')
        self._convert_hourly_to_daily(
            merged_forcing_path,
            get_in_var('precipitation'),
            'Pobs',
            stat='mean', # Changed from sum to mean because we convert to rate (mm/day) first
            output_file_name_txt=self.output_path / 'Pobs.txt',
            in_units=in_u_p,
            out_units=out_u_p
        )

    def _normalize_units(self, unit_str: str) -> str:
        """Normalize unit strings for Pint compatibility."""
        if not unit_str:
            return unit_str
        import re
        norm = unit_str.strip()
        # Handle 'X-N' -> 'X^-N' (e.g. m-2 -> m^-2)
        norm = re.sub(r'([a-zA-Z_]\w*)-(\d+)', r'\1^-\2', norm)
        # Standardize operators
        norm = norm.replace('/', ' / ').replace('*', ' * ')
        norm = ' '.join(norm.split())
        return norm.replace('/ /', '/')

    def _convert_hourly_to_daily(
        self,
        input_file_name: Path,
        variable_in: str,
        variable_out: str,
        var_time: str = 'time',
        var_id: str = 'hruId',
        stat: str = 'max',
        output_file_name_txt: Optional[Path] = None,
        in_units: Optional[str] = None,
        out_units: Optional[str] = None
    ) -> xr.Dataset:
        """Helper to resample hourly NetCDF to daily text file."""
        with xr.open_dataset(input_file_name) as ds:
            ds = ds.copy()
            
            # Robustly handle hruId: ensure it's a coordinate of the data variables
            actual_id_level = var_id
            if var_id in ds.data_vars and var_id not in ds.coords:
                ds = ds.set_coords(var_id)
            
            # Cast ID to integer
            if var_id in ds.coords:
                ds.coords[var_id] = ds.coords[var_id].astype(int)
            elif var_id in ds.data_vars:
                ds[var_id] = ds[var_id].astype(int)

            # If hruId exists as a coordinate but not a dimension, and 'hru' is the dimension,
            # we need to make sure it's used for unstacking later.
            if 'hruId' in ds.coords and 'hru' in ds.dims:
                actual_id_level = 'hruId'

            # Keep only required variables
            variables_to_keep = [variable_in, var_time]
            if var_id is not None:
                variables_to_keep.append(var_id)
            
            # Apply Unit Conversion
            if in_units and out_units and in_units != out_units:
                try:
                    # Normalize units first
                    norm_in = self._normalize_units(in_units)
                    norm_out = self._normalize_units(out_units)
                    
                    # Special case for precipitation (mass flux -> depth rate)
                    # 1 kg/m2/s = 86400 mm/day (assuming water density = 1000 kg/m3)
                    is_precip_mass_to_depth = (
                        ('kg' in norm_in and 'm^-2' in norm_in and 's^-1' in norm_in) and
                        ('mm' in norm_out and 'day' in norm_out)
                    )
                    
                    if is_precip_mass_to_depth:
                        a = 86400.0
                        b = 0.0
                    else:
                        # Deduce linear coefficients y = ax + b using Pint
                        val0 = 0.0
                        q0 = ureg.Quantity(val0, norm_in)
                        res0 = q0.to(norm_out).magnitude
                        b = res0
                        
                        val1 = 100.0
                        q1 = ureg.Quantity(val1, norm_in)
                        res1 = q1.to(norm_out).magnitude
                        a = (res1 - res0) / val1
                    
                    ds[variable_in] = ds[variable_in] * a + b
                    # self.logger.debug(f"Converted {variable_in} from {in_units} to {out_units} (a={a}, b={b})")
                except Exception as e:
                    self.logger.warning(f"Unit conversion failed for {variable_in} ({in_units}->{out_units}): {e}")

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

            # Extract variable
            da = ds_daily[variable_in]
            
            # Use actual HRU IDs for the dimension coordinates to ensure headers match GeoData
            if 'hruId' in ds.coords and 'hru' in da.dims:
                da = da.assign_coords(hru=ds.coords['hruId'].values.astype(int))
                actual_id_level = 'hru'
            elif 'hru' in da.dims:
                actual_id_level = 'hru'
            else:
                actual_id_level = var_id

            # Ensure no singleton spatial dimensions
            if 'longitude' in da.dims and da.sizes['longitude'] == 1:
                da = da.squeeze('longitude')
            if 'latitude' in da.dims and da.sizes['latitude'] == 1:
                da = da.squeeze('latitude')
                
            # Convert to dataframe and unstack
            series = da.to_series()
            
            # Dynamically determine the ID level name
            if actual_id_level not in series.index.names:
                for fallback in ['hruId', 'hru', 'id', 'subid']:
                    if fallback in series.index.names:
                        actual_id_level = fallback
                        break
            
            df = series.unstack(level=actual_id_level)
            
            # Ensure columns (subids) are integers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
                
            df.columns = [int(float(c)) for c in df.columns]
            
            # HYPE subbasin IDs must start from 1. If 0 is present, shift ALL IDs.
            # Only do this if it's strictly necessary (min ID is 0)
            if min(df.columns) == 0:
                df.columns = [c + 1 for c in df.columns]
            
            df.columns.name = None
            df.index.name = 'time'
            
            # Ensure time index is formatted as YYYY-MM-DD for HYPE
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            
            if output_file_name_txt:
                # HYPE observation files: header is 'time' then subids
                # Separated by tabs
                df.to_csv(output_file_name_txt, sep='\t', na_rep='-9999.0', index=True, float_format='%.3f')
            
            return ds_daily
