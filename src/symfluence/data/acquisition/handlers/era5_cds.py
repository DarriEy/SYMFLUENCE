"""
ERA5 CDS Data Acquisition Handler for SYMFLUENCE.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from symfluence.core.constants import UnitConversion

@AcquisitionRegistry.register('ERA5_CDS')
class ERA5CDSAcquirer(BaseAcquisitionHandler):
    """
    ERA5 data acquisition handler using the Copernicus Climate Data Store (CDS) API.
    """

    def download(self, output_dir: Path) -> Path:
        """Download and process ERA5 data from CDS."""
        if not HAS_CDSAPI:
            raise ImportError(
                "cdsapi package is required for ERA5 CDS downloads. "
                "Install it with 'pip install cdsapi'."
            )

        # Initialize CDS client
        try:
            c = cdsapi.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            self.logger.error("Ensure ~/.cdsapirc exists or CDSAPI_URL/CDSAPI_KEY env vars are set.")
            raise

        self.logger.info(f"Downloading ERA5 data from CDS for {self.domain_name}...")

        # Build temporal parameters restricted to actual requested range
        dates = pd.date_range(self.start_date, self.end_date, freq='h')
        years = sorted(list(set([str(d.year) for d in dates])))
        months = sorted(list(set([f"{d.month:02d}" for d in dates])))
        days = sorted(list(set([f"{d.day:02d}" for d in dates])))
        times = sorted(list(set([f"{d.hour:02d}:00" for d in dates])))

        # ERA5 variables mapping (CDS names)
        # We need these variables for SUMMA: airtemp, airpres, windspd, spechum, pptrate, SWRadAtm, LWRadAtm
        variables = [
            '2m_temperature',
            'surface_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation',
            'surface_solar_radiation_downwards',
            'surface_thermal_radiation_downwards',
            '2m_dewpoint_temperature' # Needed for specific humidity
        ]

        # Bounding box for CDS (North, West, South, East)
        area = [
            self.bbox['lat_max'],
            self.bbox['lon_min'],
            self.bbox['lat_min'],
            self.bbox['lon_max']
        ]

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_file = output_dir / f"{self.domain_name}_era5_cds_temp.nc"

        request = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': variables,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': area,
        }

        # Retrieve data
        c.retrieve('reanalysis-era5-single-levels', request, str(temp_file))

        # Check if file exists and is not empty
        if not temp_file.exists():
            raise FileNotFoundError(f"Downloaded ERA5 file not found at {temp_file}")
        
        # Log file type using shell 'file' command if available
        import subprocess
        is_zip = False
        try:
            ftype = subprocess.check_output(['file', str(temp_file)], stderr=subprocess.STDOUT).decode()
            self.logger.info(f"Downloaded file type: {ftype.strip()}")
            if 'zip' in ftype.lower():
                is_zip = True
        except Exception:
            pass

        if is_zip:
            self.logger.info("Unzipping downloaded ERA5 data...")
            import zipfile
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                # Find the first .nc file in the zip
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_files:
                    raise RuntimeError("No NetCDF file found in the zipped CDS result")
                
                zip_ref.extract(nc_files[0], path=output_dir)
                extracted_file = output_dir / nc_files[0]
                
                # Replace temp_file with extracted file
                temp_file.unlink()
                extracted_file.rename(temp_file)

        file_size = temp_file.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Downloaded ERA5 file is empty: {temp_file}")
        
        self.logger.info(f"Downloaded ERA5 file size: {file_size / 1024**2:.2f} MB")

        # Process the downloaded file
        self.logger.info(f"Processing downloaded ERA5 data...")
        # Explicitly try netcdf4 engine
        try:
            ds = xr.open_dataset(temp_file, engine='netcdf4')
        except Exception as e:
            self.logger.warning(f"Failed to open with netcdf4 engine: {e}. Trying to guess engine.")
            ds = xr.open_dataset(temp_file)

        # Build output path
        final_f = output_dir / f"domain_{self.domain_name}_ERA5_CDS_{self.start_date.year}_{self.end_date.year}.nc"

        with ds:
            # 1. Coordinate and Dimension standardization
            if 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})

            if 'expver' in ds.dims:
                if ds.sizes['expver'] > 1:
                    ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
                else:
                    ds = ds.isel(expver=0)
            
            # Selection of first member if ensemble data returned
            if 'number' in ds.dims:
                self.logger.info("Ensemble data detected (dimension 'number'). Selecting first member.")
                ds = ds.isel(number=0)

            ds = ds.sortby('time')
            ds = ds.sel(time=slice(self.start_date, self.end_date))
            ds.load()
            
            self.logger.info(f"Available variables in downloaded file: {list(ds.variables.keys())}")

            # 2. Variable Renaming and Unit conversions
            # We create a new dictionary of variables to ensure clean assignment
            processed_vars = {}

            # Find precipitation
            v_tp = next((v for v in ds.variables if any(x in v.lower() for x in ['total_precip', 'tp'])), None)
            if v_tp:
                self.logger.info(f"Processing precipitation from {v_tp}")
                pptrate = (ds[v_tp] * 1000.0 / UnitConversion.SECONDS_PER_HOUR).astype('float32')
                pptrate.attrs = {'units': 'kg m-2 s-1', 'long_name': 'precipitation rate'}
                processed_vars['pptrate'] = pptrate
            else:
                self.logger.warning("Could not find precipitation variable in CDS output")

            # Shortwave radiation - ERA5 accumulated, needs time-differencing
            v_ssrd = next((v for v in ds.variables if any(x in v.lower() for x in ['solar_radiation_down', 'ssrd'])), None)
            if v_ssrd:
                self.logger.info(f"Processing shortwave from {v_ssrd}")
                val = ds[v_ssrd]

                # De-accumulate: take time difference then divide by timestep
                # ERA5 provides cumulative radiation that needs differencing
                dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')
                sw_diff = val.diff('time').where(val.diff('time') >= 0, 0)  # Handle resets
                sw_rad = (sw_diff / dt).clip(min=0).astype('float32')

                # Pad the first timestep (lost in diff) with the first valid value
                sw_rad = xr.concat([sw_rad.isel(time=0), sw_rad], dim='time')

                self.logger.debug(f"De-accumulated SW radiation: max={sw_rad.max().values:.1f} W/m²")
                sw_rad.attrs = {'units': 'W m-2', 'long_name': 'shortwave radiation',
                               'standard_name': 'surface_downwelling_shortwave_flux_in_air'}
                processed_vars['SWRadAtm'] = sw_rad
            else:
                self.logger.warning("Could not find shortwave radiation variable in CDS output")
            
            # Longwave radiation - ERA5 accumulated, needs time-differencing
            v_strd = next((v for v in ds.variables if any(x in v.lower() for x in ['thermal_radiation_down', 'strd'])), None)
            if v_strd:
                self.logger.info(f"Processing longwave from {v_strd}")
                val = ds[v_strd]

                # De-accumulate: take time difference then divide by timestep
                # ERA5 provides cumulative radiation that needs differencing
                if 'dt' not in locals():
                    dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')
                lw_diff = val.diff('time').where(val.diff('time') >= 0, 0)  # Handle resets
                lw_rad = (lw_diff / dt).clip(min=0).astype('float32')

                # Pad the first timestep (lost in diff) with the first valid value
                lw_rad = xr.concat([lw_rad.isel(time=0), lw_rad], dim='time')

                self.logger.debug(f"De-accumulated LW radiation: max={lw_rad.max().values:.1f} W/m²,  mean={lw_rad.mean().values:.1f} W/m²")

                # Validate reasonable range (typical LW: 150-450 W/m²)
                if lw_rad.max() < 100 or lw_rad.mean() < 50:
                    self.logger.warning(f"LW radiation values seem too low! Max={lw_rad.max().values:.1f}, Mean={lw_rad.mean().values:.1f} W/m². Expected 200-400 W/m².")
                else:
                    self.logger.info(f"✓ LW radiation looks good: mean={lw_rad.mean().values:.1f} W/m²")

                lw_rad.attrs = {'units': 'W m-2', 'long_name': 'longwave radiation',
                               'standard_name': 'surface_downwelling_longwave_flux_in_air'}
                processed_vars['LWRadAtm'] = lw_rad
            else:
                self.logger.warning("Could not find longwave radiation variable in CDS output")

            # Temperatures and Pressure
            rename_map = {
                't2m': 'airtemp', '2m_temperature': 'airtemp',
                'sp': 'airpres', 'surface_pressure': 'airpres',
                'd2m': 'dewpoint', '2m_dewpoint_temperature': 'dewpoint',
                'u10': 'wind_u', '10m_u_component_of_wind': 'wind_u',
                'v10': 'wind_v', '10m_v_component_of_wind': 'wind_v'
            }
            
            for src, target in rename_map.items():
                if src in ds.variables:
                    processed_vars[target] = ds[src].astype('float32')

            # 3. Derived Variables
            # Wind speed
            if 'wind_u' in processed_vars and 'wind_v' in processed_vars:
                processed_vars['windspd'] = np.sqrt(processed_vars['wind_u']**2 + processed_vars['wind_v']**2).astype('float32')
                processed_vars['windspd'].attrs = {'units': 'm s-1', 'long_name': 'wind speed'}

            # Specific humidity
            if 'dewpoint' in processed_vars and 'airpres' in processed_vars:
                Td_C = processed_vars['dewpoint'] - 273.15
                P = processed_vars['airpres']
                e = 611.2 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
                processed_vars['spechum'] = (0.622 * e / (P - 0.378 * e)).astype('float32')
                processed_vars['spechum'].attrs = {'units': 'kg kg-1', 'long_name': 'specific humidity'}

            # 4. Final Assembly
            final_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
            ds_final = xr.Dataset(
                data_vars={v: processed_vars[v] for v in final_vars if v in processed_vars},
                coords={c: ds.coords[c] for c in ['time', 'latitude', 'longitude'] if c in ds.coords},
                attrs=ds.attrs
            )

            # Ensure correct dimension order
            ds_final = ds_final.transpose('time', 'latitude', 'longitude', missing_dims='ignore')
            
            # Check for missing variables
            missing = set(final_vars) - set(ds_final.data_vars)
            if missing:
                self.logger.warning(f"ERA5 dataset missing variables: {missing}")
                self.logger.info(f"Variables present in ds_final: {list(ds_final.data_vars)}")

            ds_final.to_netcdf(final_f)

        # Cleanup
        if temp_file.exists():
            temp_file.unlink()

        self.logger.info(f"ERA5 CDS data saved to {final_f}")
        return final_f