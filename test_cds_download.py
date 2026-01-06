#!/usr/bin/env python3
"""
Test CDS download with proper de-accumulation for one month.
This will verify the fix works before doing a full download.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_cds_download():
    """Download one month from CDS and verify LW radiation."""

    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi")
        return

    # Test parameters
    test_year = '2015'
    test_month = '01'
    test_area = [47.0, -122.0, 46.5, -121.5]  # Paradise SNOTEL area
    output_dir = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/test_cds')
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_file = output_dir / 'test_era5_cds_temp.nc'
    final_file = output_dir / 'test_era5_cds_processed.nc'

    print("="*70)
    print("CDS TEST DOWNLOAD - January 2015")
    print("="*70)
    print(f"Output: {output_dir}")
    print()

    # Download from CDS
    print("Step 1: Downloading raw data from CDS...")
    c = cdsapi.Client()

    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            '2m_temperature',
            'surface_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation',
            'surface_solar_radiation_downwards',
            'surface_thermal_radiation_downwards',
            '2m_dewpoint_temperature'
        ],
        'year': test_year,
        'month': test_month,
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': test_area,
    }

    c.retrieve('reanalysis-era5-single-levels', request, str(temp_file))
    print(f"✓ Downloaded to {temp_file}")

    # Check raw file
    print("\nStep 2: Checking raw CDS data...")
    with xr.open_dataset(temp_file) as ds_raw:
        print(f"  Variables: {list(ds_raw.data_vars.keys())}")

        # Find LW radiation variable
        v_strd = next((v for v in ds_raw.variables if 'thermal' in v.lower() or 'strd' in v.lower()), None)
        if v_strd:
            raw_lw = ds_raw[v_strd].values.flatten()
            print(f"\n  Raw {v_strd}:")
            print(f"    Min: {np.nanmin(raw_lw):.2f}")
            print(f"    Max: {np.nanmax(raw_lw):.2f}")
            print(f"    Mean: {np.nanmean(raw_lw):.2f}")
            print(f"    Units: {ds_raw[v_strd].attrs.get('units', 'unknown')}")

    # Process with de-accumulation
    print("\nStep 3: Processing with proper de-accumulation...")

    with xr.open_dataset(temp_file) as ds:
        # Handle dimension naming
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        ds = ds.sortby('time')

        processed_vars = {}

        # Find and process LW radiation
        v_strd = next((v for v in ds.variables if 'thermal' in v.lower() or 'strd' in v.lower()), None)
        if v_strd:
            print(f"  Processing {v_strd}...")
            val = ds[v_strd]

            # De-accumulate: take time difference then divide by timestep
            dt = (ds['time'].diff('time') / np.timedelta64(1, 's')).astype('float32')
            lw_diff = val.diff('time').where(val.diff('time') >= 0, 0)
            lw_rad = (lw_diff / dt).clip(min=0).astype('float32')

            # Pad first timestep
            lw_rad = xr.concat([lw_rad.isel(time=0), lw_rad], dim='time')

            lw_rad.attrs = {
                'units': 'W m-2',
                'long_name': 'longwave radiation',
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            }
            processed_vars['LWRadAtm'] = lw_rad

            # Check results
            lw_values = lw_rad.values.flatten()
            lw_mean = np.nanmean(lw_values)
            print(f"\n  Processed LWRadAtm:")
            print(f"    Min: {np.nanmin(lw_values):.2f} W/m²")
            print(f"    Max: {np.nanmax(lw_values):.2f} W/m²")
            print(f"    Mean: {lw_mean:.2f} W/m²")

            print("\n" + "="*70)
            if 200 <= lw_mean <= 450:
                print("✅✅✅ SUCCESS! De-accumulation fix works! ✅✅✅")
                print("The CDS pathway is now FIXED.")
                print("\nNext: Clean old data and re-run full download with CDS")
            else:
                print("❌ Still wrong - need more investigation")
            print("="*70)

        # Save processed file
        ds_out = xr.Dataset(processed_vars)
        ds_out.to_netcdf(final_file)
        print(f"\n✓ Saved processed test file: {final_file}")

if __name__ == '__main__':
    test_cds_download()
