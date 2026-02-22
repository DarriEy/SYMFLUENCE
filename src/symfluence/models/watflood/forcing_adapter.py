"""
WATFLOOD Forcing Adapter.

Converts CFIF (CF-Intermediate Format) forcing data to WATFLOOD .met format.
WATFLOOD only requires precipitation and temperature forcing (simplified
energy balance).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class WATFLOODForcingAdapter:
    """Convert CFIF forcing data to WATFLOOD .met format.

    WATFLOOD uses a simplified energy balance requiring only:
    - Precipitation (P) in mm
    - Temperature (T) in degrees Celsius

    The .met format is a columnar text file with header metadata
    followed by timestep data rows.
    """

    # CFIF -> WATFLOOD variable mapping
    VARIABLE_MAP = {
        'precipitation_flux': 'precip',      # kg/m2/s -> mm/h
        'air_temperature': 'temperature',     # K -> degC
    }

    # Required variables (P and T only)
    REQUIRED_VARIABLES = ['precipitation_flux', 'air_temperature']

    def __init__(self, config_dict: dict, logger_instance=None):
        self.config_dict = config_dict
        self.logger = logger_instance or logger

    def convert_forcing(
        self,
        cfif_dir: Path,
        output_dir: Path,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Convert CFIF forcing data to WATFLOOD .met format.

        Args:
            cfif_dir: Directory containing CFIF NetCDF files
            output_dir: Directory for WATFLOOD .met files
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if conversion succeeded
        """
        try:
            import xarray as xr
            import numpy as np

            output_dir.mkdir(parents=True, exist_ok=True)

            # Find CFIF files
            nc_files = sorted(cfif_dir.glob('*.nc'))
            if not nc_files:
                self.logger.error(f"No CFIF files found in {cfif_dir}")
                return False

            ds = xr.open_mfdataset(nc_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')

            # Extract and convert variables
            precip = None
            temp = None

            for var in ds.data_vars:
                if 'precip' in var.lower() or 'pr' == var.lower():
                    precip = ds[var].values
                    # Convert kg/m2/s -> mm/h (x 3600)
                    precip = precip * 3600.0
                elif 'temp' in var.lower() or 'tas' == var.lower():
                    temp = ds[var].values
                    # Convert K -> degC if needed
                    if temp.mean() > 100:
                        temp = temp - 273.15

            if precip is None or temp is None:
                self.logger.error("Missing required variables (precip, temp)")
                return False

            times = ds['time'].values

            # Write .met file
            met_path = output_dir / 'forcing.met'
            with open(met_path, 'w') as f:
                f.write(":FileType met  ASCII  WATFLOOD\n")
                f.write(":SourceFile SYMFLUENCE CFIF conversion\n")
                f.write(f":StartDate {start_date}\n")
                f.write(f":EndDate {end_date}\n")
                f.write(":ColumnMetaData\n")
                f.write(":  precip(mm/h)  temp(C)\n")
                f.write(":EndColumnMetaData\n")

                for i, t in enumerate(times):
                    ts = pd.Timestamp(t)
                    p_val = float(np.mean(precip[i])) if precip.ndim > 1 else float(precip[i])
                    t_val = float(np.mean(temp[i])) if temp.ndim > 1 else float(temp[i])
                    f.write(f"{ts.year:4d} {ts.month:2d} {ts.day:2d} {ts.hour:2d} "
                            f"{p_val:10.4f} {t_val:8.3f}\n")

            ds.close()
            self.logger.info(f"Wrote WATFLOOD .met file: {met_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error converting forcing: {e}")
            return False
