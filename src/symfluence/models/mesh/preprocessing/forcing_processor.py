"""
MESH Forcing Processor

Handles forcing file preparation and splitting.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as NC4Dataset

from .config_defaults import MESHConfigDefaults


class MESHForcingProcessor:
    """
    Processes forcing files for MESH model.

    Handles:
    - Direct forcing preparation from basin-averaged data
    - Post-processing meshflow output for MESH compatibility
    - Creating split forcing files for distributed mode
    """

    def __init__(
        self,
        forcing_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize forcing processor.

        Args:
            forcing_dir: Directory for MESH files
            config: Configuration dictionary (meshflow config)
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def prepare_forcing_direct(self) -> None:
        """
        Prepare MESH forcing directly from basin-averaged data.

        This bypasses meshflow's CDO-based forcing prep which has issues
        with frequency inference on multi-file datasets.
        """
        forcing_files = sorted(glob.glob(self.config.get('forcing_files', '')))
        if not forcing_files:
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError("No forcing files found")

        self.logger.info(f"Loading {len(forcing_files)} forcing files")

        ds = xr.open_mfdataset(forcing_files, combine='by_coords', parallel=False)

        forcing_vars = self.config.get('forcing_vars', {})
        var_rename = {v: k for k, v in forcing_vars.items()}

        # Find spatial dimension
        self.config.get('hru_dim', 'hru')
        spatial_dim = None
        for dim in ['hru', 'subbasin', 'N', 'gru', 'GRU_ID']:
            if dim in ds.dims:
                spatial_dim = dim
                break

        if spatial_dim is None:
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError("Could not find spatial dimension")

        n_spatial = ds.dims[spatial_dim]
        self.logger.info(f"Found spatial dimension '{spatial_dim}' with {n_spatial} elements")

        forcing_path = self.forcing_dir / "MESH_forcing.nc"

        with NC4Dataset(forcing_path, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('time', None)
            ncfile.createDimension('subbasin', n_spatial)

            time_data = ds['time'].values
            var_time = ncfile.createVariable('time', 'f8', ('time',))
            var_time.long_name = 'time'
            var_time.standard_name = 'time'
            var_time.units = 'hours since 1900-01-01'
            var_time.calendar = 'gregorian'

            reference = pd.Timestamp('1900-01-01')
            time_hours = np.array([
                (pd.Timestamp(t) - reference).total_seconds() / 3600.0
                for t in time_data
            ])
            var_time[:] = time_hours

            var_n = ncfile.createVariable('subbasin', 'i4', ('subbasin',))
            var_n[:] = np.arange(1, n_spatial + 1)

            for src_var, standard_name in var_rename.items():
                if src_var in ds:
                    mesh_name = MESHConfigDefaults.MESH_VAR_NAMES.get(standard_name, src_var)
                    var_data = ds[src_var].values

                    if var_data.ndim == 2:
                        if var_data.shape[0] != len(time_data):
                            var_data = var_data.T

                    var = ncfile.createVariable(mesh_name, 'f4', ('time', 'subbasin'), fill_value=-9999.0)
                    var.long_name = standard_name.replace('_', ' ')
                    var.units = MESHConfigDefaults.get_var_units(mesh_name)
                    var[:] = var_data
                    self.logger.debug(f"Created {mesh_name} from {src_var}")

            ncfile.title = 'MESH Forcing Data'
            ncfile.Conventions = 'CF-1.6'
            ncfile.history = f'Created by SYMFLUENCE on {pd.Timestamp.now()}'

        self.logger.info(f"Created MESH forcing file: {forcing_path}")
        ds.close()

    def postprocess_meshflow_output(self) -> None:
        """Post-process meshflow output for MESH compatibility."""
        self._rename_subbasin_dimension()
        self._rename_forcing_variables()
        self.create_split_forcing_files()

    def _rename_subbasin_dimension(self) -> None:
        """Rename dimension to 'subbasin' in all NetCDF files."""
        for nc_file in [
            self.forcing_dir / "MESH_forcing.nc",
            self.forcing_dir / "MESH_drainage_database.nc"
        ]:
            if nc_file.exists():
                try:
                    with xr.open_dataset(nc_file) as ds:
                        rename_dict: Dict[Any, str] = {}
                        if 'N' in ds.dims:
                            rename_dict['N'] = 'subbasin'
                        elif 'subbasin' not in ds.dims and len(ds.dims) > 0:
                            # If no standard dimension found, find the one that isn't 'time' or 'landclass'
                            other_dims = [d for d in ds.dims if d not in ['time', 'landclass', 'land', 'NGRU']]
                            if other_dims:
                                rename_dict[other_dims[0]] = 'subbasin'

                        if rename_dict:
                            ds_renamed = ds.rename(rename_dict)
                            # Also rename variables that match the old dimension
                            for old_dim, new_dim in rename_dict.items():
                                if old_dim in ds_renamed.variables:
                                    ds_renamed = ds_renamed.rename({old_dim: new_dim})

                            temp_path = nc_file.with_suffix('.tmp.nc')
                            ds_renamed.to_netcdf(temp_path)
                            os.replace(temp_path, nc_file)
                            self.logger.info(f"Renamed dimension to 'subbasin' in {nc_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to rename dimension in {nc_file.name}: {e}")

    def _rename_forcing_variables(self) -> None:
        """Rename forcing variables for MESH 1.5 compatibility."""
        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        if not forcing_nc.exists():
            return

        try:
            with xr.open_dataset(forcing_nc) as ds:
                rename_map = {
                    'airpres': 'PRES', 'spechum': 'QA', 'airtemp': 'TA',
                    'windspd': 'UV', 'pptrate': 'PRE', 'SWRadAtm': 'FSIN',
                    'LWRadAtm': 'FLIN', 'air_pressure': 'PRES',
                    'specific_humidity': 'QA', 'air_temperature': 'TA',
                    'wind_speed': 'UV', 'precipitation': 'PRE',
                    'shortwave_radiation': 'FSIN', 'longwave_radiation': 'FLIN',
                }

                existing_rename = {k: v for k, v in rename_map.items() if k in ds.variables}
                if existing_rename:
                    ds_renamed = ds.rename(existing_rename)

                    if 'time' in ds_renamed.dims and 'subbasin' in ds_renamed.dims:
                        ds_renamed = ds_renamed.transpose('time', 'subbasin', ...)

                    temp_path = forcing_nc.with_suffix('.tmp.nc')
                    ds_renamed.to_netcdf(temp_path, unlimited_dims=['time'])
                    os.replace(temp_path, forcing_nc)
                    self.logger.info("Renamed forcing variables for MESH 1.5")

        except Exception as e:
            self.logger.warning(f"Failed to rename forcing variables: {e}")

    def create_split_forcing_files(self) -> None:
        """Create individual forcing files per variable for distributed mode."""
        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        if not forcing_nc.exists():
            self.logger.warning("MESH_forcing.nc not found")
            return

        try:
            with xr.open_dataset(forcing_nc) as ds:
                n_dim = 'N' if 'N' in ds.dims else 'subbasin' if 'subbasin' in ds.dims else None
                if not n_dim:
                    self.logger.warning("No spatial dimension found")
                    return

                n_size = ds.dims[n_dim]
                time_data = ds['time'].values if 'time' in ds else None

                # Get coordinate variables once
                lat_data = ds['lat'].values if 'lat' in ds else None
                lon_data = ds['lon'].values if 'lon' in ds else None
                has_crs = 'crs' in ds

                for mesh_var, filename in MESHConfigDefaults.VAR_TO_FILE.items():
                    if mesh_var in ds:
                        var_path = self.forcing_dir / filename
                        var_data = ds[mesh_var].values

                        with NC4Dataset(var_path, 'w', format='NETCDF4') as ncfile:
                            ncfile.createDimension('time', None)
                            ncfile.createDimension('subbasin', n_size)

                            # Add CRS if available
                            if has_crs:
                                var_crs = ncfile.createVariable('crs', 'i4')
                                for attr in ds['crs'].attrs:
                                    var_crs.setncattr(attr, ds['crs'].attrs[attr])

                            # Add spatial coordinates
                            if lat_data is not None:
                                var_lat = ncfile.createVariable('lat', 'f8', ('subbasin',))
                                var_lat.standard_name = 'latitude'
                                var_lat.long_name = 'latitude'
                                var_lat.units = 'degrees_north'
                                var_lat.axis = 'Y'
                                var_lat[:] = lat_data

                            if lon_data is not None:
                                var_lon = ncfile.createVariable('lon', 'f8', ('subbasin',))
                                var_lon.standard_name = 'longitude'
                                var_lon.long_name = 'longitude'
                                var_lon.units = 'degrees_east'
                                var_lon.axis = 'X'
                                var_lon[:] = lon_data

                            # Add forcing variable
                            var = ncfile.createVariable(
                                mesh_var, 'f4', ('time', 'subbasin'), fill_value=-9999.0
                            )
                            var.long_name = MESHConfigDefaults.get_var_long_name(mesh_var)
                            var.units = MESHConfigDefaults.get_var_units(mesh_var)
                            if has_crs:
                                var.grid_mapping = 'crs'
                            if lat_data is not None and lon_data is not None:
                                var.coordinates = 'lat lon'
                            var[:] = var_data

                            # Add time coordinate
                            if time_data is not None:
                                var_time = ncfile.createVariable('time', 'f8', ('time',))
                                var_time.long_name = 'time'
                                var_time.standard_name = 'time'
                                var_time.units = 'hours since 1900-01-01'
                                var_time.calendar = 'gregorian'

                                reference = pd.Timestamp('1900-01-01')
                                time_hours = [
                                    (pd.Timestamp(t) - reference).total_seconds() / 3600.0
                                    for t in time_data
                                ]
                                var_time[:] = time_hours

                            # Add subbasin coordinate
                            var_n = ncfile.createVariable('subbasin', 'i4', ('subbasin',))
                            var_n[:] = np.arange(1, n_size + 1)

                        self.logger.debug(f"Created {filename}")

                self.logger.info("Created split forcing files for MESH distributed mode")

        except Exception as e:
            self.logger.warning(f"Failed to create split forcing files: {e}")
