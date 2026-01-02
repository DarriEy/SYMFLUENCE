"""
Mixin for building xarray datasets with standard coordinate patterns.

Provides utilities for creating forcing datasets with consistent coordinate
structures, attributes, and variable handling across model preprocessors.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import xarray as xr


class DatasetBuilderMixin:
    """
    Mixin for building xarray datasets with standard patterns.

    Provides methods for creating coordinate systems, adding variables with
    consistent attributes, and handling common dataset operations used across
    model preprocessors.
    """

    def create_forcing_coords(
        self,
        longitude: Union[float, np.ndarray, List[float]],
        latitude: Union[float, np.ndarray, List[float]],
        time_index: pd.DatetimeIndex,
        time_unit: str = 'D',
        reference_date: str = '1970-01-01'
    ) -> Dict[str, Tuple[str, Any]]:
        """
        Create standard coordinate dictionary for forcing datasets.

        Converts time to numeric values and wraps scalars in lists for
        consistent xarray coordinate structure.

        Args:
            longitude: Longitude value(s) - scalar or array
            latitude: Latitude value(s) - scalar or array
            time_index: Pandas DatetimeIndex for time coordinate
            time_unit: Time unit for numeric conversion ('h' for hours, 'D' for days)
            reference_date: Reference date for time conversion

        Returns:
            Dict suitable for xr.Dataset(coords=...)
        """
        # Convert time to numeric values
        ref_ts = pd.Timestamp(reference_date)
        if time_unit == 'h':
            time_numeric = ((time_index - ref_ts).total_seconds() / 3600).values
        else:
            time_numeric = (time_index - ref_ts).days.values

        # Ensure arrays for coordinates
        lon_arr = [longitude] if np.isscalar(longitude) else np.asarray(longitude)
        lat_arr = [latitude] if np.isscalar(latitude) else np.asarray(latitude)

        return {
            'longitude': ('longitude', lon_arr),
            'latitude': ('latitude', lat_arr),
            'time': ('time', time_numeric)
        }

    def create_elevation_band_coords(
        self,
        longitude: Union[float, np.ndarray, List[float]],
        latitude: Union[float, np.ndarray, List[float]],
        n_bands: int = 1
    ) -> Dict[str, Tuple[str, Any]]:
        """
        Create coordinate dictionary for elevation band datasets.

        Args:
            longitude: Longitude value(s)
            latitude: Latitude value(s)
            n_bands: Number of elevation bands

        Returns:
            Dict suitable for xr.Dataset(coords=...)
        """
        lon_arr = [longitude] if np.isscalar(longitude) else np.asarray(longitude)
        lat_arr = [latitude] if np.isscalar(latitude) else np.asarray(latitude)

        return {
            'longitude': ('longitude', lon_arr),
            'latitude': ('latitude', lat_arr),
            'elevation_band': ('elevation_band', list(range(1, n_bands + 1)))
        }

    def add_coord_attrs(
        self,
        ds: xr.Dataset,
        time_units: Optional[str] = None,
        lon_units: str = 'degreesE',
        lat_units: str = 'degreesN'
    ) -> xr.Dataset:
        """
        Add standard attributes to coordinate variables.

        Args:
            ds: Dataset to modify (modified in place)
            time_units: Time units string (e.g., 'days since 1970-01-01')
            lon_units: Longitude units
            lat_units: Latitude units

        Returns:
            Modified dataset (same object, for chaining)
        """
        if 'longitude' in ds.coords:
            ds.longitude.attrs = {'units': lon_units, 'long_name': 'longitude'}
        if 'latitude' in ds.coords:
            ds.latitude.attrs = {'units': lat_units, 'long_name': 'latitude'}
        if 'time' in ds.coords and time_units:
            ds.time.attrs = {'units': time_units, 'long_name': 'time'}
        if 'elevation_band' in ds.coords:
            ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}

        return ds

    def add_variable(
        self,
        ds: xr.Dataset,
        name: str,
        data: np.ndarray,
        dims: Tuple[str, ...],
        units: str,
        long_name: str,
        fill_value: float = -9999.0,
        dtype: str = 'float32'
    ) -> xr.Dataset:
        """
        Add a variable to the dataset with standard attributes and NaN handling.

        Args:
            ds: Dataset to modify (modified in place)
            name: Variable name
            data: Data array
            dims: Dimension names tuple
            units: Units string for attributes
            long_name: Long name for attributes
            fill_value: Value to use for NaN replacement
            dtype: Data type for storage

        Returns:
            Modified dataset (same object, for chaining)
        """
        # Handle NaN values
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=fill_value)

        # Ensure correct dtype
        data = data.astype(dtype)

        ds[name] = xr.DataArray(
            data,
            dims=dims,
            coords=ds.coords,
            attrs={
                'units': units,
                'long_name': long_name
            }
        )

        return ds

    def get_standard_encoding(
        self,
        ds: xr.Dataset,
        fill_value: float = -9999.0,
        dtype: str = 'float32'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get standard encoding dictionary for netCDF output.

        Args:
            ds: Dataset to create encoding for
            fill_value: Fill value for all variables
            dtype: Data type for all variables

        Returns:
            Encoding dict suitable for ds.to_netcdf(encoding=...)
        """
        return {
            var: {'_FillValue': fill_value, 'dtype': dtype}
            for var in ds.data_vars
        }

    def reshape_to_spatial_dims(
        self,
        data: np.ndarray,
        spatial_shape: Tuple[int, ...],
        time_length: int
    ) -> np.ndarray:
        """
        Reshape 1D time series data to match spatial structure.

        Args:
            data: 1D array of shape (time,) or 2D of shape (time, spatial)
            spatial_shape: Target spatial shape (e.g., (1, 1) for lumped)
            time_length: Expected time dimension length

        Returns:
            Reshaped array of shape (time, *spatial_shape)
        """
        if len(data.shape) == 1:
            # Time series only - tile across spatial dimensions
            return np.tile(
                data[:, np.newaxis, np.newaxis],
                (1,) + tuple(1 if s == 1 else s for s in spatial_shape)
            )
        elif len(data.shape) == 2:
            # Already has one spatial dimension
            if spatial_shape[0] > spatial_shape[1]:
                return data[:, :, np.newaxis]
            else:
                return data[:, np.newaxis, :]
        else:
            return data

    def create_lumped_forcing_dataset(
        self,
        longitude: float,
        latitude: float,
        time_index: pd.DatetimeIndex,
        time_unit: str = 'D',
        time_units_str: str = 'days since 1970-01-01'
    ) -> xr.Dataset:
        """
        Create an empty forcing dataset for lumped mode.

        Convenience method combining coordinate creation and attribute setup.

        Args:
            longitude: Catchment centroid longitude
            latitude: Catchment centroid latitude
            time_index: Time coordinate
            time_unit: Unit for time conversion ('D' or 'h')
            time_units_str: NetCDF time units string

        Returns:
            Empty xr.Dataset with coordinates and attributes set
        """
        coords = self.create_forcing_coords(
            longitude=longitude,
            latitude=latitude,
            time_index=time_index,
            time_unit=time_unit
        )
        ds = xr.Dataset(coords=coords)
        return self.add_coord_attrs(ds, time_units=time_units_str)
