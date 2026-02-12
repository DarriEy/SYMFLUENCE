"""
Dataset Alignment Manager

Provides utilities for aligning multiple datasets to a common time dimension.
Consolidates time truncation/padding logic that was previously duplicated
across FUSE, NGEN, GR, and SUMMA preprocessors.

This module handles:
- Finding overlapping time periods across multiple datasets
- Truncating datasets to common time ranges
- Padding datasets with fill values or last valid values
- Reindexing datasets to common time coordinates
"""

import logging
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)


class DatasetAlignmentManager:
    """
    Manages alignment of multiple datasets to common time dimensions.

    This class handles the common pattern of:
    1. Finding the overlapping time period across multiple datasets
    2. Aligning all datasets to that common period
    3. Handling mismatches through truncation or padding

    Attributes:
        logger: Logger instance for diagnostics
        fill_value: Value to use when padding (default: -9999.0)
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        fill_value: float = -9999.0
    ):
        """
        Initialize the dataset alignment manager.

        Args:
            logger: Logger instance (creates default if None)
            fill_value: Value to use when padding is needed
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.fill_value = fill_value

    def find_common_time_period(
        self,
        datasets: List[xr.Dataset],
        time_var: str = 'time'
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Find the overlapping time period across multiple datasets.

        Args:
            datasets: List of xarray Datasets with time coordinates
            time_var: Name of time coordinate variable

        Returns:
            Tuple of (start_time, end_time) representing the overlap

        Raises:
            ValueError: If no overlapping time period exists
        """
        if not datasets:
            raise ValueError("No datasets provided")

        start_times = []
        end_times = []

        for i, ds in enumerate(datasets):
            if ds is None:
                continue

            if time_var not in ds.coords and time_var not in ds.dims:
                self.logger.warning(f"Dataset {i} has no '{time_var}' coordinate")
                continue

            times = self._get_datetime_index(ds[time_var])
            start_times.append(times.min())
            end_times.append(times.max())

        if not start_times:
            raise ValueError("No valid time coordinates found in any dataset")

        start_time = max(start_times)
        end_time = min(end_times)

        if start_time >= end_time:
            raise ValueError(
                f"No overlapping time period found. "
                f"Starts: {start_times}, Ends: {end_times}"
            )

        return pd.Timestamp(start_time), pd.Timestamp(end_time)

    def align_datasets(
        self,
        datasets: Dict[str, Optional[xr.Dataset]],
        time_var: str = 'time',
        resample_freq: Optional[str] = None
    ) -> Dict[str, Optional[xr.Dataset]]:
        """
        Align multiple datasets to a common time period.

        Args:
            datasets: Dictionary mapping names to datasets
            time_var: Name of time coordinate variable
            resample_freq: Optional frequency for time index (e.g., 'D', 'h')

        Returns:
            Dictionary of aligned datasets
        """
        # Filter out None datasets
        valid_datasets = {k: v for k, v in datasets.items() if v is not None}
        if not valid_datasets:
            return datasets

        # Find common time period
        start, end = self.find_common_time_period(
            list(valid_datasets.values()), time_var
        )
        self.logger.info(f"Aligning datasets to common period: {start} to {end}")

        # Create common time index if frequency specified
        if resample_freq:
            common_time = pd.date_range(start=start, end=end, freq=resample_freq)
        else:
            # Use the first dataset's time index within the period
            first_ds = next(iter(valid_datasets.values()))
            common_time = self._get_datetime_index(first_ds[time_var])
            common_time = common_time[(common_time >= start) & (common_time <= end)]

        # Align each dataset
        aligned: Dict[str, Optional[xr.Dataset]] = {}
        for name, ds in datasets.items():
            if ds is None:
                aligned[name] = None
            else:
                aligned[name] = self._align_single_dataset(
                    ds, common_time, time_var, name
                )

        return aligned

    def _align_single_dataset(
        self,
        ds: xr.Dataset,
        target_time: pd.DatetimeIndex,
        time_var: str,
        name: str
    ) -> xr.Dataset:
        """Align a single dataset to the target time index."""
        try:
            # First select the time slice
            start, end = target_time.min(), target_time.max()
            ds_sliced = ds.sel({time_var: slice(start, end)})

            # Then reindex to exact target times
            ds_aligned = ds_sliced.reindex({time_var: target_time})

            self.logger.debug(
                f"Aligned {name}: {len(ds[time_var])} -> {len(ds_aligned[time_var])} timesteps"
            )
            return ds_aligned

        except Exception as e:
            self.logger.warning(f"Error aligning {name}: {e}")
            return ds

    def align_array_to_time_length(
        self,
        data: np.ndarray,
        target_length: int,
        pad_strategy: str = 'last',
        axis: int = 0
    ) -> np.ndarray:
        """
        Align a numpy array to a target time dimension length.

        Handles both truncation (if data is longer) and padding (if data is shorter).

        Args:
            data: Input numpy array
            target_length: Desired length along time axis
            pad_strategy: How to pad if needed: 'last' (repeat last value),
                         'mean' (use mean), 'fill' (use fill_value)
            axis: Which axis is the time dimension

        Returns:
            Array with correct length along time axis
        """
        current_length = data.shape[axis]

        if current_length == target_length:
            return data

        if current_length > target_length:
            # Truncate
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(target_length)
            return data[tuple(slices)]

        # Pad
        pad_length = target_length - current_length

        if pad_strategy == 'last':
            # Get last value along the time axis
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(-1, None)
            last_value = data[tuple(slices)]
            pad_values = np.repeat(last_value, pad_length, axis=axis)
        elif pad_strategy == 'mean':
            mean_value = np.mean(data, axis=axis, keepdims=True)
            shape = list(data.shape)
            shape[axis] = pad_length
            pad_values = np.broadcast_to(mean_value, shape)
        else:  # 'fill'
            shape = list(data.shape)
            shape[axis] = pad_length
            pad_values = np.full(shape, self.fill_value)

        return np.concatenate([data, pad_values], axis=axis)

    def reshape_for_spatial_dims(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, ...],
        n_subcatchments: int = 1
    ) -> np.ndarray:
        """
        Reshape data to match target spatial dimensions.

        Handles common cases:
        - 1D time series -> 3D (time, lat, lon)
        - 2D (time, space) -> 3D (time, lat, lon)

        Args:
            data: Input data array
            target_shape: Expected output shape (time, dim1, dim2)
            n_subcatchments: Number of subcatchments for replication

        Returns:
            Reshaped array matching target_shape
        """
        time_length = target_shape[0]

        # Ensure time dimension matches
        if data.shape[0] != time_length:
            data = self.align_array_to_time_length(data, time_length)

        if len(data.shape) == 1:
            # 1D time series - replicate to spatial dimensions
            if target_shape[1] > target_shape[2]:
                return np.tile(data[:, np.newaxis, np.newaxis], (1, target_shape[1], 1))
            else:
                return np.tile(data[:, np.newaxis, np.newaxis], (1, 1, target_shape[2]))

        elif len(data.shape) == 2 and data.shape[1] == n_subcatchments:
            # 2D with subcatchments - add missing dimension
            if target_shape[1] > target_shape[2]:
                return data[:, :, np.newaxis]
            else:
                return data[:, np.newaxis, :]

        else:
            # Generic case - tile to match
            if target_shape[1] > target_shape[2]:
                return np.tile(data.reshape(-1, 1, 1), (1, target_shape[1], 1))
            else:
                return np.tile(data.reshape(-1, 1, 1), (1, 1, target_shape[2]))

    def handle_nan_values(
        self,
        data: np.ndarray,
        strategy: str = 'fill'
    ) -> np.ndarray:
        """
        Handle NaN values in data.

        Args:
            data: Input array potentially containing NaN
            strategy: How to handle: 'fill' (use fill_value), 'interpolate'

        Returns:
            Array with NaN handled
        """
        if not np.any(np.isnan(data)):
            return data

        if strategy == 'fill':
            return np.nan_to_num(data, nan=self.fill_value)
        elif strategy == 'interpolate':
            # Simple linear interpolation along first axis
            from scipy import interpolate
            mask = np.isnan(data)
            if len(data.shape) == 1:
                valid_idx = np.where(~mask)[0]
                if len(valid_idx) > 1:
                    f = interpolate.interp1d(
                        valid_idx, data[valid_idx],
                        kind='linear', fill_value='extrapolate'
                    )
                    data = f(np.arange(len(data)))
            return data
        else:
            return np.nan_to_num(data, nan=self.fill_value)

    def _get_datetime_index(
        self,
        time_coord: Union[xr.DataArray, np.ndarray]
    ) -> pd.DatetimeIndex:
        """
        Convert time coordinate to pandas DatetimeIndex.

        Handles both datetime64 and numeric time coordinates.
        """
        if isinstance(time_coord, xr.DataArray):
            values = time_coord.values
            attrs = time_coord.attrs
        else:
            values = time_coord
            attrs = {}

        # Check if already datetime
        if np.issubdtype(values.dtype, np.datetime64):
            return pd.DatetimeIndex(values)

        # Handle numeric time (days or hours since reference)
        if np.issubdtype(values.dtype, np.number):
            units = attrs.get('units', '')
            if 'hour' in units:
                return pd.Timestamp('1970-01-01') + pd.to_timedelta(values, unit='h')
            else:
                # Assume days since 1970-01-01
                return pd.Timestamp('1970-01-01') + pd.to_timedelta(values, unit='D')

        # Try pandas parser as fallback
        return pd.DatetimeIndex(values)


def align_forcing_datasets(
    forcing_ds: xr.Dataset,
    pet_data: xr.DataArray,
    obs_ds: Optional[xr.Dataset] = None,
    resample_freq: str = 'D',
    logger: Optional[logging.Logger] = None
) -> Tuple[xr.Dataset, xr.DataArray, Optional[xr.Dataset], pd.DatetimeIndex]:
    """
    Convenience function to align forcing, PET, and observation datasets.

    This is the common alignment pattern used across FUSE, NGEN, GR, and SUMMA.

    Args:
        forcing_ds: Main forcing dataset
        pet_data: PET data array
        obs_ds: Optional observation dataset
        resample_freq: Frequency for common time index
        logger: Logger instance

    Returns:
        Tuple of (aligned_forcing, aligned_pet, aligned_obs, common_time_index)
    """
    manager = DatasetAlignmentManager(logger=logger)

    # Build datasets dict
    datasets = {
        'forcing': forcing_ds,
        'pet': pet_data.to_dataset(name='pet'),
    }
    if obs_ds is not None:
        datasets['obs'] = obs_ds

    # Find common period and create index
    ds_list = [ds for ds in datasets.values() if ds is not None]
    start, end = manager.find_common_time_period(ds_list)
    common_time = pd.date_range(start=start, end=end, freq=resample_freq)

    # Align each dataset
    aligned = manager.align_datasets(datasets, resample_freq=resample_freq)

    # Extract back to original types
    aligned_forcing = aligned['forcing']
    assert aligned_forcing is not None, "Forcing dataset should not be None"
    aligned_pet = aligned['pet']['pet'] if aligned['pet'] is not None else pet_data
    aligned_obs = aligned.get('obs')

    return aligned_forcing, aligned_pet, aligned_obs, common_time
