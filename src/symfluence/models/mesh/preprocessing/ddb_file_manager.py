"""
Drainage Database File Manager

Handles all operations on the MESH drainage database NetCDF file (GRU columns,
normalization, spatial dimensions).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


class DDBFileManager:
    """Manages the MESH drainage database (MESH_drainage_database.nc).

    Provides methods for querying GRU counts, trimming GRU columns,
    normalizing GRU fractions, and identifying spatial dimensions.

    Args:
        ddb_path: Path to MESH_drainage_database.nc
        logger: Logger instance
    """

    def __init__(self, ddb_path: Path, logger: logging.Logger):
        self._path = ddb_path
        self.logger = logger

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_num_cells(self) -> int:
        """Get number of cells (subbasins) from drainage database."""
        if not self._path.exists():
            return 1
        try:
            with xr.open_dataset(self._path) as ds:
                for dim in ['subbasin', 'N']:
                    if dim in ds.sizes:
                        return int(ds.sizes[dim])
        except Exception:
            pass
        return 1

    def get_gru_count(self) -> Optional[int]:
        """Get the number of GRU columns (NGRU dimension) in the DDB."""
        if not self._path.exists():
            return None
        try:
            with xr.open_dataset(self._path) as ds:
                if 'NGRU' not in ds.dims:
                    return None
                return int(ds.sizes['NGRU'])
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def get_mesh_active_gru_count(self) -> Optional[int]:
        """Return the number of GRUs that MESH will actually read (NGRU-1).

        MESH has an off-by-one issue: it reads NGRU-1 GRUs from the
        drainage database.
        """
        if not self._path.exists():
            return None
        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return None
                ngru_dim = int(ds.sizes['NGRU'])
                mesh_gru_count = max(1, ngru_dim - 1) if ngru_dim > 1 else ngru_dim
                self.logger.debug(
                    f"MESH will read {mesh_gru_count} GRU(s) (NGRU dimension = {ngru_dim})"
                )
                return mesh_gru_count
        except Exception as e:
            self.logger.debug(f"Could not determine MESH active GRU count: {e}")
            return None

    @staticmethod
    def get_spatial_dim(ds: xr.Dataset) -> Optional[str]:
        """Get the spatial dimension name from dataset ('N' or 'subbasin')."""
        if 'N' in ds.dims:
            return 'N'
        elif 'subbasin' in ds.dims:
            return 'subbasin'
        return None

    def get_domain_latitude(self) -> Optional[float]:
        """Get representative latitude from drainage database."""
        if not self._path.exists():
            return None
        try:
            with xr.open_dataset(self._path) as ds:
                if 'lat' in ds:
                    return float(ds['lat'].values.mean())
        except (OSError, ValueError, KeyError) as e:
            self.logger.debug(f"Could not read latitude from drainage database: {e}")
        return None

    # ------------------------------------------------------------------
    # GRU trimming
    # ------------------------------------------------------------------

    def trim_to_active_grus(self, target_count: int) -> None:
        """Trim DDB GRU columns to exactly target_count, keeping first N columns.

        Keeps the first N GRU columns (not the largest) to maintain
        correspondence with CLASS parameter blocks. Renormalizes fractions.
        """
        if not self._path.exists():
            return

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                current_count = int(ds.sizes['NGRU'])
                if current_count <= target_count:
                    return

                gru = ds['GRU']
                sum_dim = self.get_spatial_dim(ds)
                if not sum_dim:
                    return

                sums = gru.sum(sum_dim).values
                removed_fractions = sums[target_count:]
                total_removed = sum(removed_fractions)

                self.logger.info(
                    f"Trimming DDB from {current_count} to {target_count} GRU columns "
                    f"(removing last {current_count - target_count} GRUs with "
                    f"total fraction {total_removed:.4f})"
                )

                keep_indices = list(range(target_count))
                ds_trim = ds.isel(NGRU=keep_indices)

                if 'GRU' in ds_trim:
                    sum_per = ds_trim['GRU'].sum('NGRU')
                    sum_safe = xr.where(sum_per == 0, 1.0, sum_per)
                    ds_trim['GRU'] = ds_trim['GRU'] / sum_safe
                    new_gru = ds_trim['GRU'].values
                    self.logger.debug(f"Renormalized GRU fractions: {new_gru}")

                temp_path = self._path.with_suffix('.tmp.nc')
                ds_trim.to_netcdf(temp_path)
                os.replace(temp_path, self._path)
                self.logger.info(
                    f"Trimmed DDB to {target_count} GRU column(s) and "
                    f"renormalized fractions to sum to 1.0"
                )
        except Exception as e:
            self.logger.warning(f"Failed to trim DDB to active GRUs: {e}")

    def trim_empty_gru_columns(self, min_total: float = 0.02) -> Optional[list]:
        """Trim empty GRU columns from drainage database.

        Args:
            min_total: Minimum total fraction to keep a column

        Returns:
            Boolean keep_mask list, or None on failure
        """
        if not self._path.exists():
            return None

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return None

                gru = ds['GRU']
                sum_dim = self.get_spatial_dim(ds)
                if not sum_dim:
                    return None

                sums = gru.sum(sum_dim)
                keep = sums > min_total
                keep_mask = keep.values.tolist()

                if int(keep.sum()) < int(gru.sizes['NGRU']):
                    removed = int(gru.sizes['NGRU'] - keep.sum())
                    ds_trim = ds.isel(NGRU=keep)

                    try:
                        sum_per = ds_trim['GRU'].sum('NGRU')
                        sum_safe = xr.where(sum_per == 0, 1.0, sum_per)
                        ds_trim['GRU'] = ds_trim['GRU'] / sum_safe
                    except Exception as e:
                        self.logger.debug(
                            f"Could not renormalize GRU fractions after trim: {e}"
                        )

                    temp_path = self._path.with_suffix('.tmp.nc')
                    ds_trim.to_netcdf(temp_path)
                    os.replace(temp_path, self._path)
                    self.logger.info(f"Removed {removed} empty GRU column(s)")

                return keep_mask

        except Exception as e:
            self.logger.warning(f"Failed to trim empty GRU columns: {e}")
            return None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def renormalize_active_grus(self, active_count: int) -> None:
        """Renormalize the first N GRU fractions to sum to 1.0.

        MESH reads only the first (NGRU-1) GRU columns. This normalizes
        only those active columns without changing the DDB dimension.

        Args:
            active_count: Number of GRUs that MESH will actually read
        """
        if not self._path.exists():
            return

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                ngru_count = int(ds.sizes['NGRU'])
                if active_count >= ngru_count:
                    self.ensure_gru_normalization()
                    return

                gru = ds['GRU'].values

                active_sums = gru[:, :active_count].sum(axis=1)
                if np.allclose(active_sums, 1.0, atol=1e-4):
                    self.logger.debug(
                        f"First {active_count} GRU fractions already sum to 1.0"
                    )
                    return

                self.logger.info(
                    f"Renormalizing first {active_count} GRU fractions to sum to 1.0 "
                    f"(current sum: {active_sums[0]:.4f})"
                )

                for i in range(gru.shape[0]):
                    row_sum = gru[i, :active_count].sum()
                    if row_sum > 0:
                        gru[i, :active_count] = gru[i, :active_count] / row_sum
                    else:
                        gru[i, 0] = 1.0
                    gru[i, active_count:] = 0.0

                ds['GRU'].values = gru

                temp_path = self._path.with_suffix('.tmp.nc')
                ds.to_netcdf(temp_path)
                os.replace(temp_path, self._path)
                self.logger.debug(f"Renormalized GRU fractions: {gru}")

        except Exception as e:
            self.logger.warning(f"Failed to renormalize MESH active GRUs: {e}")

    def ensure_gru_normalization(self) -> None:
        """Ensure GRU fractions in DDB sum to 1.0 for every subbasin."""
        if not self._path.exists():
            return

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                gru = ds['GRU']
                self.logger.debug(f"GRU values before norm: {gru.values}")
                n_dim = self.get_spatial_dim(ds)
                if not n_dim:
                    return

                sums = gru.sum('NGRU')
                self.logger.debug(f"GRU sums: {sums.values}")

                if np.allclose(sums.values, 1.0, atol=1e-4):
                    self.logger.debug("GRU fractions already normalized")
                    return

                self.logger.info("Normalizing GRU fractions in DDB to sum to 1.0")
                safe_sums = xr.where(sums == 0, 1.0, sums)
                ds['GRU'] = gru / safe_sums

                zero_sum_mask = (sums == 0)
                if zero_sum_mask.any():
                    self.logger.warning(
                        f"Found {int(zero_sum_mask.sum())} subbasins with 0 GRU coverage. "
                        f"Setting first GRU to 1.0."
                    )
                    gru_vals = ds['GRU'].values
                    zero_indices = np.where(zero_sum_mask.values)[0]
                    for idx in zero_indices:
                        gru_vals[idx, 0] = 1.0
                    ds['GRU'].values = gru_vals

                temp_path = self._path.with_suffix('.tmp.nc')
                ds.to_netcdf(temp_path)
                os.replace(temp_path, self._path)

        except Exception as e:
            self.logger.warning(f"Failed to normalize GRUs: {e}")

    # ------------------------------------------------------------------
    # Remove small GRUs from DDB
    # ------------------------------------------------------------------

    def remove_small_grus(self, min_fraction: float = 0.05) -> Optional[np.ndarray]:
        """Remove GRUs below the minimum fraction threshold from DDB.

        Returns the keep_mask (boolean array) for use with CLASS block removal,
        or None if no removal was needed.
        """
        if not self._path.exists():
            return None

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return None

                gru = ds['GRU']
                ngru = int(ds.sizes['NGRU'])

                sum_dim = self.get_spatial_dim(ds)
                if not sum_dim:
                    return None

                gru_fractions = gru.sum(sum_dim).values
                if gru.sizes[sum_dim] > 1:
                    gru_fractions = gru_fractions / gru.sizes[sum_dim]

                keep_mask = gru_fractions >= min_fraction
                n_keep = int(keep_mask.sum())
                n_remove = ngru - n_keep

                if n_remove == 0:
                    self.logger.debug(
                        f"All {ngru} GRUs above {min_fraction:.1%} threshold"
                    )
                    return None

                if n_keep == 0:
                    largest_idx = int(np.argmax(gru_fractions))
                    keep_mask[largest_idx] = True
                    n_keep = 1
                    n_remove = ngru - 1
                    self.logger.warning(
                        f"All GRUs below {min_fraction:.1%} threshold, "
                        f"keeping largest (idx={largest_idx})"
                    )

                removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]
                removed_fractions = [gru_fractions[i] for i in removed_indices]
                self.logger.info(
                    f"Removing {n_remove} GRUs below {min_fraction:.1%} threshold: "
                    f"indices {removed_indices} with fractions "
                    f"{[f'{f:.3f}' for f in removed_fractions]}"
                )

                keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                ds_filtered = ds.isel(NGRU=keep_indices)

                if 'GRU' in ds_filtered:
                    gru_sum = ds_filtered['GRU'].sum('NGRU')
                    gru_sum_safe = xr.where(gru_sum == 0, 1.0, gru_sum)
                    ds_filtered['GRU'] = ds_filtered['GRU'] / gru_sum_safe

                    new_fractions = ds_filtered['GRU'].sum(sum_dim).values
                    if ds_filtered['GRU'].sizes[sum_dim] > 1:
                        new_fractions = new_fractions / ds_filtered['GRU'].sizes[sum_dim]
                    self.logger.debug(
                        f"Renormalized GRU fractions: "
                        f"{[f'{f:.3f}' for f in new_fractions]}"
                    )

                temp_path = self._path.with_suffix('.tmp.nc')
                ds_filtered.to_netcdf(temp_path)
                os.replace(temp_path, self._path)

                self.logger.info(
                    f"Removed {n_remove} small GRU(s), {n_keep} remaining"
                )

                return keep_mask

        except Exception as e:
            self.logger.warning(f"Failed to remove small GRUs from DDB: {e}")
            return None

    # ------------------------------------------------------------------
    # Collapse to single GRU
    # ------------------------------------------------------------------

    def collapse_to_single_gru(self) -> None:
        """Collapse all GRUs to a single dominant land cover class in DDB.

        Creates a 2-column DDB (MESH reads NGRU-1=1) with the dominant
        GRU getting 100% fraction.
        """
        self.logger.info("MESH_FORCE_SINGLE_GRU enabled - collapsing to single GRU")

        if not self._path.exists():
            self.logger.warning("No DDB found, cannot collapse to single GRU")
            return

        try:
            with xr.open_dataset(self._path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    self.logger.warning("No GRU data in DDB")
                    return

                gru = ds['GRU'].values
                ngru = int(ds.sizes['NGRU'])

                gru_sums = gru.sum(axis=0)
                dominant_idx = int(np.argmax(gru_sums))
                dominant_fraction = gru_sums[dominant_idx]

                self.logger.info(
                    f"Dominant GRU index: {dominant_idx} "
                    f"(fraction: {dominant_fraction:.4f})"
                )

                new_gru = np.zeros((gru.shape[0], 2))
                new_gru[:, 0] = 1.0

                ds_new = ds.copy()
                ds_new = ds_new.drop_dims('NGRU')
                ds_new['GRU'] = xr.DataArray(
                    new_gru,
                    dims=['subbasin', 'NGRU'],
                    coords={'subbasin': ds['subbasin']}
                )

                for var in ['LandUse', 'LandClass']:
                    if var in ds:
                        old_vals = ds[var].values
                        if old_vals.ndim == 1:
                            new_vals = np.array([old_vals[dominant_idx], 0])
                        else:
                            new_vals = np.zeros((old_vals.shape[0], 2))
                            new_vals[:, 0] = old_vals[:, dominant_idx]
                        ds_new[var] = xr.DataArray(
                            new_vals,
                            dims=ds[var].dims if len(new_vals.shape) > 1 else ['NGRU']
                        )

                temp_path = self._path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path)
                os.replace(temp_path, self._path)

                self.logger.info(
                    f"Collapsed DDB from {ngru} to 2 GRU columns (MESH reads 1)"
                )

        except Exception as e:
            self.logger.warning(f"Failed to collapse DDB to single GRU: {e}")
