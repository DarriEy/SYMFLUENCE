"""
MESH Drainage Database Handler

Handles drainage database topology fixes, completeness, and normalization.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import xarray as xr
import geopandas as gpd

from symfluence.core.mixins import ConfigMixin


class MESHDrainageDatabase(ConfigMixin):
    """
    Manages MESH drainage database fixes and completeness.

    Handles:
    - Topology fixes when meshflow fails to build routing
    - Adding missing required variables (IREACH, IAK, AL, DA)
    - Reordering by Rank and normalizing GRU fractions
    """

    def __init__(
        self,
        forcing_dir: Path,
        rivers_path: Path,
        rivers_name: str,
        catchment_path: Path,
        catchment_name: str,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize drainage database handler.

        Args:
            forcing_dir: Directory containing MESH files
            rivers_path: Path to river network directory
            rivers_name: River network shapefile name
            catchment_path: Path to catchment directory
            catchment_name: Catchment shapefile name
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.rivers_path = rivers_path
        self.rivers_name = rivers_name
        self.catchment_path = catchment_path
        self.catchment_name = catchment_name
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)

    @property
    def ddb_path(self) -> Path:
        """Path to drainage database file."""
        return self.forcing_dir / "MESH_drainage_database.nc"

    def fix_drainage_topology(self) -> None:
        """
        Fix drainage database topology if meshflow failed to build it properly.

        When meshflow fails to establish routing connectivity, all Next values
        become 0 (all GRUs treated as outlets). This method rebuilds the topology
        from the river network shapefile.
        """
        if not self.ddb_path.exists():
            self.logger.warning("Drainage database not found, skipping topology fix")
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    return

                n_size = ds.dims[n_dim]

                # Check if topology is broken
                if 'Next' not in ds:
                    self.logger.warning("Next array missing from drainage database")
                    needs_fix = True
                else:
                    next_arr = ds['Next'].values
                    needs_fix = np.all(next_arr <= 0)
                    if needs_fix:
                        self.logger.warning(f"Topology broken: all {n_size} GRUs have Next=0")

                if not needs_fix:
                    self.logger.info("Drainage topology appears valid")
                    return

            # Rebuild topology
            self._rebuild_topology(n_size)

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to fix drainage topology: {e}")
            self.logger.debug(traceback.format_exc())

    def _rebuild_topology(self, n_size: int) -> None:
        """Rebuild topology from river network."""
        self.logger.info("Rebuilding drainage topology from river network")

        # Load river network
        river_shp = self.rivers_path / self.rivers_name
        if not river_shp.exists():
            self.logger.error(f"River network shapefile not found: {river_shp}")
            return

        riv_gdf = gpd.read_file(river_shp)

        if 'LINKNO' not in riv_gdf.columns or 'DSLINKNO' not in riv_gdf.columns:
            self.logger.error("River network missing LINKNO or DSLINKNO columns")
            return

        # Build mappings
        linkno_to_dslink = dict(zip(riv_gdf['LINKNO'], riv_gdf['DSLINKNO']))
        valid_linknos = set(riv_gdf['LINKNO'].values)

        # Load river basins
        basin_shp = self.catchment_path / self.catchment_name
        if not basin_shp.exists():
            self.logger.error(f"River basins shapefile not found: {basin_shp}")
            return

        basin_gdf = gpd.read_file(basin_shp)

        # Find GRU ID field
        gru_id_field = None
        for field in ['GRU_ID', 'GRUID', 'gru_id', 'LINKNO']:
            if field in basin_gdf.columns:
                gru_id_field = field
                break

        if gru_id_field is None:
            self.logger.error("Could not find GRU ID field in river basins")
            return

        gru_ids = basin_gdf[gru_id_field].values
        n_grus = len(gru_ids)

        if n_grus != n_size:
            self.logger.warning(f"GRU count mismatch: basins={n_grus}, DDB={n_size}")

        # Build GRU_ID -> index mapping
        gru_to_idx = {int(gid): i for i, gid in enumerate(gru_ids)}
        outlet_value = self._get_config_value(lambda: self.config.model.mesh.outlet_value, default=-9999, dict_key='MESH_OUTLET_VALUE')

        # Build downstream GRU mapping
        gru_to_ds_gru = {}
        for gid in gru_ids:
            gid = int(gid)
            ds_linkno = linkno_to_dslink.get(gid)
            if ds_linkno is None or ds_linkno == outlet_value or ds_linkno not in valid_linknos:
                gru_to_ds_gru[gid] = 0
            elif ds_linkno in gru_to_idx:
                gru_to_ds_gru[gid] = int(ds_linkno)
            else:
                gru_to_ds_gru[gid] = 0

        # Build DA-based connectivity if no downstream links
        if all(v == 0 for v in gru_to_ds_gru.values()):
            gru_to_ds_gru = self._build_da_connectivity(gru_ids, n_grus)

        # Compute topological levels
        levels = self._compute_levels(gru_ids, gru_to_ds_gru, n_grus)

        # Sort and assign ranks
        sorted_grus = sorted(gru_ids, key=lambda g: (-levels.get(int(g), 0), int(g)))
        gru_to_rank = {int(g): i + 1 for i, g in enumerate(sorted_grus)}

        # Build arrays
        self._update_ddb_topology(gru_ids, gru_to_ds_gru, gru_to_rank, levels, n_grus)

    def _build_da_connectivity(self, gru_ids: np.ndarray, n_grus: int) -> Dict[int, int]:
        """Build connectivity based on drainage area."""
        self.logger.warning("River network has no downstream links; building DA-based connectivity")

        gru_to_ds_gru = {int(gid): 0 for gid in gru_ids}

        with xr.open_dataset(self.ddb_path) as ds:
            if 'DA' in ds:
                da_vals = ds['DA'].values
            elif 'GridArea' in ds:
                da_vals = ds['GridArea'].values
            else:
                return gru_to_ds_gru

            n_dim = self._get_spatial_dim(ds)
            if n_dim in ds:
                ddb_ids = [int(v) for v in ds[n_dim].values]
            elif 'N' in ds:
                ddb_ids = [int(v) for v in ds['N'].values]
            else:
                ddb_ids = [int(v) for v in gru_ids]

            da_map = {ddb_ids[i]: float(da_vals[i]) for i in range(min(len(ddb_ids), len(da_vals)))}
            sorted_by_da = sorted(da_map.items(), key=lambda x: x[1], reverse=True)

            for i, (gid, _) in enumerate(sorted_by_da):
                if i == 0:
                    gru_to_ds_gru[gid] = 0
                else:
                    gru_to_ds_gru[gid] = int(sorted_by_da[i - 1][0])

        return gru_to_ds_gru

    def _compute_levels(
        self,
        gru_ids: np.ndarray,
        gru_to_ds_gru: Dict[int, int],
        n_grus: int
    ) -> Dict[int, int]:
        """Compute topological levels using iterative propagation."""
        levels = {int(gid): 0 for gid in gru_ids}
        changed = True
        max_iter = n_grus + 1
        iteration = 0

        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for gid in gru_ids:
                gid = int(gid)
                ds_gid = gru_to_ds_gru.get(gid, 0)
                if ds_gid == 0:
                    if levels[gid] != 1:
                        levels[gid] = 1
                        changed = True
                else:
                    ds_level = levels.get(ds_gid, 0)
                    if ds_level > 0:
                        new_level = ds_level + 1
                        if levels[gid] != new_level:
                            levels[gid] = new_level
                            changed = True

        return levels

    def _update_ddb_topology(
        self,
        gru_ids: np.ndarray,
        gru_to_ds_gru: Dict[int, int],
        gru_to_rank: Dict[int, int],
        levels: Dict[int, int],
        n_grus: int
    ) -> None:
        """Update drainage database with corrected topology."""
        with xr.open_dataset(self.ddb_path) as ds:
            n_dim = self._get_spatial_dim(ds)

            if n_dim in ds:
                ddb_ids = [int(v) for v in ds[n_dim].values]
            elif 'N' in ds:
                ddb_ids = [int(v) for v in ds['N'].values]
            else:
                ddb_ids = [int(v) for v in gru_ids]

        next_arr = np.zeros(n_grus, dtype=np.int32)
        rank_arr = np.zeros(n_grus, dtype=np.int32)

        for i, gid in enumerate(ddb_ids):
            rank_arr[i] = gru_to_rank.get(gid, i + 1)
            ds_gid = gru_to_ds_gru.get(gid, 0)
            if ds_gid == 0:
                next_arr[i] = 0
            else:
                next_arr[i] = gru_to_rank.get(ds_gid, 0)

        n_outlets = np.sum(next_arr == 0)
        max_level = max(levels.values()) if levels else 0

        self.logger.info(f"Rebuilt topology: {n_grus} GRUs, {n_outlets} outlet(s), max level={max_level}")

        # Reorder by rank and remap Next values
        with xr.open_dataset(self.ddb_path) as ds:
            n_dim = self._get_spatial_dim(ds)
            order_idx = np.argsort(rank_arr)
            ds_new = ds.isel({n_dim: order_idx}).copy(deep=True)

            # After sorting, ranks become sequential [1, 2, 3, ...]
            old_ranks_sorted = rank_arr[order_idx]
            new_ranks = np.arange(1, n_grus + 1, dtype=np.int32)

            # Build mapping from old rank -> new rank
            rank_remap = {int(old): int(new) for old, new in zip(old_ranks_sorted, new_ranks)}

            # Remap Next values to point to new ranks
            next_arr_sorted = next_arr[order_idx]
            next_arr_remapped = np.array([
                rank_remap.get(int(val), 0) if val > 0 else 0
                for val in next_arr_sorted
            ], dtype=np.int32)

            # CRITICAL: For single-cell (lumped) domains, MESH uses max(Next) to
            # determine the number of active cells for array sizing. With Next=0,
            # max(Next)=0, so arrays are sized to 0 and nothing can be read.
            # Fix: Set Next=1 (self-reference) for single-cell domains.
            if n_grus == 1 and next_arr_remapped[0] == 0:
                next_arr_remapped[0] = 1
                self.logger.info("Single-cell domain: set Next=1 (self-reference) for MESH array sizing")

            ds_new['Next'] = xr.DataArray(
                next_arr_remapped,
                dims=[n_dim],
                attrs={'long_name': 'Downstream cell rank', 'units': '1'}
            )
            ds_new['Rank'] = xr.DataArray(
                new_ranks,
                dims=[n_dim],
                attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
            )

            if 'GRU' in ds_new and 'NGRU' in ds_new.dims:
                # Ensure GRU fractions sum to 1.0 for each subbasin
                # gru is (N, NGRU)
                gru_da = ds_new['GRU']
                gru_sums = gru_da.sum('NGRU')
                # Avoid division by zero, default to 1.0 if all zero
                safe_sums = xr.where(gru_sums == 0, 1.0, gru_sums)
                ds_new['GRU'] = gru_da / safe_sums

                # If sum was 0, set the first GRU to 1.0 as a fallback
                if (gru_sums == 0).any():
                    self.logger.warning("Found subbasins with 0 GRU coverage during topology update. Setting first GRU to 1.0.")
                    vals = ds_new['GRU'].values
                    zero_indices = np.where(gru_sums.values == 0)[0]
                    for idx in zero_indices:
                        vals[idx, 0] = 1.0
                    ds_new['GRU'].values = vals

            temp_path = self.ddb_path.with_suffix('.tmp.nc')
            ds_new.to_netcdf(temp_path)
            os.replace(temp_path, self.ddb_path)

        self.logger.info("Updated drainage database with corrected topology")

    def ensure_completeness(self) -> None:
        """Ensure drainage database has all required variables for MESH."""
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    return

                n_size = ds.dims[n_dim]
                modified = False

                # MESH 1.5 with nc_subbasin expects 'subbasin' as the spatial dimension
                target_n_dim = 'subbasin'
                if n_dim != target_n_dim:
                    self.logger.info(f"Renaming spatial dimension '{n_dim}' to '{target_n_dim}'")
                    ds = ds.rename({n_dim: target_n_dim})
                    n_dim = target_n_dim
                    modified = True

                # Also rename variables if they match the old dimension name
                for old_name in ['N']:
                    if old_name in ds.coords or old_name in ds.data_vars:
                        ds = ds.rename({old_name: target_n_dim})
                        modified = True

                # Ensure dimension 'subbasin' is the index and has correct values
                ds[target_n_dim] = xr.DataArray(
                    np.arange(1, n_size + 1, dtype=np.int32),
                    dims=[target_n_dim],
                    attrs={'long_name': 'Grid index', 'units': '1'}
                )
                modified = True

                # Rename NGRU or land dimension to 'landclass'
                old_lc_dim = 'land' if 'land' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else 'landclass' if 'landclass' in ds.dims else None
                if old_lc_dim and old_lc_dim != 'landclass':
                    self.logger.info(f"Renaming dimension '{old_lc_dim}' to 'landclass'")
                    ds = ds.rename({old_lc_dim: 'landclass'})
                    modified = True

                # Ensure GRU variable exists and is on (subbasin, landclass)
                if 'GRU' in ds:
                    # Correct dimension names if needed
                    if ds['GRU'].dims != (target_n_dim, 'landclass'):
                        ds['GRU'] = ((target_n_dim, 'landclass'), ds['GRU'].values, ds['GRU'].attrs)
                        modified = True
                    ds['GRU'].attrs['grid_mapping'] = 'crs'

                    # For lumped mode, optionally force single landclass
                    force_single_gru = bool(self.config_dict.get('MESH_FORCE_SINGLE_GRU', False))
                    if force_single_gru and 'landclass' in ds.dims and ds.dims['landclass'] > 1:
                        self.logger.info(
                            f"Trimming landclass dimension from {ds.dims['landclass']} to 1 "
                            f"(MESH_FORCE_SINGLE_GRU enabled)"
                        )
                        ds = ds.isel(landclass=slice(0, 1))
                        ds['GRU'] = ((target_n_dim, 'landclass'), np.ones((n_size, 1), dtype=np.float64), ds['GRU'].attrs)
                        modified = True

                elif 'landclass' in ds:
                    # If landclass is present but GRU is missing, treat landclass as GRU fractions
                    self.logger.info("Found 'landclass' variable without GRU; renaming to GRU")
                    ds['GRU'] = ((target_n_dim, 'landclass'), ds['landclass'].values, ds['landclass'].attrs)
                    ds = ds.drop_vars('landclass')
                    ds['GRU'].attrs['grid_mapping'] = 'crs'
                    modified = True

                    # For lumped mode, optionally force single landclass
                    force_single_gru = bool(self.config_dict.get('MESH_FORCE_SINGLE_GRU', False))
                    if force_single_gru and 'landclass' in ds.dims and ds.dims['landclass'] > 1:
                        self.logger.info(
                            f"Trimming landclass dimension from {ds.dims['landclass']} to 1 "
                            f"(MESH_FORCE_SINGLE_GRU enabled)"
                        )
                        ds = ds.isel(landclass=slice(0, 1))
                        ds['GRU'] = ((target_n_dim, 'landclass'), np.ones((n_size, 1), dtype=np.float64), ds['GRU'].attrs)
                        modified = True
                else:
                    # Create GRU variable if it doesn't exist (e.g., for lumped mode)
                    # For lumped mode, we have 1 subbasin and 1 landclass, with 100% coverage
                    self.logger.info("Creating GRU variable for lumped mode (1 landclass, 100% coverage)")
                    n_landclass = 1  # Single landcover class for lumped mode
                    landclass_data = np.ones((n_size, n_landclass), dtype=np.float64)
                    ds['GRU'] = ((target_n_dim, 'landclass'), landclass_data, {
                        'long_name': 'Land cover class fractions',
                        'units': '1',
                        'grid_mapping': 'crs'
                    })
                    modified = True

                # Ensure all variables are 1D over N (except landclass which is 2D)
                # and remove potentially conflicting variables
                vars_to_remove = ['landclass_dim', 'time']
                for v in vars_to_remove:
                    if v in ds:
                        ds = ds.drop_vars(v)

                # Don't drop 'subbasin' as it's now a coordinate we need

                # Also drop 'time' dimension if it exists as a coordinate or dim
                if 'time' in ds.dims:
                    ds = ds.isel(time=0, drop=True)

                # Robustly add/ensure CRS variable
                if 'crs' not in ds:
                    ds['crs'] = xr.DataArray(
                        np.array(0, dtype=np.int32),
                        attrs={
                            'grid_mapping_name': 'latitude_longitude',
                            'semi_major_axis': 6378137.0,
                            'inverse_flattening': 298.257223563,
                            'longitude_of_prime_meridian': 0.0
                        }
                    )
                    modified = True
                elif ds['crs'].dtype != np.int32:
                    ds['crs'] = ds['crs'].astype(np.int32)
                    modified = True

                for var_name in list(ds.data_vars):
                    if var_name == 'crs':
                        continue

                    if var_name == 'GRU':
                        ds[var_name].attrs['grid_mapping'] = 'crs'
                        continue

                    if n_dim not in ds[var_name].dims:
                        self.logger.warning(f"Variable {var_name} missing dimension {n_dim}. Forcing it.")
                        # If it has another dimension, take the first value
                        if ds[var_name].values.size > 0:
                            temp_data = ds[var_name].values.flatten()[0]
                        else:
                            temp_data = 0
                        ds[var_name] = xr.DataArray(
                            np.full(n_size, temp_data, dtype=ds[var_name].dtype),
                            dims=[n_dim],
                            attrs=ds[var_name].attrs
                        )
                        modified = True
                    elif len(ds[var_name].dims) > 1:
                        # Drop other dims if they are singletons
                        other_dims = [d for d in ds[var_name].dims if d != n_dim]
                        self.logger.debug(f"Squeezing {var_name} to 1D over {n_dim} (removing {other_dims})")
                        ds[var_name] = ds[var_name].isel({d: 0 for d in other_dims}, drop=True)
                        modified = True

                    # Ensure it is explicitly on n_dim
                    ds[var_name] = ds[var_name].transpose(n_dim)
                    ds[var_name].attrs['grid_mapping'] = 'crs'
                    # Remove coordinate attributes that might point to 'time' or other missing coords
                    if 'coordinates' in ds[var_name].attrs:
                        coords_str = ds[var_name].attrs['coordinates']
                        new_coords = " ".join([c for c in coords_str.split() if c in ['lat', 'lon', 'N']])
                        if new_coords:
                            ds[var_name].attrs['coordinates'] = new_coords
                        else:
                            del ds[var_name].attrs['coordinates']

                # Special handling for lat/lon - must be 1D on subbasin
                for coord in ['lat', 'lon']:
                    if coord in ds:
                        # Force to data variable if it's a coordinate
                        if coord in ds.coords:
                            ds = ds.reset_coords(coord)

                        # Re-create as clean 1D variable on subbasin, preserving all values
                        vals = ds[coord].values.flatten()
                        if len(vals) >= n_size:
                            # Use the first n_size values (handles case where we have enough)
                            coord_vals = vals[:n_size].astype(np.float64)
                        elif len(vals) == 1:
                            # Single value - broadcast to all subbasins (legacy behavior)
                            coord_vals = np.full(n_size, vals[0], dtype=np.float64)
                        else:
                            # Fewer values than needed - pad with last value
                            coord_vals = np.zeros(n_size, dtype=np.float64)
                            coord_vals[:len(vals)] = vals
                            coord_vals[len(vals):] = vals[-1]
                            self.logger.warning(
                                f"Lat/lon array has {len(vals)} values but need {n_size}, "
                                f"padding with last value"
                            )

                        ds[coord] = (target_n_dim, coord_vals, {
                            'units': 'degrees_north' if coord == 'lat' else 'degrees_east',
                            'long_name': 'latitude' if coord == 'lat' else 'longitude',
                            'standard_name': 'latitude' if coord == 'lat' else 'longitude',
                            'grid_mapping': 'crs'
                        })
                        modified = True

                # Ensure 1D coordinate for landclass dimension
                if 'landclass' in ds.dims:
                    n_land = int(ds.dims['landclass'])
                    ds = ds.assign_coords(landclass=np.arange(1, n_land + 1, dtype=np.int32))
                    modified = True

                # Remove global attributes that might confuse MESH
                for attr in ['crs', 'grid_mapping', 'featureType']:
                    if attr in ds.attrs:
                        del ds.attrs[attr]

                # Add IREACH if missing (0 = no reservoir, positive = reservoir ID)
                # Note: Do NOT use _FillValue for IREACH - MESH interprets it incorrectly
                # and reads -2147483648 (INT_MIN) which causes reservoir mismatch errors
                if 'IREACH' not in ds:
                    ds['IREACH'] = xr.DataArray(
                        np.zeros(n_size, dtype=np.int32),
                        dims=[n_dim],
                        attrs={'long_name': 'Reservoir ID', 'units': '1'}
                    )
                    modified = True
                    self.logger.info("Added IREACH to drainage database")
                else:
                    # Fix existing IREACH if it has problematic _FillValue attribute
                    if '_FillValue' in ds['IREACH'].attrs:
                        # Remove the _FillValue and ensure clean int32 encoding
                        ireach_vals = ds['IREACH'].values.copy()
                        # Replace any fill values with 0 (no reservoir)
                        fill_val = ds['IREACH'].attrs['_FillValue']
                        ireach_vals = np.where(ireach_vals == fill_val, 0, ireach_vals)
                        ireach_vals = np.where(np.isnan(ireach_vals.astype(float)), 0, ireach_vals)
                        ds['IREACH'] = xr.DataArray(
                            ireach_vals.astype(np.int32),
                            dims=[n_dim],
                            attrs={'long_name': 'Reservoir ID', 'units': '1'}
                        )
                        modified = True
                        self.logger.info("Fixed IREACH encoding (removed problematic _FillValue)")

                # Add IAK if missing (river class, 1 = default single class)
                # Note: Do NOT use _FillValue for IAK - same issue as IREACH
                if 'IAK' not in ds:
                    ds['IAK'] = xr.DataArray(
                        np.ones(n_size, dtype=np.int32),
                        dims=[n_dim],
                        attrs={'long_name': 'River class', 'units': '1'}
                    )
                    modified = True
                    self.logger.info("Added IAK to drainage database")
                elif '_FillValue' in ds['IAK'].attrs:
                    # Fix existing IAK if it has problematic _FillValue attribute
                    iak_vals = ds['IAK'].values.copy()
                    fill_val = ds['IAK'].attrs['_FillValue']
                    iak_vals = np.where(iak_vals == fill_val, 1, iak_vals)
                    ds['IAK'] = xr.DataArray(
                        iak_vals.astype(np.int32),
                        dims=[n_dim],
                        attrs={'long_name': 'River class', 'units': '1'}
                    )
                    modified = True
                    self.logger.info("Fixed IAK encoding (removed problematic _FillValue)")

                # Add AL (characteristic length)
                # AL is used in MESH routing for time-of-concentration calculations
                # Prefer: perimeter-based > channel length > sqrt(area) fallback
                if 'AL' not in ds and 'GridArea' in ds:
                    grid_area = ds['GridArea'].values

                    if 'Perimeter' in ds:
                        # Best: use equivalent radius from perimeter (A = pi*r^2, P = 2*pi*r)
                        perimeter = ds['Perimeter'].values
                        # Characteristic length = perimeter / (2*pi) for circular approximation
                        char_length = perimeter / (2 * np.pi)
                        method = 'perimeter-based'
                    elif 'ChnlLength' in ds:
                        # Good: use channel length if available (represents flow path)
                        char_length = ds['ChnlLength'].values
                        method = 'channel length'
                    elif 'ChnlLen' in ds:
                        char_length = ds['ChnlLen'].values
                        method = 'channel length'
                    else:
                        # Fallback: sqrt(area) assumes square cells (less accurate for irregular polygons)
                        char_length = np.sqrt(grid_area)
                        method = 'sqrt(area) fallback'
                        self.logger.warning(
                            "AL computed from sqrt(area) - may be inaccurate for irregular catchments. "
                            "Consider adding Perimeter or ChnlLength to input shapefiles."
                        )

                    # Ensure positive values
                    char_length = np.maximum(char_length, 1.0)

                    ds['AL'] = xr.DataArray(
                        char_length,
                        dims=[n_dim],
                        attrs={
                            'long_name': 'Characteristic length of grid',
                            'units': 'm',
                            '_FillValue': np.nan
                        }
                    )
                    modified = True
                    self.logger.info(f"Added AL ({method}): min={char_length.min():.1f}m, max={char_length.max():.1f}m")

                # Add DA (drainage area)
                if 'DA' not in ds and 'GridArea' in ds and 'Next' in ds and 'Rank' in ds:
                    ds['DA'] = self._compute_drainage_area(ds, n_dim, n_size)
                    modified = True

                # Add Area if missing (alias of GridArea)
                if 'Area' not in ds and 'GridArea' in ds:
                    ds['Area'] = xr.DataArray(
                        ds['GridArea'].values,
                        dims=[n_dim],
                        attrs={'long_name': 'Grid area', 'units': 'm2'}
                    )
                    modified = True
                    self.logger.info("Added Area (alias of GridArea) to drainage database")

                # Add Slope if missing (use ChnlSlope if available)
                if 'Slope' not in ds:
                    if 'ChnlSlope' in ds:
                        slope_vals = ds['ChnlSlope'].values
                    else:
                        slope_vals = np.full(n_size, 0.001, dtype=np.float64)

                    # Ensure no NaNs or non-positive values
                    slope_vals = np.where(np.isnan(slope_vals), 0.001, slope_vals)
                    slope_vals = np.where(slope_vals <= 0, 0.001, slope_vals)

                    ds['Slope'] = xr.DataArray(
                        slope_vals,
                        dims=[n_dim],
                        attrs={'long_name': 'Mean basin slope', 'units': '1'}
                    )
                    modified = True
                    self.logger.info("Added Slope to drainage database")

                # Add ChnlLen if missing (alias of ChnlLength)
                if 'ChnlLen' not in ds and 'ChnlLength' in ds:
                    ds['ChnlLen'] = xr.DataArray(
                        ds['ChnlLength'].values,
                        dims=[n_dim],
                        attrs={'long_name': 'Channel length', 'units': 'm'}
                    )
                    modified = True
                    self.logger.info("Added ChnlLen (alias of ChnlLength) to drainage database")

                # Normalize integer-like fields to int32
                for name in ['Rank', 'Next', 'IAK', 'IREACH', 'N']:
                    if name in ds and ds[name].dtype != np.int32:
                        ds[name] = ds[name].astype(np.int32)
                        modified = True
                        self.logger.info(f"Coerced {name} to int32 in drainage database")

                if modified:
                    # Final GRU/landclass normalization check
                    lc_var = 'landclass' if 'landclass' in ds else 'GRU' if 'GRU' in ds else None
                    lc_dim = 'landclass' if 'landclass' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else None

                    if lc_var and lc_dim:
                        lc_da = ds[lc_var]
                        lc_sums = lc_da.sum(lc_dim)
                        safe_sums = xr.where(lc_sums == 0, 1.0, lc_sums)
                        ds[lc_var] = lc_da / safe_sums

                        if (lc_sums == 0).any():
                            self.logger.warning(f"Found subbasins with 0 {lc_var} coverage during completeness check. Setting first entry to 1.0.")
                            vals = ds[lc_var].values
                            zero_indices = np.where(lc_sums.values == 0)[0]
                            for idx in zero_indices:
                                vals[idx, 0] = 1.0
                            ds[lc_var].values = vals

                    temp_path = self.ddb_path.with_suffix('.tmp.nc')
                    ds.to_netcdf(temp_path)
                    os.replace(temp_path, self.ddb_path)

        except Exception as e:
            self.logger.warning(f"Failed to ensure DDB completeness: {e}")

    def _compute_drainage_area(
        self,
        ds: xr.Dataset,
        n_dim: str,
        n_size: int
    ) -> xr.DataArray:
        """Compute accumulated drainage area."""
        grid_area = ds['GridArea'].values
        next_arr = ds['Next'].values.astype(int)
        rank_arr = ds['Rank'].values.astype(int)

        da = grid_area.copy()

        for _ in range(n_size):
            changed = False
            for i in range(n_size):
                if next_arr[i] > 0:
                    ds_idx = np.where(rank_arr == next_arr[i])[0]
                    if len(ds_idx) > 0:
                        ds_idx = ds_idx[0]
                        # Skip self-referencing outlets (subbasin drains to itself)
                        if ds_idx == i:
                            continue
                        new_da = da[ds_idx] + grid_area[i]
                        if new_da != da[ds_idx]:
                            da[ds_idx] = new_da
                            changed = True
            if not changed:
                break

        self.logger.info(f"Added DA: max={da.max()/1e6:.1f} km²")

        return xr.DataArray(
            da,
            dims=[n_dim],
            attrs={
                'long_name': 'Drainage area',
                'units': 'm**2',
                'coordinates': 'lon lat',
                '_FillValue': np.nan
            }
        )

    def reorder_by_rank_and_normalize(self) -> None:
        """Reorder drainage database by Rank and normalize GRU fractions."""
        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("No spatial dimension found")
                    return

                if 'Rank' not in ds or 'Next' not in ds:
                    self.logger.warning("Rank/Next not found, skipping reorder")
                    return

                rank_arr = ds['Rank'].values.astype(int)
                next_arr = ds['Next'].values.astype(int)
                order_idx = np.argsort(rank_arr)

                ds_new = ds.isel({n_dim: order_idx}).copy(deep=True)
                old_rank_sorted = rank_arr[order_idx]
                new_rank_arr = np.arange(1, len(old_rank_sorted) + 1, dtype=np.int32)
                rank_map = {int(old): int(new) for old, new in zip(old_rank_sorted, new_rank_arr)}

                next_sorted = next_arr[order_idx]
                next_remap = np.array(
                    [rank_map.get(int(val), 0) if int(val) > 0 else 0 for val in next_sorted],
                    dtype=np.int32
                )

                # CRITICAL: For single-cell (lumped) domains, set Next=1 (self-reference)
                # so MESH can properly size its internal arrays
                n_grus = len(new_rank_arr)
                if n_grus == 1 and next_remap[0] == 0:
                    next_remap[0] = 1
                    self.logger.debug("Single-cell domain: preserving Next=1 for MESH")

                ds_new['Rank'] = xr.DataArray(
                    new_rank_arr,
                    dims=[n_dim],
                    attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
                )
                ds_new['Next'] = xr.DataArray(
                    next_remap,
                    dims=[n_dim],
                    attrs={'long_name': 'Downstream cell rank', 'units': '1'}
                )

                if 'GRU' in ds_new and 'NGRU' in ds_new.dims:
                    # Ensure GRU fractions sum to 1.0 for each subbasin
                    gru_da = ds_new['GRU']
                    gru_sums = gru_da.sum('NGRU')
                    # Avoid division by zero, default to 1.0 if all zero
                    safe_sums = xr.where(gru_sums == 0, 1.0, gru_sums)
                    ds_new['GRU'] = gru_da / safe_sums

                    # If sum was 0, set the first GRU to 1.0 as a fallback
                    if (gru_sums == 0).any():
                        self.logger.warning("Found subbasins with 0 GRU coverage during reorder. Setting first GRU to 1.0.")
                        vals = ds_new['GRU'].values
                        zero_indices = np.where(gru_sums.values == 0)[0]
                        for idx in zero_indices:
                            vals[idx, 0] = 1.0
                        ds_new['GRU'].values = vals

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

            self.logger.info("Reordered by Rank and normalized GRU fractions")

        except Exception as e:
            self.logger.warning(f"Failed to reorder: {e}")

    def _get_spatial_dim(self, ds: xr.Dataset) -> Optional[str]:
        """Get the spatial dimension name."""
        if 'N' in ds.dims:
            return 'N'
        elif 'subbasin' in ds.dims:
            return 'subbasin'
        return None
