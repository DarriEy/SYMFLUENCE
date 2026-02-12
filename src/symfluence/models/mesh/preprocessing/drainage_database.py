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

    def _get_landcover_class_ids(self, n_land: int) -> Optional[list]:
        """Get actual landcover class IDs from landcover stats file.

        Reads the frac_* column names from the landcover stats CSV to determine
        which NALCMS/IGBP class IDs are present.

        Args:
            n_land: Expected number of landcover classes

        Returns:
            List of integer class IDs, or None if not determinable
        """
        import re
        import pandas as pd

        # Look for landcover stats file in forcing directory or attributes
        possible_paths = [
            self.forcing_dir / "temp_modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv",
            self.forcing_dir.parent.parent / "attributes" / "gistool-outputs" / "modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv",
        ]

        for lc_path in possible_paths:
            if lc_path.exists():
                try:
                    df = pd.read_csv(lc_path)
                    # Extract class IDs from frac_* columns
                    frac_cols = [col for col in df.columns if col.startswith('frac_')]
                    class_ids = []
                    for col in frac_cols:
                        match = re.match(r'frac_(\d+)', col)
                        if match:
                            class_ids.append(int(match.group(1)))

                    if class_ids:
                        self.logger.debug(f"Found landcover class IDs from {lc_path.name}: {sorted(class_ids)}")
                        return sorted(class_ids)
                except Exception as e:
                    self.logger.warning(f"Failed to read landcover classes from {lc_path}: {e}")

        return None

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

                n_size = ds.sizes[n_dim]

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

                n_size = ds.sizes[n_dim]
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

                # Rename land or landclass dimension to 'NGRU' (MESH's expected name)
                old_lc_dim = 'land' if 'land' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else None
                if old_lc_dim and old_lc_dim != 'NGRU':
                    self.logger.info(f"Renaming dimension '{old_lc_dim}' to 'NGRU'")
                    ds = ds.rename({old_lc_dim: 'NGRU'})
                    modified = True

                # Ensure GRU variable exists and is on (subbasin, NGRU)
                if 'GRU' in ds:
                    # Correct dimension names if needed
                    if ds['GRU'].dims != (target_n_dim, 'NGRU'):
                        ds['GRU'] = ((target_n_dim, 'NGRU'), ds['GRU'].values, ds['GRU'].attrs)
                        modified = True
                    ds['GRU'].attrs['grid_mapping'] = 'crs'

                    # For lumped mode, force single GRU only when explicitly enabled
                    # or when spatial mode implies lumped behavior.
                    force_single_gru = self._should_force_single_gru()
                    if force_single_gru and 'NGRU' in ds.dims and ds.sizes['NGRU'] > 2:
                        self.logger.info(
                            f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                            f"(NGRU=2 for MESH off-by-one workaround)"
                        )
                        # Create NGRU=2: [0.998, 0.002] so MESH reads 1 GRU
                        gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
                        if n_size > 1:
                            gru_data = np.tile(gru_data, (n_size, 1))
                        vars_to_drop = [v for v in ds.data_vars if 'NGRU' in ds[v].dims]
                        if vars_to_drop:
                            ds = ds.drop_vars(vars_to_drop, errors='ignore')
                        if 'NGRU' in ds.coords:
                            ds = ds.drop_vars('NGRU', errors='ignore')
                        ds['GRU'] = xr.DataArray(
                            gru_data,
                            dims=[target_n_dim, 'NGRU'],
                            attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
                        )
                        modified = True

                elif 'NGRU' in ds:
                    # If NGRU variable is present but GRU is missing, treat it as GRU fractions
                    self.logger.info("Found 'NGRU' variable without GRU; renaming to GRU")
                    ds['GRU'] = ((target_n_dim, 'NGRU'), ds['NGRU'].values, ds['NGRU'].attrs)
                    ds = ds.drop_vars('NGRU')
                    ds['GRU'].attrs['grid_mapping'] = 'crs'
                    modified = True

                    # For lumped mode, force single GRU only when explicitly enabled
                    # or when spatial mode implies lumped behavior.
                    force_single_gru = self._should_force_single_gru()
                    if force_single_gru and 'NGRU' in ds.dims and ds.sizes['NGRU'] > 2:
                        self.logger.info(
                            f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                            f"(NGRU=2 for MESH off-by-one workaround)"
                        )
                        # Create NGRU=2: [0.998, 0.002] so MESH reads 1 GRU
                        gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
                        if n_size > 1:
                            gru_data = np.tile(gru_data, (n_size, 1))
                        vars_to_drop = [v for v in ds.data_vars if 'NGRU' in ds[v].dims]
                        if vars_to_drop:
                            ds = ds.drop_vars(vars_to_drop, errors='ignore')
                        if 'NGRU' in ds.coords:
                            ds = ds.drop_vars('NGRU', errors='ignore')
                        ds['GRU'] = xr.DataArray(
                            gru_data,
                            dims=[target_n_dim, 'NGRU'],
                            attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
                        )
                        modified = True
                else:
                    # Create GRU variable if it doesn't exist (e.g., for lumped mode)
                    # For lumped mode, we have 1 subbasin and 1 landclass, with 100% coverage
                    self.logger.info("Creating GRU variable for lumped mode (1 landclass, 100% coverage)")
                    n_landclass = 1  # Single landcover class for lumped mode
                    landclass_data = np.ones((n_size, n_landclass), dtype=np.float64)
                    ds['GRU'] = ((target_n_dim, 'NGRU'), landclass_data, {
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

                # Trim zero-fraction GRUs before setting coordinates
                # Meshflow may create extra GRU entries with zero fractions
                # Skip for elevation band mode — the padding column is required for MESH off-by-one
                _sub_grid_trim = self.config_dict.get('SUB_GRID_DISCRETIZATION', 'GRUS')
                _skip_trim = isinstance(_sub_grid_trim, str) and _sub_grid_trim.lower() == 'elevation'
                if 'NGRU' in ds.dims and 'GRU' in ds and not _skip_trim:
                    gru_vals = ds['GRU'].values
                    # Sum GRU fractions across subbasins for each GRU
                    if gru_vals.ndim == 2:
                        gru_sums = gru_vals.sum(axis=0)
                    else:
                        gru_sums = gru_vals

                    # Find non-zero GRUs
                    nonzero_mask = gru_sums > 0.001
                    n_nonzero = nonzero_mask.sum()
                    n_original = len(nonzero_mask)

                    if n_nonzero < n_original:
                        self.logger.info(
                            f"Trimming {n_original - n_nonzero} zero-fraction GRU(s) "
                            f"(keeping {n_nonzero} of {n_original})"
                        )
                        # Trim NGRU dimension
                        ds = ds.isel(NGRU=nonzero_mask)

                        # Renormalize GRU fractions to sum to 1.0
                        if 'GRU' in ds:
                            gru_trimmed = ds['GRU'].values
                            if gru_trimmed.ndim == 2:
                                row_sums = gru_trimmed.sum(axis=1, keepdims=True)
                                row_sums = np.where(row_sums == 0, 1, row_sums)
                                ds['GRU'] = (ds['GRU'].dims, gru_trimmed / row_sums)
                            else:
                                total = gru_trimmed.sum()
                                if total > 0:
                                    ds['GRU'] = (ds['GRU'].dims, gru_trimmed / total)
                        modified = True

                # MESH does NOT expect a coordinate variable for the NGRU dimension.
                # Having NGRU(NGRU) confuses MESH - it tries to map it as a data variable
                # and reports "cannot be mapped from elemental dimension 'GRU'".
                # Remove any NGRU coordinate variable if present.
                if 'NGRU' in ds.dims and 'NGRU' in ds.coords:
                    ds = ds.drop_vars('NGRU')
                    self.logger.info("Removed NGRU coordinate variable (MESH expects dimension only)")
                    modified = True

                # FINAL CHECK: Enforce single GRU for lumped mode (after all other processing)
                # MESH has an off-by-one issue: it reads NGRU-1 GRUs.
                # For MESH to see 1 GRU, we need NGRU=2 with a padding column.
                # Skip for elevation band mode to preserve multi-subbasin structure.
                force_single_gru = self._should_force_single_gru()
                if force_single_gru and 'NGRU' in ds.dims and 'GRU' in ds and ds.sizes['NGRU'] > 1:
                    self.logger.info(
                        f"Enforcing lumped mode: collapsing {ds.sizes['NGRU']} GRUs to 1 "
                        f"(set MESH_FORCE_SINGLE_GRU=false to use multiple GRUs)"
                    )
                    # Create NGRU=2: [0.998, 0.002] so MESH reads 1 GRU (due to off-by-one)
                    gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
                    if n_size > 1:
                        gru_data = np.tile(gru_data, (n_size, 1))
                    # Drop all vars with NGRU dimension
                    vars_to_drop = [v for v in ds.data_vars if 'NGRU' in ds[v].dims]
                    if vars_to_drop:
                        ds = ds.drop_vars(vars_to_drop, errors='ignore')
                    if 'NGRU' in ds.coords:
                        ds = ds.drop_vars('NGRU', errors='ignore')
                    ds['GRU'] = xr.DataArray(
                        gru_data,
                        dims=[target_n_dim, 'NGRU'],
                        attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
                    )
                    modified = True

                # Ensure GRU has proper attributes that MESH expects
                if 'GRU' in ds:
                    ds['GRU'].attrs.update({
                        'long_name': 'Group Response Unit',
                        'standard_name': 'GRU',
                        'grid_mapping': 'crs'
                    })
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
                    lc_var = 'NGRU' if 'NGRU' in ds else 'GRU' if 'GRU' in ds else None
                    lc_dim = 'NGRU' if 'NGRU' in ds.dims else 'NGRU' if 'NGRU' in ds.dims else None

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
                    # Ensure landclass is written as float64
                    encoding = {}
                    if 'NGRU' in ds.coords:
                        encoding['NGRU'] = {'dtype': 'float64'}
                    ds.to_netcdf(temp_path, encoding=encoding)
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

                # MESH does NOT expect a coordinate variable for the NGRU dimension.
                # Having NGRU(NGRU) confuses MESH - it tries to map it as a data variable.
                # Remove any NGRU coordinate variable if present.
                encoding: dict[str, dict[str, bool]] = {}
                if 'NGRU' in ds_new.dims and 'NGRU' in ds_new.coords:
                    ds_new = ds_new.drop_vars('NGRU')

                # Enforce single GRU only when explicitly enabled or in lumped mode.
                force_single_gru = self._should_force_single_gru()

                # Check GRU shape directly since dimension checks can be tricky
                ngru_count = 1
                if 'GRU' in ds_new:
                    gru_shape = ds_new['GRU'].shape
                    ngru_count = gru_shape[-1] if len(gru_shape) >= 2 else 1

                if force_single_gru and ngru_count > 1:
                    n_size = ds_new.sizes[n_dim] if n_dim else 1
                    self.logger.info(
                        f"Enforcing lumped mode: collapsing {ngru_count} GRUs to 1 "
                        f"(set MESH_FORCE_SINGLE_GRU=false to use multiple GRUs)"
                    )
                    # MESH has an off-by-one issue: it reads NGRU-1 GRUs.
                    # For MESH to see 1 GRU, we need NGRU=2: [0.998, 0.002]
                    # The second column is padding with a small value.
                    gru_data = np.array([[0.998, 0.002]], dtype=np.float64)
                    if n_size > 1:
                        gru_data = np.tile(gru_data, (n_size, 1))

                    # Drop all vars with NGRU dimension and the NGRU coord if present
                    vars_to_drop = [v for v in ds_new.data_vars if 'NGRU' in ds_new[v].dims]
                    if vars_to_drop:
                        ds_new = ds_new.drop_vars(vars_to_drop, errors='ignore')
                    if 'NGRU' in ds_new.coords:
                        ds_new = ds_new.drop_vars('NGRU', errors='ignore')

                    # Create new GRU with NGRU=2 for MESH off-by-one workaround
                    ds_new['GRU'] = xr.DataArray(
                        gru_data,
                        dims=[n_dim, 'NGRU'],
                        attrs={'long_name': 'Group Response Unit', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
                    )
                    self.logger.info(f"Created GRU with shape {gru_data.shape} (NGRU=2 for MESH to see 1)")

                # Ensure GRU has proper attributes that MESH expects
                if 'GRU' in ds_new:
                    ds_new['GRU'].attrs.update({
                        'long_name': 'Group Response Unit',
                        'standard_name': 'GRU',
                        'grid_mapping': 'crs'
                    })

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path, encoding=encoding)
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

    def _as_bool(self, value: Any, default: bool = False) -> bool:
        """Parse a truthy/falsey config value with a safe default."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ('true', '1', 'yes', 'y', 'on'):
                return True
            if val in ('false', '0', 'no', 'n', 'off'):
                return False
        return default

    def _should_force_single_gru(self) -> bool:
        """Determine whether to collapse GRUs to a single class.

        Logic:
        - If MESH_FORCE_SINGLE_GRU is explicitly set (and not "default"), honor it.
        - Otherwise, auto-enable only for lumped/point spatial modes.
        - Always disable for elevation-band discretization.
        """
        raw = None
        try:
            raw = self.config_dict.get('MESH_FORCE_SINGLE_GRU', None)
        except Exception:
            raw = None

        if raw is not None and raw != 'default':
            force = self._as_bool(raw, default=False)
        else:
            spatial_mode = ''
            domain_method = ''
            try:
                spatial_mode = str(self.config_dict.get('MESH_SPATIAL_MODE', 'auto')).lower()
                domain_method = str(self.config_dict.get('DOMAIN_DEFINITION_METHOD', '')).lower()
            except Exception:
                pass
            force = spatial_mode in ('lumped', 'point') or domain_method in ('lumped', 'point')

        try:
            sub_grid = self.config_dict.get('SUB_GRID_DISCRETIZATION', 'GRUS')
            if isinstance(sub_grid, str) and sub_grid.lower() == 'elevation':
                return False
        except Exception:
            pass

        return force

    def convert_to_elevation_band_grus(self, hru_shapefile: Path) -> tuple:
        """Convert DDB to use elevation bands as GRUs instead of landcover classes.

        This method reads the elevation band HRU shapefile created during discretization
        and rebuilds the DDB GRU variable to have one GRU per elevation band.

        Args:
            hru_shapefile: Path to the elevation band HRU shapefile

        Returns:
            Tuple of (n_elev_bands, elevation_info) where elevation_info is a list of
            dicts with 'elevation', 'fraction' for each band. Returns (0, []) on failure.
        """
        if not hru_shapefile.exists():
            self.logger.warning(f"Elevation band shapefile not found: {hru_shapefile}")
            return 0, []

        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return 0, []

        try:
            # Read elevation band HRU shapefile
            gdf = gpd.read_file(hru_shapefile)
            self.logger.info(f"Read elevation band shapefile with {len(gdf)} HRUs")

            # Get area column name
            area_col = 'HRU_area' if 'HRU_area' in gdf.columns else 'area'
            if area_col not in gdf.columns:
                # Calculate areas
                gdf['HRU_area'] = gdf.geometry.area
                area_col = 'HRU_area'

            # Calculate total area and fraction for each elevation band
            total_area = gdf[area_col].sum()
            n_elev_bands = len(gdf)

            elev_fractions = (gdf[area_col] / total_area).values
            self.logger.info(
                f"Elevation band fractions: {[f'{f:.3f}' for f in elev_fractions]}"
            )

            # Get mean elevation for each band
            elev_col = 'avg_elevcl' if 'avg_elevcl' in gdf.columns else 'elev_mean'
            if elev_col in gdf.columns:
                mean_elevs = gdf[elev_col].values
            else:
                # Default elevations if not available
                mean_elevs = [1500 + i * 400 for i in range(n_elev_bands)]

            self.logger.info(
                f"Elevation band means: {[f'{e:.0f}m' for e in mean_elevs]}"
            )

            # Build elevation info for CLASS block creation
            elevation_info = [
                {'elevation': float(mean_elevs[i]), 'fraction': float(elev_fractions[i])}
                for i in range(n_elev_bands)
            ]

            # Update DDB with elevation band GRUs
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("Could not determine spatial dimension")
                    return 0, []

                n_size = int(ds.sizes[n_dim])

                # Create new GRU array with elevation bands
                # MESH reads NGRU-1, so we need n_elev_bands + 1 columns
                n_gru_cols = n_elev_bands + 1
                new_gru = np.zeros((n_size, n_gru_cols), dtype=np.float64)

                # Set elevation band fractions (first n_elev_bands columns)
                for i in range(n_elev_bands):
                    new_gru[:, i] = elev_fractions[i]

                # Create new dataset
                ds_new = ds.drop_vars(['GRU'] if 'GRU' in ds else [], errors='ignore')
                if 'NGRU' in ds_new.dims:
                    ds_new = ds_new.drop_dims('NGRU')

                # Add new GRU variable
                ds_new['GRU'] = xr.DataArray(
                    new_gru,
                    dims=[n_dim, 'NGRU'],
                    attrs={'long_name': 'Elevation band fractions'}
                )

                # Save updated DDB
                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

            self.logger.info(
                f"Converted DDB to {n_elev_bands} elevation band GRUs "
                f"(NGRU={n_gru_cols}, MESH reads {n_elev_bands})"
            )
            return n_elev_bands, elevation_info

        except Exception as e:
            self.logger.warning(f"Failed to convert to elevation band GRUs: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0, []

    def convert_to_multi_subbasin_elevation_bands(self, hru_shapefile: Path) -> tuple:
        """Convert DDB from 1 subbasin × N GRUs to N subbasins × (N+1) GRUs.

        Each elevation band becomes its own subbasin with its own forcing,
        enabling temperature lapsing. Uses an identity GRU matrix so each
        subbasin has exactly one active GRU.

        Args:
            hru_shapefile: Path to the elevation band HRU shapefile

        Returns:
            Tuple of (n_elev_bands, elevation_info) where elevation_info is a list of
            dicts with 'elevation', 'fraction' for each band. Returns (0, []) on failure.
        """
        if not hru_shapefile.exists():
            self.logger.warning(f"Elevation band shapefile not found: {hru_shapefile}")
            return 0, []

        if not self.ddb_path.exists():
            self.logger.warning("MESH_drainage_database.nc not found")
            return 0, []

        try:
            # Read elevation band HRU shapefile
            gdf = gpd.read_file(hru_shapefile)
            self.logger.info(f"Read elevation band shapefile with {len(gdf)} HRUs")

            # Get area column name
            area_col = 'HRU_area' if 'HRU_area' in gdf.columns else 'area'
            if area_col not in gdf.columns:
                gdf['HRU_area'] = gdf.geometry.area
                area_col = 'HRU_area'

            total_area_from_shp = gdf[area_col].sum()
            n_bands = len(gdf)
            band_fractions = (gdf[area_col] / total_area_from_shp).values

            # Get mean elevation for each band
            elev_col = 'avg_elevcl' if 'avg_elevcl' in gdf.columns else 'elev_mean'
            if elev_col in gdf.columns:
                mean_elevs = gdf[elev_col].values
            else:
                mean_elevs = np.array([1500 + i * 400 for i in range(n_bands)])

            self.logger.info(
                f"Elevation bands: {n_bands} bands, "
                f"elevations={[f'{e:.0f}m' for e in mean_elevs]}, "
                f"fractions={[f'{f:.3f}' for f in band_fractions]}"
            )

            elevation_info = [
                {'elevation': float(mean_elevs[i]), 'fraction': float(band_fractions[i])}
                for i in range(n_bands)
            ]

            # Read existing DDB (single subbasin)
            with xr.open_dataset(self.ddb_path) as ds:
                n_dim = self._get_spatial_dim(ds)
                if not n_dim:
                    self.logger.warning("Could not determine spatial dimension")
                    return 0, []

                # Get original values from the single subbasin
                orig_total_area = float(ds['GridArea'].values[0]) if 'GridArea' in ds else total_area_from_shp
                orig_lat = float(ds['lat'].values[0]) if 'lat' in ds else 0.0
                orig_lon = float(ds['lon'].values[0]) if 'lon' in ds else 0.0

                # Collect other 1D variables to replicate
                chnl_slope = float(ds['ChnlSlope'].values[0]) if 'ChnlSlope' in ds else 0.001
                chnl_length = float(ds['ChnlLength'].values[0]) if 'ChnlLength' in ds else 1000.0
                # Check for optional variables
                has_slope = 'Slope' in ds
                slope_val = float(ds['Slope'].values[0]) if has_slope else chnl_slope
                has_area = 'Area' in ds

            # Build new multi-subbasin DDB
            target_dim = 'subbasin'

            # GRU matrix: identity pattern with padding column for MESH off-by-one
            # MESH reads NGRU-1 GRUs, so we need n_bands + 1 columns
            n_gru_cols = n_bands + 1
            gru_data = np.zeros((n_bands, n_gru_cols), dtype=np.float64)
            for i in range(n_bands):
                gru_data[i, i] = 1.0  # Each subbasin has exactly one active GRU

            # Spatial arrays
            grid_area = orig_total_area * band_fractions
            lat_arr = np.full(n_bands, orig_lat, dtype=np.float64)
            lon_arr = np.full(n_bands, orig_lon, dtype=np.float64)

            # Rank and Next: all outlets in noroute mode
            rank_arr = np.arange(1, n_bands + 1, dtype=np.int32)
            next_arr = np.zeros(n_bands, dtype=np.int32)

            # Build dataset
            ds_new = xr.Dataset()

            # CRS
            ds_new['crs'] = xr.DataArray(
                np.array(0, dtype=np.int32),
                attrs={
                    'grid_mapping_name': 'latitude_longitude',
                    'semi_major_axis': 6378137.0,
                    'inverse_flattening': 298.257223563,
                    'longitude_of_prime_meridian': 0.0
                }
            )

            # Coordinate
            ds_new[target_dim] = xr.DataArray(
                np.arange(1, n_bands + 1, dtype=np.int32),
                dims=[target_dim],
                attrs={'long_name': 'Grid index', 'units': '1'}
            )

            # GRU
            ds_new['GRU'] = xr.DataArray(
                gru_data,
                dims=[target_dim, 'NGRU'],
                attrs={'long_name': 'Elevation band fractions', 'standard_name': 'GRU', 'grid_mapping': 'crs'}
            )

            # 1D variables replicated/scaled across subbasins
            ds_new['GridArea'] = xr.DataArray(
                grid_area, dims=[target_dim],
                attrs={'long_name': 'Grid area', 'units': 'm2', 'grid_mapping': 'crs'}
            )
            ds_new['lat'] = xr.DataArray(
                lat_arr, dims=[target_dim],
                attrs={'long_name': 'latitude', 'standard_name': 'latitude',
                       'units': 'degrees_north', 'grid_mapping': 'crs'}
            )
            ds_new['lon'] = xr.DataArray(
                lon_arr, dims=[target_dim],
                attrs={'long_name': 'longitude', 'standard_name': 'longitude',
                       'units': 'degrees_east', 'grid_mapping': 'crs'}
            )
            ds_new['Rank'] = xr.DataArray(
                rank_arr, dims=[target_dim],
                attrs={'long_name': 'Cell rank in topological order', 'units': '1'}
            )
            ds_new['Next'] = xr.DataArray(
                next_arr, dims=[target_dim],
                attrs={'long_name': 'Downstream cell rank', 'units': '1'}
            )
            ds_new['ChnlSlope'] = xr.DataArray(
                np.full(n_bands, chnl_slope, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel slope', 'units': '1', 'grid_mapping': 'crs'}
            )
            ds_new['ChnlLength'] = xr.DataArray(
                np.full(n_bands, chnl_length, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel length', 'units': 'm', 'grid_mapping': 'crs'}
            )
            ds_new['AL'] = xr.DataArray(
                np.sqrt(grid_area), dims=[target_dim],
                attrs={'long_name': 'Characteristic length of grid', 'units': 'm', 'grid_mapping': 'crs'}
            )
            ds_new['DA'] = xr.DataArray(
                grid_area.copy(), dims=[target_dim],
                attrs={'long_name': 'Drainage area', 'units': 'm**2', 'grid_mapping': 'crs'}
            )
            ds_new['IAK'] = xr.DataArray(
                np.ones(n_bands, dtype=np.int32), dims=[target_dim],
                attrs={'long_name': 'River class', 'units': '1'}
            )
            ds_new['IREACH'] = xr.DataArray(
                np.zeros(n_bands, dtype=np.int32), dims=[target_dim],
                attrs={'long_name': 'Reservoir ID', 'units': '1'}
            )

            if has_slope:
                ds_new['Slope'] = xr.DataArray(
                    np.full(n_bands, slope_val, dtype=np.float64), dims=[target_dim],
                    attrs={'long_name': 'Mean basin slope', 'units': '1', 'grid_mapping': 'crs'}
                )
            if has_area:
                ds_new['Area'] = xr.DataArray(
                    grid_area.copy(), dims=[target_dim],
                    attrs={'long_name': 'Grid area', 'units': 'm2', 'grid_mapping': 'crs'}
                )
            ds_new['ChnlLen'] = xr.DataArray(
                np.full(n_bands, chnl_length, dtype=np.float64), dims=[target_dim],
                attrs={'long_name': 'Channel length', 'units': 'm', 'grid_mapping': 'crs'}
            )

            # Save
            temp_path = self.ddb_path.with_suffix('.tmp.nc')
            ds_new.to_netcdf(temp_path)
            os.replace(temp_path, self.ddb_path)

            self.logger.info(
                f"Converted DDB to multi-subbasin elevation bands: "
                f"subbasin={n_bands}, NGRU={n_gru_cols}, "
                f"total GridArea={grid_area.sum():.0f} m² "
                f"(original={orig_total_area:.0f} m²)"
            )
            return n_bands, elevation_info

        except Exception as e:
            self.logger.warning(f"Failed to convert to multi-subbasin elevation bands: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0, []
