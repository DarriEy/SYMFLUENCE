# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
High Resolution Rapid Refresh (HRRR) data acquisition from AWS S3.

Provides automated download and processing of HRRR atmospheric forcing data
with spatial subsetting, coordinate transformation, and NetCDF export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler
from ..mixins.retry import RetryMixin


@R.acquisition_handlers.add('HRRR')
class HRRRAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    Download and process High Resolution Rapid Refresh (HRRR) atmospheric forcing data.

    HRRR is a NOAA operational weather model providing 3 km resolution hourly forecasts
    for the continental United States. This acquirer accesses analysis (0-hour forecast)
    data from AWS S3 Zarr archives with Lambert Conformal Conic projection handling.

    Dataset Characteristics:
        Source: NOAA/NCEP High Resolution Rapid Refresh
        Spatial Coverage: CONUS (Continental United States)
        Spatial Resolution: ~3 km (Lambert Conformal Conic projection)
        Temporal Coverage: 2014-07-30 to near-present (operational)
        Temporal Resolution: Hourly (analysis fields at top of hour)
        Format: Zarr (one archive per hour, organized by variable/level)
        S3 Bucket: hrrrzarr
        Access: Anonymous (no AWS credentials required)
        Update Frequency: Hourly (operational, ~30-60 min delay)

    Variables Available:
        - TMP (2m_above_ground): Air temperature at 2m (K)
        - SPFH (2m_above_ground): Specific humidity at 2m (kg/kg)
        - PRES (surface): Surface pressure (Pa)
        - UGRD (10m_above_ground): U-component wind at 10m (m/s)
        - VGRD (10m_above_ground): V-component wind at 10m (m/s)
        - DSWRF (surface): Downward shortwave radiation flux (W/m²)
        - DLWRF (surface): Downward longwave radiation flux (W/m²)

    Coordinate System:
        Native Projection: Lambert Conformal Conic
            - Reference latitude: 38.5°N
            - Reference longitude: -97.5°W (central US)
            - Standard parallel: 38.5°N
            - False easting/northing: 0, 0
            - Ellipsoid radius: 6371229 m

        Transformation:
            - Native coordinates: projection_x_coordinate, projection_y_coordinate (meters)
            - Geographic coordinates: latitude, longitude (degrees)
            - Automatic transformation using pyproj when needed
            - 2D coordinate arrays (curvilinear grid)

    Workflow:
        1. **S3 Initialization**: Anonymous S3 filesystem connection
        2. **Variable Selection**: Map variables to required levels
        3. **Bounding Box**: Use HRRR_BOUNDING_BOX_COORDS or default bbox
        4. **Hourly Iteration**: Loop through date range hour-by-hour
        5. **Variable Merging**: Open and merge multiple variable Zarr stores
        6. **Spatial Masking**: Compute bbox mask on first successful hour
        7. **Spatial Subset**: Apply mask to all subsequent hours
        8. **Temporal Concatenation**: Merge all hourly datasets
        9. **Time Resampling** (optional): Subsample to N-hourly intervals
        10. **Coordinate Transform**: Convert Lambert Conformal to lat/lon if needed
        11. **Type Conversion**: Float16 → Float32 for NetCDF compatibility
        12. **NetCDF Export**: Save combined dataset

    Zarr Archive Structure:
        Path pattern: hrrrzarr/sfc/{YYYYMMDD}/{YYYYMMDD}_{HH}z_anl.zarr/{level}/{var}/{level}
        Alternative path: hrrrzarr/sfc/{YYYYMMDD}/{YYYYMMDD}_{HH}z_anl.zarr/{level}/{var}

        Example:
            hrrrzarr/sfc/20220101/20220101_00z_anl.zarr/2m_above_ground/TMP/2m_above_ground

    Spatial Subsetting Strategy:
        - The hrrrzarr variable groups carry NO latitude/longitude coordinates,
          only 1-D projection_x/y_coordinate arrays on the fixed HRRR Lambert
          Conformal Conic grid. The bbox is therefore projected to LCC x/y with
          pyproj and converted to an index window BEFORE any chunk is fetched.
        - Store x/y slice indices for reuse across hours
        - Lazy loading: Only download windowed region
        - Typical reduction: 99%+ for small basins
        - HARD RULE: if no spatial window can be determined, acquisition RAISES
          instead of silently downloading the full CONUS domain (~1.3 GB/day)

    Error Handling:
        - Hourly failures silently skipped (operational gaps)
        - Variable failures silently skipped (not all vars in all archives)
        - At least one hour required (raises ValueError if all fail)
        - S3 connection errors propagate to caller

    Configuration Requirements:
        Required (inherited from BaseAcquisitionHandler):
            - DOMAIN_NAME: Basin identifier
            - EXPERIMENT_TIME_START: Download start (YYYY-MM-DD HH:MM)
            - EXPERIMENT_TIME_END: Download end (YYYY-MM-DD HH:MM)
            - DOMAIN_BOUNDING_BOX: Spatial extent [lon_min, lat_min, lon_max, lat_max]

        Optional:
            - HRRR_BOUNDING_BOX_COORDS: Override bbox for HRRR (larger region)
            - HRRR_VARS: List of variables to download (subset of default)
            - HRRR_TIME_STEP_HOURS: Resample to N-hourly (1=hourly, 3=3-hourly, etc.)

    Output Format:
        - Filename: {DOMAIN_NAME}_HRRR_hourly_{YYYYMMDD}-{YYYYMMDD}.nc
        - Format: NetCDF4
        - Dimensions: time, projection_y_coordinate, projection_x_coordinate
        - Coordinates: latitude (2D), longitude (2D)
        - Data type: Float32 (converted from Float16)

    Performance Notes:
        - Zarr enables efficient spatial subsetting (no full file download)
        - Hourly iteration: ~0.5-2 seconds per hour
        - S3 transfer speed: Variable (10-100 MB/s)
        - Typical download: ~50-150 MB per day for small basin
        - Memory usage: One hour in memory at a time
        - Processing time: ~2-10 minutes for 1 week period
        - Coordinate transformation: ~5-15 seconds (pyproj overhead)

    Operational Gaps:
        - HRRR data occasionally missing for specific hours
        - Missing hours silently skipped (no error raised)
        - Archive reorganizations may change Zarr path structure
        - Fallback paths attempted for robustness

    Example:
        >>> config = {
        ...     'DOMAIN_NAME': 'boulder_creek',
        ...     'EXPERIMENT_TIME_START': '2022-01-01 00:00',
        ...     'EXPERIMENT_TIME_END': '2022-01-07 23:00',
        ...     'DOMAIN_BOUNDING_BOX': [-105.5, 40.0, -105.0, 40.3],
        ...     'HRRR_TIME_STEP_HOURS': 1  # Hourly
        ... }
        >>> acquirer = HRRRAcquirer(config, logger)
        >>> output = acquirer.download(Path('./forcing/raw'))
        >>> print(output)
        ./forcing/raw/boulder_creek_HRRR_hourly_20220101-20220107.nc
        # Size: ~120 MB for 7 days hourly
        # Variables: TMP, SPFH, PRES, UGRD, VGRD, DSWRF, DLWRF

    Notes:
        - HRRR is operational; historical data availability starts 2014-07-30
        - Analysis fields (0-hour forecast) used, not forecast hours
        - Lambert Conformal projection preserved in coordinates
        - Geographic lat/lon added as 2D auxiliary coordinates
        - Float16 compression in Zarr converted to Float32 for NetCDF
        - Suitable for high-resolution basins (<1000 km²)
        - For large regions, consider AORC or CONUS404 instead

    See Also:
        - data.preprocessing.dataset_handlers.hrrr_utils.HRRRHandler: Variable processing
        - data.acquisition.base.BaseAcquisitionHandler: Base acquisition interface
        - data.acquisition.registry.AcquisitionRegistry: Handler registration
    """

    #: HRRR native Lambert Conformal Conic projection (fixed, published grid).
    HRRR_LCC_PROJ = (
        "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
        "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
    )

    @staticmethod
    def _spatial_dims(ds: xr.Dataset) -> tuple:
        """Return the (y_dim, x_dim) names of the spatial dimensions.

        hrrrzarr groups use ``projection_y/x_coordinate``; ``y``/``x`` is
        accepted for grids that have already been renamed.

        Raises:
            ValueError: If no recognized spatial dimensions are present.
        """
        for y_dim, x_dim in (
            ("projection_y_coordinate", "projection_x_coordinate"),
            ("y", "x"),
        ):
            if y_dim in ds.dims and x_dim in ds.dims:
                return y_dim, x_dim
        raise ValueError(
            f"Cannot identify HRRR spatial dimensions in {list(ds.dims)}; "
            "expected projection_y/x_coordinate (or y/x)."
        )

    def _lcc_index_window(self, proj_y: np.ndarray, proj_x: np.ndarray, bbox: dict) -> tuple:
        """Compute the (y_slice, x_slice) index window of *bbox* on the LCC grid.

        Projects a densified bbox boundary to HRRR Lambert Conformal Conic
        x/y with pyproj (densified because lines of constant lat/lon curve in
        LCC space), buffers by two grid cells, and converts the LCC extent to
        index slices on the 1-D projection coordinate arrays. This happens
        BEFORE any data chunk is fetched, so only the windowed chunks are
        ever downloaded.

        Args:
            proj_y: 1-D projection_y_coordinate values (meters)
            proj_x: 1-D projection_x_coordinate values (meters)
            bbox: dict with lat_min/lat_max/lon_min/lon_max

        Returns:
            (y_slice, x_slice) index slices

        Raises:
            ValueError: If the bbox does not intersect the HRRR grid.
        """
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4326", self.HRRR_LCC_PROJ, always_xy=True)

        # Densify the bbox boundary: in LCC space the x/y extremes of a
        # geographic rectangle can fall on edge midpoints, not corners.
        n = 25
        lons_edge = np.linspace(bbox["lon_min"], bbox["lon_max"], n)
        lats_edge = np.linspace(bbox["lat_min"], bbox["lat_max"], n)
        boundary_lon = np.concatenate([
            lons_edge, lons_edge,
            np.full(n, bbox["lon_min"]), np.full(n, bbox["lon_max"]),
        ])
        boundary_lat = np.concatenate([
            np.full(n, bbox["lat_min"]), np.full(n, bbox["lat_max"]),
            lats_edge, lats_edge,
        ])
        bx, by = tr.transform(boundary_lon, boundary_lat)

        # Two-cell buffer, like the grid spacing itself derived from the coords
        dx = float(np.median(np.abs(np.diff(proj_x)))) or 3000.0
        dy = float(np.median(np.abs(np.diff(proj_y)))) or 3000.0
        x_min, x_max = np.min(bx) - 2 * dx, np.max(bx) + 2 * dx
        y_min, y_max = np.min(by) - 2 * dy, np.max(by) + 2 * dy

        in_x = np.where((proj_x >= x_min) & (proj_x <= x_max))[0]
        in_y = np.where((proj_y >= y_min) & (proj_y <= y_max))[0]
        if in_x.size > 0 and in_y.size > 0:
            return (slice(in_y.min(), in_y.max() + 1), slice(in_x.min(), in_x.max() + 1))

        # Bbox smaller than one grid cell: take the nearest cell to its
        # center — but only if the center actually lies on the grid.
        cx, cy = tr.transform(
            (bbox["lon_min"] + bbox["lon_max"]) / 2,
            (bbox["lat_min"] + bbox["lat_max"]) / 2,
        )
        on_grid = (
            proj_x.min() - dx <= cx <= proj_x.max() + dx
            and proj_y.min() - dy <= cy <= proj_y.max() + dy
        )
        if not on_grid:
            raise ValueError(
                f"Bounding box {bbox} does not intersect the HRRR CONUS grid "
                "(HRRR covers the continental United States only)."
            )
        ix = int(np.abs(proj_x - cx).argmin())
        iy = int(np.abs(proj_y - cy).argmin())
        self.logger.info(
            "Bbox smaller than HRRR grid cell; using nearest grid point at "
            f"x={proj_x[ix]:.0f} m, y={proj_y[iy]:.0f} m"
        )
        return (slice(iy, iy + 1), slice(ix, ix + 1))

    def _determine_xy_window(self, ds: xr.Dataset, bbox: dict) -> tuple:
        """Determine the spatial index window of *bbox* for a HRRR dataset.

        Preferred route: project the bbox to the fixed HRRR LCC grid using
        the 1-D projection coordinates the hrrrzarr groups carry (the groups
        have NO latitude/longitude, so a geographic mask is impossible
        there). Falls back to a 2-D latitude/longitude mask when present.

        Raises:
            ValueError: If no spatial window can be determined. Acquisition
                must NEVER silently fall back to the full CONUS domain.
        """
        y_dim, x_dim = self._spatial_dims(ds)

        if (
            y_dim in ds.coords and x_dim in ds.coords
            and ds[y_dim].ndim == 1 and ds[x_dim].ndim == 1
        ):
            return self._lcc_index_window(ds[y_dim].values, ds[x_dim].values, bbox)

        if "latitude" in ds.coords and "longitude" in ds.coords:
            mask = (
                (ds.latitude >= bbox["lat_min"])
                & (ds.latitude <= bbox["lat_max"])
                & (ds.longitude >= bbox["lon_min"])
                & (ds.longitude <= bbox["lon_max"])
            )
            iy, ix = np.where(mask)
            if len(iy) > 0:
                return (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))
            # Bbox smaller than grid resolution; find nearest grid point
            center_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
            center_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
            dist = (ds.latitude - center_lat) ** 2 + (ds.longitude - center_lon) ** 2
            min_idx = np.unravel_index(dist.values.argmin(), dist.shape)
            self.logger.info(
                "Bbox smaller than HRRR grid cell; using nearest grid point at "
                f"lat={float(ds.latitude.values[min_idx]):.4f}, "
                f"lon={float(ds.longitude.values[min_idx]):.4f}"
            )
            return (slice(min_idx[0], min_idx[0] + 1), slice(min_idx[1], min_idx[1] + 1))

        raise ValueError(
            "Cannot determine a spatial window for the HRRR request: the "
            "dataset has neither 1-D projection_y/x coordinates nor "
            "latitude/longitude coordinates. Refusing to download the full "
            "CONUS domain (~1.3 GB/day)."
        )

    def download(self, output_dir: Path) -> Path:
        """
        Download HRRR data from AWS S3 Zarr archives with projection handling.

        Iterates hour-by-hour through the date range, downloading variables from
        S3 Zarr archives, performing spatial subsetting, merging variables, and
        transforming coordinates from Lambert Conformal Conic to geographic.

        Args:
            output_dir: Directory to save downloaded NetCDF file

        Returns:
            Path to downloaded NetCDF file:
                Format: {output_dir}/{domain_name}_HRRR_hourly_{YYYYMMDD}-{YYYYMMDD}.nc

        Raises:
            ValueError: If no HRRR data successfully downloaded for any hour
            Exception: If S3 connection fails or coordinate transformation errors

        Process:
            1. Initialize S3 filesystem (anonymous)
            2. Define variable-level mapping (7 variables across 3 levels)
            3. Parse bounding box (HRRR-specific or default)
            4. For each hour in date range:
               a. Construct S3 Zarr paths for each variable
               b. Attempt to open primary and fallback paths
               c. Merge successfully loaded variables
               d. On first success: compute spatial mask and x/y slices
               e. Apply spatial subset to current hour
               f. Append to dataset list
            5. Concatenate all hours along time dimension
            6. Optional: Resample to N-hourly intervals
            7. If projection coordinates only: transform to lat/lon
            8. Convert Float16 to Float32
            9. Export to NetCDF4

        Variable-Level Mapping:
            Maps HRRR variable names to atmospheric levels::

                TMP: 2m_above_ground (air temperature)
                SPFH: 2m_above_ground (specific humidity)
                PRES: surface (surface pressure)
                UGRD: 10m_above_ground (U wind component)
                VGRD: 10m_above_ground (V wind component)
                DSWRF: surface (downward shortwave radiation)
                DLWRF: surface (downward longwave radiation)

        Coordinate Transformation:
            When coordinates are in projection space (projection_x_coordinate,
            projection_y_coordinate), transforms to geographic (latitude, longitude):

            - Uses pyproj Transformer with HRRR Lambert Conformal parameters
            - Creates 2D meshgrid from 1D projection coordinates
            - Transforms entire grid to lat/lon
            - Assigns as auxiliary 2D coordinates

        Time Resampling:
            If HRRR_TIME_STEP_HOURS > 1::

                step = 3  # 3-hourly
                ds_resampled = ds.isel(time=slice(0, None, 3))
                # Keeps hours: 0, 3, 6, 9, 12, 15, 18, 21

        Float16 Handling:
            HRRR Zarr uses Float16 for compression::

                if var.dtype == np.float16:
                    var = var.astype(np.float32)

            Required because NetCDF4 doesn't support Float16.

        Performance:
            - Hourly downloads: ~0.5-2 seconds each
            - Spatial subsetting: ~99% reduction for small basins
            - Memory: One hour at a time (~10-50 MB)
            - Coordinate transform: ~5-15 seconds overhead
            - Total: ~2-10 minutes for 1 week of hourly data

        Example:
            >>> acquirer = HRRRAcquirer(config, logger)
            >>> output = acquirer.download(Path('./forcing/raw'))
            # Downloads: 168 hours (7 days × 24 hours)
            # Skips: 3 missing hours (operational gaps)
            # Final: 165 hourly timesteps
            # Size: ~118 MB
        """
        self.logger.info("Downloading HRRR data from S3")
        fs = s3fs.S3FileSystem(
            anon=True,
            config_kwargs={
                'connect_timeout': 60,
                'read_timeout': 120,
                'retries': {'max_attempts': 10, 'mode': 'adaptive'},
            },
        )
        vars_map = {"TMP": "2m_above_ground", "SPFH": "2m_above_ground", "PRES": "surface", "UGRD": "10m_above_ground", "VGRD": "10m_above_ground", "DSWRF": "surface", "DLWRF": "surface"}
        req_vars = self._get_config_value(lambda: None, default=None, dict_key='HRRR_VARS')
        if req_vars: vars_map = {k: v for k, v in vars_map.items() if k in req_vars}
        hrrr_bbox = self._parse_bbox(self._get_config_value(lambda: None, default=None, dict_key='HRRR_BOUNDING_BOX_COORDS'))
        bbox = hrrr_bbox if hrrr_bbox else self.bbox
        all_datasets, xy_slice = [], None
        curr = self.start_date.date()
        total_days = (self.end_date.date() - self.start_date.date()).days + 1
        while curr <= self.end_date.date():
            dstr = curr.strftime("%Y%m%d")
            day_num = (curr - self.start_date.date()).days + 1
            self.logger.info(f"HRRR download progress: day {day_num}/{total_days} ({dstr})")
            for h in range(24):
                cdt = pd.Timestamp(f"{dstr} {h:02d}:00:00")
                if cdt < self.start_date or cdt > self.end_date: continue
                ds_h = None
                try:
                    v_ds = []
                    for v, level in vars_map.items():
                        try:
                            s1 = s3fs.S3Map(f"hrrrzarr/sfc/{dstr}/{dstr}_{h:02d}z_anl.zarr/{level}/{v}/{level}", s3=fs)
                            s2 = s3fs.S3Map(f"hrrrzarr/sfc/{dstr}/{dstr}_{h:02d}z_anl.zarr/{level}/{v}", s3=fs)
                            ds = self.execute_with_retry(
                                lambda s1=s1, s2=s2: xr.open_mfdataset([s1, s2], engine="zarr", consolidated=False, decode_timedelta=False, data_vars='minimal', coords='minimal', compat='override'),
                                max_retries=3,
                                base_delay=5,
                                backoff_factor=2.0,
                                max_delay=60,
                            )
                            v_ds.append(ds)
                        except Exception as e:  # noqa: BLE001 — preprocessing resilience
                            self.logger.debug(f"Variable {v} not available for {dstr} {h:02d}z: {e}", exc_info=True)
                            continue
                    if v_ds:
                        ds_h = xr.merge(v_ds, compat="override")
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.debug(f"Hour {dstr} {h:02d}z not available: {e}", exc_info=True)
                    continue
                if ds_h is None:
                    continue
                # Window determination is OUTSIDE the per-hour resilience
                # try/except: failing to find a window must abort the
                # acquisition, never silently fetch the full CONUS domain.
                if xy_slice is None:
                    xy_slice = self._determine_xy_window(ds_h, bbox)
                    self.logger.info(
                        f"HRRR spatial window: y={xy_slice[0].start}:{xy_slice[0].stop}, "
                        f"x={xy_slice[1].start}:{xy_slice[1].stop} (windowed before download)"
                    )
                y_dim, x_dim = self._spatial_dims(ds_h)
                all_datasets.append(ds_h.isel({y_dim: xy_slice[0], x_dim: xy_slice[1]}))
            curr += pd.Timedelta(days=1)
        if not all_datasets: raise ValueError("No HRRR data downloaded")
        self.logger.info(f"HRRR download complete: {len(all_datasets)} hours acquired")
        ds_final = xr.concat(all_datasets, dim="time").sortby("time")
        step = int(self._get_config_value(lambda: None, default=1, dict_key='HRRR_TIME_STEP_HOURS'))
        if step > 1: ds_final = ds_final.isel(time=slice(0, None, step))
        if "latitude" not in ds_final.coords and "projection_x_coordinate" in ds_final.coords:
            from pyproj import Transformer
            tr = Transformer.from_crs(self.HRRR_LCC_PROJ, "EPSG:4326", always_xy=True)
            proj_x = ds_final.coords["projection_x_coordinate"].values
            proj_y = ds_final.coords["projection_y_coordinate"].values
            x_mesh, y_mesh = np.meshgrid(proj_x, proj_y)
            lon_flat, lat_flat = tr.transform(x_mesh.ravel(), y_mesh.ravel())
            lon_m = lon_flat.reshape(x_mesh.shape).astype(np.float32)
            lat_m = lat_flat.reshape(y_mesh.shape).astype(np.float32)
            ds_final = ds_final.assign_coords(
                longitude=(["projection_y_coordinate", "projection_x_coordinate"], lon_m),
                latitude=(["projection_y_coordinate", "projection_x_coordinate"], lat_m),
            )

        # Convert float16 to float32 (NetCDF doesn't support float16)
        for var in ds_final.data_vars:
            if ds_final[var].dtype == np.float16:
                ds_final[var] = ds_final[var].astype(np.float32)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_f = output_dir / f"{self.domain_name}_HRRR_hourly_{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}.nc"
        ds_final.to_netcdf(out_f)
        return out_f
