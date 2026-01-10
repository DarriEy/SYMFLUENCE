import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from .era5_cds import ERA5CDSAcquirer
from .era5_processing import era5_to_summa_schema

def has_cds_credentials():
    """Check if CDS API credentials are available."""
    return os.path.exists(os.path.expanduser('~/.cdsapirc')) or 'CDSAPI_KEY' in os.environ

@AcquisitionRegistry.register('ERA5')
class ERA5Acquirer(BaseAcquisitionHandler):
    """
    Dispatcher for ERA5 data acquisition, choosing between ARCO (Zarr) and CDS (NetCDF) pathways.
    """
    def download(self, output_dir: Path) -> Path:
        # Default to ARCO if libraries available, falling back to CDS

        # Get ERA5_USE_CDS from typed config (supports both typed and dict config)
        use_cds = self._get_config_value(lambda: self.config.forcing.era5_use_cds)

        # Also check era5 subsection
        if use_cds is None:
            use_cds = self._get_config_value(lambda: self.config.forcing.era5.use_cds)

        # Check environment variable as fallback
        if use_cds is None:
            env_use_cds = os.environ.get('ERA5_USE_CDS')
            if env_use_cds:
                use_cds = env_use_cds.lower() in ('true', 'yes', '1', 'on')
                self.logger.info(f"Using ERA5_USE_CDS from environment: {use_cds}")

        self.logger.info(f"ERA5_USE_CDS config value: {use_cds} (type: {type(use_cds)})")

        if use_cds is None:
            # Auto-detect preference: ARCO (faster) > CDS
            # Both pathways now have the longwave radiation fix, so prefer ARCO for speed
            try:
                import gcsfs
                import xarray
                self.logger.info("Auto-detecting ERA5 pathway: ARCO (Google Cloud) - faster, no queue")
                self.logger.info("  To use CDS instead, set ERA5_USE_CDS=true in config or environment")
                use_cds = False
            except ImportError:
                if has_cds_credentials():
                    self.logger.info("gcsfs not available, falling back to CDS pathway")
                    use_cds = True
                else:
                    self.logger.error("Neither gcsfs nor CDS credentials available for ERA5 download")
                    raise ImportError("Install gcsfs (pip install gcsfs) or configure CDS credentials (~/.cdsapirc)")
        else:
            # Handle string values like "true", "True", "yes", etc.
            if isinstance(use_cds, str):
                use_cds = use_cds.lower() in ('true', 'yes', '1', 'on')

        self.logger.info(f"Using CDS pathway: {use_cds}")

        if use_cds:
            self.logger.info("Using CDS pathway for ERA5")
            try:
                return ERA5CDSAcquirer(self.config, self.logger).download(output_dir)
            except Exception as e:
                self.logger.warning(f"CDS pathway failed: {e}. Falling back to ARCO if possible.")
        
        self.logger.info("Using ARCO (Google Cloud) pathway for ERA5")
        return ERA5ARCOAcquirer(self.config, self.logger).download(output_dir)

class ERA5ARCOAcquirer(BaseAcquisitionHandler):
    """
    ERA5 data acquisition handler using the Google Cloud ARCO-ERA5 (Zarr) pathway.
    """
    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading ERA5 data from Google Cloud ARCO-ERA5")
        domain_name = self.domain_name

        try:
            import gcsfs
            from pandas.tseries.offsets import MonthEnd
        except ImportError as e:
            raise ImportError("gcsfs and xarray are required for ERA5 cloud access.") from e

        gcs = gcsfs.GCSFileSystem(token="anon")
        default_store = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        zarr_store = self._get_config_value(
            lambda: self.config.forcing.era5.zarr_path, default=default_store
        )

        mapper = gcs.get_mapper(zarr_store)
        ds = xr.open_zarr(mapper, consolidated=True, chunks={})
        ds = ds.assign_coords(longitude=ds.longitude.load(), latitude=ds.latitude.load(), time=ds.time.load())

        raw_lon1, raw_lon2 = float(self.bbox["lon_min"]), float(self.bbox["lon_max"])
        raw_lat1, raw_lat2 = float(self.bbox["lat_min"]), float(self.bbox["lat_max"])
        lon_min_raw, lon_max_raw = sorted([raw_lon1, raw_lon2])
        lon_min, lon_max = (lon_min_raw + 360) % 360 if lon_min_raw < 0 else lon_min_raw, (lon_max_raw + 360) % 360 if lon_max_raw < 0 else lon_max_raw
        lat_min_raw, lat_max_raw = sorted([raw_lat1, raw_lat2])

        step = self._get_config_value(
            lambda: self.config.forcing.era5.time_step_hours, default=1
        )
        step = int(step)
        era5_start = self.start_date - pd.Timedelta(hours=step)
        era5_end = self.end_date

        current_month_start = era5_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        chunks = []
        while current_month_start <= era5_end:
            month_end = (current_month_start + MonthEnd(1)).replace(hour=23, minute=0, second=0, microsecond=0)
            chunk_start, chunk_end = max(era5_start, current_month_start), min(era5_end, month_end)
            if chunk_start <= chunk_end: chunks.append((chunk_start, chunk_end))
            current_month_start = (current_month_start.replace(day=28) + pd.Timedelta(days=4)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        default_vars = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure", "total_precipitation", "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards"]
        requested_vars = self._get_config_value(
            lambda: self.config.forcing.era5.variables, default=default_vars
        ) or default_vars
        available_vars = [v for v in requested_vars if v in ds.data_vars]

        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []

        # Default to parallel processing if not specified
        n_workers_cfg = self._get_config_value(lambda: self.config.system.mpi_processes)
        if n_workers_cfg is not None:
            n_workers = int(n_workers_cfg)
        else:
            import os
            # Use available CPUs but cap at 8 to avoid overwhelming I/O
            n_workers = min(8, os.cpu_count() or 1)
            
        self.logger.info(f"Processing ERA5 with {n_workers} workers")

        if n_workers <= 1:
            for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                self.logger.info(f"Processing ERA5 chunk {i}/{len(chunks)}: {chunk_start.strftime('%Y-%m')} to {chunk_end.strftime('%Y-%m')}")
                time_start = chunk_start if i == 1 else chunk_start - pd.Timedelta(hours=step)
                ds_t = ds.sel(time=slice(time_start, chunk_end))
                if "time" not in ds_t.dims or ds_t.sizes["time"] < 2: continue
                ds_ts = ds_t.sel(latitude=slice(lat_max_raw, lat_min_raw), longitude=slice(lon_min, lon_max))

                # Check for empty spatial dimensions (bounding box too small for grid resolution)
                if "latitude" not in ds_ts.dims or "longitude" not in ds_ts.dims:
                    self.logger.warning(f"Chunk {i}: Missing spatial dimensions after bounding box selection")
                    continue
                if ds_ts.sizes.get("latitude", 0) == 0 or ds_ts.sizes.get("longitude", 0) == 0:
                    self.logger.warning(f"Chunk {i}: Empty spatial dimensions after bounding box selection. "
                                       f"Bounding box may be too small for ERA5 resolution (0.25°)")
                    continue

                if step > 1 and "time" in ds_ts.dims: ds_ts = ds_ts.isel(time=slice(0, None, step))
                if "time" not in ds_ts.dims or ds_ts.sizes["time"] < 2: continue
                ds_chunk = era5_to_summa_schema(ds_ts[[v for v in available_vars if v in ds_ts.data_vars]], source='arco', logger=self.logger)
                if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 1: continue
                file_year, file_month = chunk_start.year, chunk_start.month
                chunk_file = output_dir / f"domain_{domain_name}_ERA5_merged_{file_year}{file_month:02d}.nc"
                encoding = {var: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for var in ds_chunk.data_vars}
                ds_chunk.to_netcdf(chunk_file, encoding=encoding, compute=True)
                self.logger.info(f"✓ Successfully saved ERA5 chunk {i}/{len(chunks)} to {chunk_file.name}")
                chunk_files.append(chunk_file)
        else:
            from concurrent.futures import ThreadPoolExecutor
            def process_chunk(i, chunk_start, chunk_end):
                return _process_era5_chunk_threadsafe(i, (chunk_start, chunk_end), ds, available_vars, step, lat_min_raw, lat_max_raw, lon_min, lon_max, output_dir, domain_name, len(chunks), self.logger)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(process_chunk, i, *chunks[i-1]) for i in range(1, len(chunks)+1)]
                for future in futures:
                    _, cf, _ = future.result()
                    if cf: chunk_files.append(cf)

        return output_dir if len(chunk_files) > 1 else (chunk_files[0] if chunk_files else output_dir)


def _process_era5_chunk_threadsafe(idx, times, ds, vars, step, lat_min, lat_max, lon_min, lon_max, out_dir, dom, total, logger=None):
    start, end = times
    try:
        ts = start if idx == 1 else start - pd.Timedelta(hours=step)
        ds_t = ds.sel(time=slice(ts, end))
        if "time" not in ds_t.dims or ds_t.sizes["time"] < 2: return idx, None, "skipped"
        ds_ts = ds_t.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

        # Check for empty spatial dimensions
        if "latitude" not in ds_ts.dims or "longitude" not in ds_ts.dims:
            return idx, None, "skipped: missing spatial dimensions"
        if ds_ts.sizes.get("latitude", 0) == 0 or ds_ts.sizes.get("longitude", 0) == 0:
            return idx, None, "skipped: empty spatial dimensions (bbox too small for ERA5 resolution)"

        if step > 1 and "time" in ds_ts.dims: ds_ts = ds_ts.isel(time=slice(0, None, step))
        if "time" not in ds_ts.dims or ds_ts.sizes["time"] < 2: return idx, None, "skipped"
        ds_chunk = era5_to_summa_schema(ds_ts[[v for v in vars if v in ds_ts.data_vars]], source='arco', logger=logger)
        cf = out_dir / f"domain_{dom}_ERA5_merged_{start.year}{start.month:02d}.nc"
        ds_chunk.to_netcdf(cf, encoding={v: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for v in ds_chunk.data_vars})
        return idx, cf, "success"
    except Exception as e: return idx, None, str(e)
