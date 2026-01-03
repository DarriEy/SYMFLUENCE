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

def has_cds_credentials():
    """Check if CDS API credentials are available."""
    return os.path.exists(os.path.expanduser('~/.cdsapirc')) or 'CDSAPI_KEY' in os.environ

@AcquisitionRegistry.register('ERA5')
class ERA5Acquirer(BaseAcquisitionHandler):
    """
    Dispatcher for ERA5 data acquisition, choosing between ARCO (Zarr) and CDS (NetCDF) pathways.
    """
    def download(self, output_dir: Path) -> Path:
        # Default to CDS if credentials exist, unless explicitly disabled
        use_cds = self.config.get('ERA5_USE_CDS', has_cds_credentials())
        
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
        zarr_store = self.config.get("ERA5_ZARR_PATH", default_store)

        mapper = gcs.get_mapper(zarr_store)
        ds = xr.open_zarr(mapper, consolidated=True, chunks={})
        ds = ds.assign_coords(longitude=ds.longitude.load(), latitude=ds.latitude.load(), time=ds.time.load())

        raw_lon1, raw_lon2 = float(self.bbox["lon_min"]), float(self.bbox["lon_max"])
        raw_lat1, raw_lat2 = float(self.bbox["lat_min"]), float(self.bbox["lat_max"])
        lon_min_raw, lon_max_raw = sorted([raw_lon1, raw_lon2])
        lon_min, lon_max = (lon_min_raw + 360) % 360 if lon_min_raw < 0 else lon_min_raw, (lon_max_raw + 360) % 360 if lon_max_raw < 0 else lon_max_raw
        lat_min_raw, lat_max_raw = sorted([raw_lat1, raw_lat2])

        step = int(self.config.get("ERA5_TIME_STEP_HOURS", 1))
        era5_start = self.start_date - pd.Timedelta(hours=step)
        era5_end = self.end_date

        current_month_start = era5_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        chunks = []
        while current_month_start <= era5_end:
            month_end = (current_month_start + MonthEnd(1)).replace(hour=23, minute=0, second=0, microsecond=0)
            chunk_start, chunk_end = max(era5_start, current_month_start), min(era5_end, month_end)
            if chunk_start <= chunk_end: chunks.append((chunk_start, chunk_end))
            current_month_start = (current_month_start.replace(day=28) + pd.Timedelta(days=4)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        requested_vars = self.config.get("ERA5_VARS", ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure", "total_precipitation", "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards"])
        available_vars = [v for v in requested_vars if v in ds.data_vars]

        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []
        n_workers = int(self.config.get('MPI_PROCESSES', 1))

        if n_workers <= 1:
            for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
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
                ds_chunk = _era5_to_summa_schema_standalone(ds_ts[[v for v in available_vars if v in ds_ts.data_vars]])
                if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 1: continue
                file_year, file_month = chunk_start.year, chunk_start.month
                chunk_file = output_dir / f"domain_{domain_name}_ERA5_merged_{file_year}{file_month:02d}.nc"
                encoding = {var: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for var in ds_chunk.data_vars}
                ds_chunk.to_netcdf(chunk_file, encoding=encoding, compute=True)
                chunk_files.append(chunk_file)
        else:
            from concurrent.futures import ThreadPoolExecutor
            def process_chunk(i, chunk_start, chunk_end):
                return _process_era5_chunk_threadsafe(i, chunk_start, chunk_end, ds, available_vars, step, lat_min_raw, lat_max_raw, lon_min, lon_max, output_dir, domain_name, len(chunks), self.logger)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(process_chunk, i, *chunks[i-1]) for i in range(1, len(chunks)+1)]
                for future in futures:
                    _, cf, _ = future.result()
                    if cf: chunk_files.append(cf)

        return output_dir if len(chunk_files) > 1 else (chunk_files[0] if chunk_files else output_dir)

def _era5_to_summa_schema_standalone(ds_chunk):
    import xarray as xr
    import numpy as np
    if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 2: return ds_chunk
    ds_chunk = ds_chunk.sortby("time")
    ds_base = ds_chunk.isel(time=slice(1, None))
    out = xr.Dataset(coords={c: ds_base.coords[c] for c in ds_base.coords})
    if "surface_pressure" in ds_base: out["airpres"] = ds_base["surface_pressure"].astype("float32").assign_attrs(units="Pa", long_name="air pressure", standard_name="air_pressure")
    if "2m_temperature" in ds_base: out["airtemp"] = ds_base["2m_temperature"].astype("float32").assign_attrs(units="K", long_name="air temperature", standard_name="air_temperature")
    if "10m_u_component_of_wind" in ds_base and "10m_v_component_of_wind" in ds_base:
        out["windspd"] = np.sqrt(ds_base["10m_u_component_of_wind"]**2 + ds_base["10m_v_component_of_wind"]**2).astype("float32").assign_attrs(units="m s-1", long_name="wind speed", standard_name="wind_speed")
    if "2m_dewpoint_temperature" in ds_base and "surface_pressure" in ds_base:
        Td_C, p = ds_base["2m_dewpoint_temperature"] - 273.15, ds_base["surface_pressure"]
        es = 611.2 * np.exp((17.67 * Td_C) / (Td_C + 243.5)) * 100.0
        r = 0.622 * es / xr.where((p - es) <= 1.0, 1.0, p - es)
        out["spechum"] = (r / (1.0 + r)).astype("float32").assign_attrs(units="kg kg-1", long_name="specific humidity", standard_name="specific_humidity")
    dt = (ds_chunk["time"].diff("time") / np.timedelta64(1, "s")).astype("float32")
    def _accum_to_rate(vn, on, u, ln, sn, sf=1.0):
        if vn in ds_chunk: out[on] = ((ds_chunk[vn].diff("time").where(ds_chunk[vn].diff("time") >= 0, 0) / dt) * sf).clip(min=0).astype("float32").assign_attrs(units=u, long_name=ln, standard_name=sn)
    _accum_to_rate("total_precipitation", "pptrate", "mm/s", "precipitation rate", "precipitation_rate", 1000.0)
    _accum_to_rate("surface_solar_radiation_downwards", "SWRadAtm", "W m-2", "shortwave radiation", "surface_downwelling_shortwave_flux_in_air")
    _accum_to_rate("surface_thermal_radiation_downwards", "LWRadAtm", "W m-2", "longwave radiation", "surface_downwelling_longwave_flux_in_air")
    return out

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
        ds_chunk = _era5_to_summa_schema_standalone(ds_ts[[v for v in vars if v in ds_ts.data_vars]])
        cf = out_dir / f"domain_{dom}_ERA5_merged_{start.year}{start.month:02d}.nc"
        ds_chunk.to_netcdf(cf, encoding={v: {"zlib": True, "complevel": 1, "chunksizes": (min(168, ds_chunk.sizes["time"]), ds_chunk.sizes["latitude"], ds_chunk.sizes["longitude"])} for v in ds_chunk.data_vars})
        return idx, cf, "success"
    except Exception as e: return idx, None, str(e)
