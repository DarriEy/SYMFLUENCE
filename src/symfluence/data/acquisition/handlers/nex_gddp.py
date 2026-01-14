"""
NEX-GDDP-CMIP6 climate projection data acquisition via THREDDS.

Provides automated download of NASA NEX-GDDP-CMIP6 downscaled climate model
outputs with support for multiple models, scenarios, and ensemble members.
"""

import datetime as dt
import shutil
from typing import Any
from pathlib import Path
import pandas as pd
import xarray as xr
import requests
import numpy as np
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('NEX-GDDP-CMIP6')
class NEXGDDPCHandler(BaseAcquisitionHandler):
    """
    Acquires NEX-GDDP-CMIP6 downscaled climate projection data via THREDDS.

    NASA NEX-GDDP-CMIP6 provides bias-corrected, downscaled (0.25°) climate
    projections from CMIP6 models. Supports multiple models, scenarios
    (historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5), and ensemble members.
    """

    def download(self, output_dir: Path) -> Path:
        exp_start = self.start_date
        exp_end = self.end_date
        exp_start.strftime("%Y-%m-%d")
        exp_end.strftime("%Y-%m-%d")
        start_dt, end_dt = exp_start.date(), exp_end.date()
        bbox = self.bbox
        lat_min, lat_max = sorted([bbox["lat_min"], bbox["lat_max"]])
        lon_min, lon_max = sorted([bbox["lon_min"], bbox["lon_max"]])
        cfg_models = self.config.get("NEX_MODELS")
        cfg_scenarios = self.config.get("NEX_SCENARIOS", ["historical"])
        variables = self.config.get("NEX_VARIABLES", ["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"])
        cfg_members = self.config.get("NEX_ENSEMBLES", ["r1i1p1f1"])
        if not cfg_models: raise ValueError("NEX_MODELS must be set.")
        ncss_base = "https://ds.nccs.nasa.gov/thredds/ncss/grid"
        cache_root = output_dir / "_nex_ncss_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        ensemble_datasets: list[Any] = []
        for model_name in cfg_models:
            for scenario_name in cfg_scenarios:
                scenario_end_dt = min(end_dt, dt.date(2014, 12, 31)) if scenario_name == "historical" else end_dt
                if start_dt > scenario_end_dt: continue
                for member in cfg_members:
                    all_nc_files_for_ens = []
                    for var in variables:
                        var_cache_dir = cache_root / model_name / scenario_name / member / var
                        var_cache_dir.mkdir(parents=True, exist_ok=True)
                        for year in range(start_dt.year, scenario_end_dt.year + 1):
                            chunk_start = max(start_dt, dt.date(year, 1, 1))
                            chunk_end = min(scenario_end_dt, dt.date(year, 12, 31))
                            if chunk_start > chunk_end: continue
                            fname = f"{var}_day_{model_name}_{scenario_name}_{member}_gn_{year}_v2.0.nc"
                            dataset_path = f"AMES/NEX/GDDP-CMIP6/{model_name}/{scenario_name}/{member}/{var}/{fname}"
                            out_nc = var_cache_dir / f"{fname.replace('.nc', '')}_{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}.nc"
                            if out_nc.exists():
                                all_nc_files_for_ens.append(str(out_nc))
                                continue
                            params = {"var": var, "north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max, "horizStride": 1, "time_start": f"{chunk_start.isoformat()}T12:00:00Z", "time_end": f"{chunk_end.isoformat()}T12:00:00Z", "accept": "netcdf4-classic"}
                            try:
                                resp = requests.get(f"{ncss_base}/{dataset_path}", params=params, stream=True, timeout=600)
                                if resp.status_code == 200:
                                    with open(out_nc, "wb") as f:
                                        for chunk in resp.iter_content(chunk_size=1024*1024): f.write(chunk)
                                    all_nc_files_for_ens.append(str(out_nc))
                            except Exception as e: self.logger.warning(f"NCSS failed: {e}")
                    if all_nc_files_for_ens:
                        ds_ens = xr.open_mfdataset(all_nc_files_for_ens, engine="netcdf4", combine="by_coords", parallel=False).chunk({"time": -1})
                        ds_ens = ds_ens.expand_dims(ensemble=[len(ensemble_datasets)]).assign_coords(model=("ensemble", [model_name]), scenario=("ensemble", [scenario_name]), member=("ensemble", [member]))
                        ensemble_datasets.append(ds_ens)
        if not ensemble_datasets:
            if cache_root.exists(): shutil.rmtree(cache_root)
            raise RuntimeError("NEX-GDDP-CMIP6: no data written.")
        ds_all = xr.concat(ensemble_datasets, dim="ensemble")
        time_vals = pd.to_datetime(ds_all["time"].values)
        month_starts = pd.date_range(time_vals[0].replace(day=1), time_vals[-1], freq="MS")
        for ms in month_starts:
            me = (ms + pd.offsets.MonthEnd(0))
            ds_m = ds_all.sel(time=slice(ms, me))
            if "time" not in ds_m.dims or ds_m.sizes["time"] == 0: continue
            if "ensemble" in ds_m.dims: ds_m = ds_m.isel(ensemble=0, drop=True)
            if "airpres" not in ds_m:
                p0, z_mean, H = 101325.0, float(self.config.get("DOMAIN_MEAN_ELEV_M", 0.0)), 8400.0
                p_surf = p0 * np.exp(-z_mean / H)
                ds_m["airpres"] = xr.full_like(ds_m["tas"], p_surf, dtype="float32").assign_attrs(long_name="synthetic surface air pressure", units="Pa")
            month_path = output_dir / f"NEXGDDP_all_{ms.year:04d}{ms.month:02d}.nc"
            ds_m.to_netcdf(month_path, engine="netcdf4")
        ds_all.close()
        if cache_root.exists(): shutil.rmtree(cache_root)
        return output_dir
