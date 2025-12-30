import xarray as xr
import pandas as pd
import numpy as np
import intake
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('CONUS404')
class CONUS404Acquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading CONUS404 data")
        cat_url = self.config.get("CONUS404_CATALOG_URL", "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml")
        cat = intake.open_catalog(cat_url)
        ds_full = cat["conus404-catalog"]["conus404-hourly-osn"].to_dask()
        lat_name = next(c for c in ["lat", "latitude"] if c in ds_full)
        lon_name = next(c for c in ["lon", "longitude"] if c in ds_full)
        lat_min, lat_max = sorted([self.bbox["lat_min"], self.bbox["lat_max"]])
        lon_min, lon_max = sorted([self.bbox["lon_min"], self.bbox["lon_max"]])
        lat_v, lon_v = ds_full[lat_name].values, ds_full[lon_name].values
        mask = (lat_v >= lat_min) & (lat_v <= lat_max) & (lon_v >= lon_min) & (lon_v <= lon_max)
        iy, ix = np.where(mask)
        if iy.size == 0: raise ValueError("No grid points in bbox")
        ds_spatial = ds_full.isel({ds_full[lat_name].dims[0]: slice(iy.min(), iy.max()+1), ds_full[lat_name].dims[1]: slice(ix.min(), ix.max()+1)})
        ds_subset = ds_spatial.sel(time=slice(self.start_date, self.end_date))
        req_vars = ["T2", "Q2", "PSFC", "U10", "V10"]
        for c in [["ACSWDNB", "SWDOWN"], ["ACLWDNB", "LWDOWN"], ["RAINRATE", "PRATE"]]:
            v = next((v for v in c if v in ds_subset.data_vars), None)
            if v: req_vars.append(v)
        ds_final = ds_subset[req_vars].load()
        ds_final.attrs.update({"source": "CONUS404", "bbox": str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        out_f = output_dir / f"{self.domain_name}_CONUS404_{self.start_date.year}-{self.end_date.year}.nc"
        ds_final.to_netcdf(out_f)
        return out_f
