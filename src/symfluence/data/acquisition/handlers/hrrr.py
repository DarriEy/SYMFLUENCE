import xarray as xr
import pandas as pd
import numpy as np
import s3fs
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('HRRR')
class HRRRAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading HRRR data from S3")
        fs = s3fs.S3FileSystem(anon=True)
        vars_map = {"TMP": "2m_above_ground", "SPFH": "2m_above_ground", "PRES": "surface", "UGRD": "10m_above_ground", "VGRD": "10m_above_ground", "DSWRF": "surface", "DLWRF": "surface"}
        req_vars = self.config.get("HRRR_VARS")
        if req_vars: vars_map = {k: v for k, v in vars_map.items() if k in req_vars}
        hrrr_bbox = self._parse_bbox(self.config.get("HRRR_BOUNDING_BOX_COORDS"))
        bbox = hrrr_bbox if hrrr_bbox else self.bbox
        all_datasets, xy_slice = [], None
        curr = self.start_date.date()
        while curr <= self.end_date.date():
            dstr = curr.strftime("%Y%m%d")
            for h in range(24):
                cdt = pd.Timestamp(f"{dstr} {h:02d}:00:00")
                if cdt < self.start_date or cdt > self.end_date: continue
                try:
                    v_ds = []
                    for v, l in vars_map.items():
                        try:
                            s1 = s3fs.S3Map(f"hrrrzarr/sfc/{dstr}/{dstr}_{h:02d}z_anl.zarr/{l}/{v}/{l}", s3=fs)
                            s2 = s3fs.S3Map(f"hrrrzarr/sfc/{dstr}/{dstr}_{h:02d}z_anl.zarr/{l}/{v}", s3=fs)
                            v_ds.append(xr.open_mfdataset([s1, s2], engine="zarr", consolidated=False))
                        except Exception: continue
                    if v_ds:
                        ds_h = xr.merge(v_ds)
                        if xy_slice is None and "latitude" in ds_h.coords:
                            mask = (
                                (ds_h.latitude >= bbox["lat_min"])
                                & (ds_h.latitude <= bbox["lat_max"])
                                & (ds_h.longitude >= bbox["lon_min"])
                                & (ds_h.longitude <= bbox["lon_max"])
                            )
                            iy, ix = np.where(mask)
                            if len(iy) > 0: xy_slice = (slice(iy.min(), iy.max()+1), slice(ix.min(), ix.max()+1))
                        all_datasets.append(ds_h.isel(y=xy_slice[0], x=xy_slice[1]) if xy_slice else ds_h)
                except Exception: continue
            curr += pd.Timedelta(days=1)
        if not all_datasets: raise ValueError("No HRRR data downloaded")
        ds_final = xr.concat(all_datasets, dim="time").sortby("time")
        step = int(self.config.get("HRRR_TIME_STEP_HOURS", 1))
        if step > 1: ds_final = ds_final.isel(time=slice(0, None, step))
        if "latitude" not in ds_final.coords and "projection_x_coordinate" in ds_final.coords:
            from pyproj import Transformer
            tr = Transformer.from_crs(
                "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs",
                "EPSG:4326",
                always_xy=True,
            )
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
