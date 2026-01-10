import xarray as xr
import pandas as pd
import numpy as np
import intake
from pathlib import Path
from typing import Dict, Any
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from symfluence.data.utilities import VariableStandardizer, create_spatial_mask


@AcquisitionRegistry.register('CONUS404')
class CONUS404Acquirer(BaseAcquisitionHandler):
    """Acquirer for CONUS404 reanalysis data from HyTEST catalog."""

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading CONUS404 data")

        # Open catalog and get dataset
        cat_url = self.config.get(
            "CONUS404_CATALOG_URL",
            "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
        )
        cat = intake.open_catalog(cat_url)
        ds_full = cat["conus404-catalog"]["conus404-hourly-osn"].to_dask()

        # Detect coordinate names
        lat_name = next(c for c in ["lat", "latitude"] if c in ds_full)
        lon_name = next(c for c in ["lon", "longitude"] if c in ds_full)

        # Spatial subsetting using centralized utility
        lat_v, lon_v = ds_full[lat_name].values, ds_full[lon_name].values
        mask = create_spatial_mask(lat_v, lon_v, self.bbox)
        iy, ix = np.where(mask)
        if iy.size == 0:
            raise ValueError("No grid points in bbox")

        ds_spatial = ds_full.isel({
            ds_full[lat_name].dims[0]: slice(iy.min(), iy.max() + 1),
            ds_full[lat_name].dims[1]: slice(ix.min(), ix.max() + 1)
        })

        # Temporal subsetting
        ds_subset = ds_spatial.sel(time=slice(self.start_date, self.end_date))

        # Select required variables (prioritize by availability)
        req_vars = ["T2", "Q2", "PSFC", "U10", "V10"]
        # Radiation and precipitation: use first available from each group
        var_groups = [
            ["ACSWDNB", "SWDOWN"],  # Shortwave
            ["ACLWDNB", "LWDOWN"],  # Longwave
            ["PREC_ACC_NC", "RAINRATE", "PRATE", "ACDRIPR"]  # Precipitation
        ]
        for group in var_groups:
            var = next((v for v in group if v in ds_subset.data_vars), None)
            if var:
                req_vars.append(var)

        ds_raw = ds_subset[req_vars].load()

        # Standardize variable names using centralized utility
        standardizer = VariableStandardizer(self.logger)
        ds_final = standardizer.standardize(ds_raw, 'CONUS404')

        # Add metadata and save
        ds_final.attrs.update({"source": "CONUS404", "bbox": str(self.bbox)})
        output_dir.mkdir(parents=True, exist_ok=True)
        out_f = output_dir / f"{self.domain_name}_CONUS404_{self.start_date.year}-{self.end_date.year}.nc"
        ds_final.to_netcdf(out_f)

        return out_f
