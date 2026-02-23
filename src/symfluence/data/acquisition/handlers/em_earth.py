"""
EM-Earth climate forcing data acquisition from AWS S3.

Provides automated download of EM-Earth reanalysis data with support for
deterministic and probabilistic variants, multi-year coverage, and spatial subsetting.
"""

from pathlib import Path

import pandas as pd
import s3fs
import xarray as xr

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry


@AcquisitionRegistry.register('EM-EARTH')
@AcquisitionRegistry.register('EM_EARTH')
class EMEarthAcquirer(BaseAcquisitionHandler):
    """
    Acquires EM-Earth global climate reanalysis data from AWS S3.

    EM-Earth provides daily meteorological variables at 0.1° resolution
    globally from 1950-2019, available in deterministic and probabilistic
    variants. Data includes precipitation, temperature (mean, range), and
    dewpoint temperature.
    """

    def download(self, output_dir: Path) -> Path:
        self.logger.info("Downloading EM-Earth data from AWS S3")
        fs = s3fs.S3FileSystem(anon=True)
        emearth_type = str(self._get_config_value(lambda: self.config.forcing.em_earth.data_type, default="deterministic", dict_key='EM_EARTH_DATA_TYPE')).lower()
        base_folder = "nc/deterministic_raw_daily" if emearth_type == "deterministic" else "nc/probabilistic_daily"
        precip_var = self._get_config_value(lambda: self.config.forcing.em_earth.prcp_var, default="prcp", dict_key='EM_PRCP')
        vars = [precip_var, "tmean", "trange", "tdew"]
        region = str(self._get_config_value(lambda: None, default="global", dict_key='EM_EARTH_REGION_FOLDER'))
        use_region = region.lower() not in ("global", "")
        start_year, end_year = max(self.start_date.year, 1950), min(self.end_date.year, 2019)
        all_datasets = {}
        for var in vars:
            var_datasets = []
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    ym = f"{year}{month:02d}"
                    fname = f"EM_Earth_{emearth_type}_daily_{var}_{ym}.nc"
                    key = f"emearth/{base_folder}/{var}/{region}/{fname}" if use_region else f"emearth/{base_folder}/{var}/{fname}"
                    try:
                        if fs.exists(key):
                            with fs.open(key, "rb") as f:
                                ds = xr.open_dataset(f, engine="h5netcdf")
                                ds_subset = ds.sel(lat=slice(self.bbox["lat_min"], self.bbox["lat_max"]), lon=slice(self.bbox["lon_min"], self.bbox["lon_max"]))
                                real_start, real_end = max(self.start_date, pd.Timestamp(f"{year}-{month:02d}-01")), min(self.end_date, pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0))
                                ds_subset = ds_subset.sel(time=slice(real_start, real_end))
                                if len(ds_subset.time) > 0: var_datasets.append(ds_subset.load())
                    except (OSError, KeyError, ValueError) as e:
                        self.logger.debug(f"EM-Earth file not available for {var} {ym}: {e}")
                        continue
            if var_datasets: all_datasets[var] = xr.concat(var_datasets, dim="time")
        if not all_datasets: raise ValueError("No EM-Earth data downloaded")
        ds_final = xr.merge(list(all_datasets.values()))
        ds_final.attrs.update({"source": "EM-Earth", "bbox": str(self.bbox)})
        save_dir = output_dir / "raw_data_em_earth" if self._get_config_value(lambda: self.config.forcing.supplement, default=False, dict_key='SUPPLEMENT_FORCING') else output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f"{self.domain_name}_EM-Earth_{emearth_type}_{start_year}-{end_year}.nc"
        ds_final.to_netcdf(output_file)
        return output_file
