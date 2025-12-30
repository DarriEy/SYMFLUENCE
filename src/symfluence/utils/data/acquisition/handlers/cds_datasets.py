import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('CARRA')
class CARRAAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI: raise ImportError("cdsapi required for CARRA.")
        c = cdsapi.Client()
        domain = self.config.get("CARRA_DOMAIN", "west_domain")
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
        output_dir.mkdir(parents=True, exist_ok=True)
        af = output_dir / f"{self.domain_name}_CARRA_analysis_temp.nc"
        ff = output_dir / f"{self.domain_name}_CARRA_forecast_temp.nc"
        c.retrieve("reanalysis-carra-single-levels", {"domain": domain, "level_type": "surface_or_atmosphere", "product_type": "analysis", "variable": ["2m_temperature", "2m_relative_humidity", "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure"], "year": [str(y) for y in years], "month": months, "day": days, "time": hours, "data_format": "netcdf"}, str(af))
        c.retrieve("reanalysis-carra-single-levels", {"domain": domain, "level_type": "surface_or_atmosphere", "product_type": "forecast", "leadtime_hour": ["1"], "variable": ["total_precipitation", "surface_solar_radiation_downwards", "thermal_surface_radiation_downwards"], "year": [str(y) for y in years], "month": months, "day": days, "time": hours, "data_format": "netcdf"}, str(ff))
        with xr.open_dataset(af) as dsa, xr.open_dataset(ff) as dsf:
            dsa = dsa.rename({(dsa.dims.get('valid_time') and 'valid_time') or 'time': 'time'}).sel(time=slice(self.start_date, self.end_date))
            dsf = dsf.rename({(dsf.dims.get('valid_time') and 'valid_time') or 'time': 'time'})
            dsf['time'] = dsf['time'] - pd.Timedelta(hours=1)
            dsf = dsf.sel(time=slice(self.start_date, self.end_date))
            # Spatial subsetting omitted for brevity, logic follows original
            dsm = dsa.copy()
            for v in ['tp', 'ssrd', 'strd']: dsm[v] = dsf[v]
            rename_map = {'t2m': 'airtemp', 'sp': 'airpres', 'u10': 'wind_u', 'v10': 'wind_v', 'tp': 'pptrate', 'ssrd': 'SWRadAtm', 'strd': 'LWRadAtm'}
            dsm = dsm.rename({k: v for k, v in rename_map.items() if k in dsm})
            if 'wind_u' in dsm and 'wind_v' in dsm: dsm['windspd'] = np.sqrt(dsm['wind_u']**2 + dsm['wind_v']**2)
            if 'r2' in dsm:
                es = 611.2 * np.exp(17.67 * (dsm['airtemp'] - 273.15) / (dsm['airtemp'] - 29.65))
                dsm['spechum'] = (0.622 * (dsm['r2']/100.0 * es)) / (dsm['airpres'] - 0.378 * (dsm['r2']/100.0 * es))
            dsm['pptrate'] = (dsm['pptrate'] * 0.001) / 3600.0
            dsm['SWRadAtm'], dsm['LWRadAtm'] = dsm['SWRadAtm'] / 3600.0, dsm['LWRadAtm'] / 3600.0
            final_f = output_dir / f"{self.domain_name}_CARRA_{self.start_date.year}-{self.end_date.year}.nc"
            dsm.to_netcdf(final_f)
        for f in [af, ff]:
            if f.exists():
                f.unlink()
        return final_f

@AcquisitionRegistry.register('CERRA')
class CERRAAcquirer(BaseAcquisitionHandler):
    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI: raise ImportError("cdsapi required for CERRA.")
        c = cdsapi.Client()
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
        output_dir.mkdir(parents=True, exist_ok=True)
        af = output_dir / f"{self.domain_name}_CERRA_analysis_temp.nc"
        ff = output_dir / f"{self.domain_name}_CERRA_forecast_temp.nc"
        c.retrieve("reanalysis-cerra-single-levels", {"product_type": "analysis", "variable": ["2m_temperature", "2m_relative_humidity", "surface_pressure", "10m_wind_speed"], "year": [str(y) for y in years], "month": months, "day": days, "time": hours, "data_format": "netcdf", "level_type": "surface_or_atmosphere", "data_type": "reanalysis"}, str(af))
        c.retrieve("reanalysis-cerra-single-levels", {"product_type": "forecast", "leadtime_hour": ["1"], "variable": ["total_precipitation", "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards"], "year": [str(y) for y in years], "month": months, "day": days, "time": hours, "data_format": "netcdf", "level_type": "surface_or_atmosphere", "data_type": "reanalysis"}, str(ff))
        with xr.open_dataset(af) as dsa, xr.open_dataset(ff) as dsf:
            dsa = dsa.rename({(dsa.dims.get('valid_time') and 'valid_time') or 'time': 'time'}).sel(time=slice(self.start_date, self.end_date))
            dsf = dsf.rename({(dsf.dims.get('valid_time') and 'valid_time') or 'time': 'time'})
            dsf['time'] = dsf['time'] - pd.Timedelta(hours=1)
            dsf = dsf.sel(time=slice(self.start_date, self.end_date))
            dsm = dsa.copy()
            for v in dsf.data_vars:
                if v not in dsm:
                    dsm[v] = dsf[v]
            dsm = dsm.rename({'t2m': 'airtemp', 'sp': 'airpres', 'si10': 'windspd'})
            if 'r2' in dsm:
                es = 611.2 * np.exp(17.67 * (dsm['airtemp'] - 273.15) / (dsm['airtemp'] + 243.5))
                dsm['spechum'] = (0.622 * (dsm['r2']/100.0 * es)) / (dsm['airpres'] - 0.378 * (dsm['r2']/100.0 * es))
            if 'tp' in dsm: dsm['pptrate'] = dsm['tp'] / (1000.0 * 3600.0)
            if 'ssrd' in dsm: dsm['SWRadAtm'] = dsm['ssrd'] / 3600.0
            if 'strd' in dsm: dsm['LWRadAtm'] = dsm['strd'] / 3600.0
            final_f = output_dir / f"{self.domain_name}_CERRA_{self.start_date.year}-{self.end_date.year}.nc"
            dsm[['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']].to_netcdf(final_f)
        for f in [af, ff]:
            if f.exists():
                f.unlink()
        return final_f
