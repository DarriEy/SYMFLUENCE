import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry

@AcquisitionRegistry.register('CARRA')
class CARRAAcquirer(BaseAcquisitionHandler):
    """
    CARRA (Copernicus Arctic Regional Reanalysis) data acquisition handler.
    Uses dual-product strategy (analysis + forecast).
    """
    
    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi package is required for CARRA downloads.")
            
        c = cdsapi.Client()
        domain = self.config.get("CARRA_DOMAIN", "west_domain")
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(self.start_date.month, self.end_date.month + 1)]
        
        # Build date list
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        days = sorted(list(set([d.strftime('%d') for d in dates])))
        
        # CARRA is hourly
        hours = [f"{h:02d}:00" for h in range(0, 24)]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        af = output_dir / f"{self.domain_name}_CARRA_analysis_temp.nc"
        ff = output_dir / f"{self.domain_name}_CARRA_forecast_temp.nc"
        
        # 1. Download Analysis (Meteorology)
        analysis_req = {
            "domain": domain,
            "level_type": "surface_or_atmosphere",
            "product_type": "analysis",
            "variable": [
                "2m_temperature",
                "2m_relative_humidity",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure"
            ],
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf"
        }
        
        # 2. Download Forecast (Fluxes)
        forecast_req = {
            "domain": domain,
            "level_type": "surface_or_atmosphere",
            "product_type": "forecast",
            "leadtime_hour": ["1"],
            "variable": [
                "total_precipitation",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards"
            ],
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf"
        }
        
        logging.info(f"Downloading CARRA analysis data for {self.domain_name}...")
        c.retrieve("reanalysis-carra-single-levels", analysis_req, str(af))
        
        logging.info(f"Downloading CARRA forecast data for {self.domain_name}...")
        c.retrieve("reanalysis-carra-single-levels", forecast_req, str(ff))
        
        # 3. Process and Merge
        with xr.open_dataset(af) as dsa, xr.open_dataset(ff) as dsf:
            # Handle naming inconsistency in time dimension
            time_name_a = 'valid_time' if 'valid_time' in dsa.dims else 'time'
            time_name_f = 'valid_time' if 'valid_time' in dsf.dims else 'time'
            
            dsa = dsa.rename({time_name_a: 'time'})
            dsf = dsf.rename({time_name_f: 'time'})
            
            # Forecast variables in CARRA are often shifted
            # Leadtime 1h means the valid_time is 1h after the nominal time
            # We want to align them.
            dsf['time'] = dsf['time'] - pd.Timedelta(hours=1)
            
            # Merge
            dsm = xr.merge([dsa, dsf], join='inner')
            
            # Spatial subsetting if bbox is available
            if hasattr(self, 'bbox') and self.bbox:
                lat = dsm.latitude.values
                lon = dsm.longitude.values
                
                # Normalize lon to [0, 360] for CARRA comparison if needed, 
                # or just use what's in the file. CARRA typically uses [0, 360].
                # Our bbox is likely [-180, 180].
                def normalize_lon(l):
                    return l % 360
                
                target_lon_min = normalize_lon(self.bbox['lon_min'])
                target_lon_max = normalize_lon(self.bbox['lon_max'])
                
                # Handle wrapping around prime meridian
                if target_lon_min > target_lon_max:
                    lon_mask = (lon >= target_lon_min) | (lon <= target_lon_max)
                else:
                    lon_mask = (lon >= target_lon_min) & (lon <= target_lon_max)

                mask = (
                    (lat >= self.bbox['lat_min']) & (lat <= self.bbox['lat_max']) &
                    lon_mask
                )
                
                y_idx, x_idx = np.where(mask)
                if len(y_idx) > 0:
                    # Add a small buffer of 2 grid cells
                    y_min = max(0, y_idx.min() - 2)
                    y_max = min(dsm.dims['y'] - 1, y_idx.max() + 2)
                    x_min = max(0, x_idx.min() - 2)
                    x_max = min(dsm.dims['x'] - 1, x_idx.max() + 2)
                    
                    dsm = dsm.isel(y=slice(y_min, y_max + 1), 
                                   x=slice(x_min, x_max + 1))
                    logging.info(f"Spatially subsetted to {dsm.dims['y']}x{dsm.dims['x']} grid")
                else:
                    logging.warning(f"No grid points found in bbox {self.bbox}, keeping full domain")
            
            # Rename to SUMMA standards
            rename_map = {
                't2m': 'airtemp',
                'sp': 'airpres',
                'u10': 'wind_u',
                'v10': 'wind_v',
                'tp': 'pptrate',
                'ssrd': 'SWRadAtm',
                'strd': 'LWRadAtm'
            }
            dsm = dsm.rename({k: v for k, v in rename_map.items() if k in dsm.variables})
            
            # Derived variables
            if 'wind_u' in dsm and 'wind_v' in dsm:
                dsm['windspd'] = np.sqrt(dsm['wind_u']**2 + dsm['wind_v']**2)
                
            if 'r2' in dsm and 'airtemp' in dsm and 'airpres' in dsm:
                # Specific humidity calculation
                T = dsm['airtemp']
                RH = dsm['r2']
                P = dsm['airpres']
                es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
                e = (RH / 100.0) * es
                dsm['spechum'] = (0.622 * e) / (P - 0.378 * e)
            
            # Unit conversions
            if 'pptrate' in dsm:
                # CARRA tp is in kg/m2 (mm) per leadtime (1h) -> m/s
                dsm['pptrate'] = (dsm['pptrate'] * 0.001) / 3600.0
                
            if 'SWRadAtm' in dsm:
                # CARRA ssrd is in J/m2 per leadtime (1h) -> W/m2
                dsm['SWRadAtm'] = dsm['SWRadAtm'] / 3600.0
                
            if 'LWRadAtm' in dsm:
                # CARRA strd is in J/m2 per leadtime (1h) -> W/m2
                dsm['LWRadAtm'] = dsm['LWRadAtm'] / 3600.0
            
            # Subset to final time range
            dsm = dsm.sel(time=slice(self.start_date, self.end_date))
            
            # Final cleanup and save
            final_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
            available_vars = [v for v in final_vars if v in dsm.variables]
            
            final_f = output_dir / f"{self.domain_name}_CARRA_{self.start_date.year}-{self.end_date.year}.nc"
            dsm[available_vars].to_netcdf(final_f)
            
        for f in [af, ff]:
            if f.exists(): f.unlink()
            
        return final_f

@AcquisitionRegistry.register('CERRA')
class CERRAAcquirer(BaseAcquisitionHandler):
    """
    CERRA (Copernicus European Regional Reanalysis) data acquisition handler.
    CERRA is 3-hourly.
    """
    
    def download(self, output_dir: Path) -> Path:
        if not HAS_CDSAPI:
            raise ImportError("cdsapi package is required for CERRA downloads.")
            
        c = cdsapi.Client()
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(self.start_date.month, self.end_date.month + 1)]
        
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        days = sorted(list(set([d.strftime('%d') for d in dates])))
        
        # CERRA is 3-hourly
        hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        af = output_dir / f"{self.domain_name}_CERRA_analysis_temp.nc"
        ff = output_dir / f"{self.domain_name}_CERRA_forecast_temp.nc"
        
        # 1. Download Analysis
        analysis_req = {
            "product_type": "analysis",
            "level_type": "surface_or_atmosphere",
            "variable": [
                "2m_temperature",
                "2m_relative_humidity",
                "surface_pressure",
                "10m_wind_speed"
            ],
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
            "data_type": "reanalysis"
        }
        
        # 2. Download Forecast
        forecast_req = {
            "product_type": "forecast",
            "level_type": "surface_or_atmosphere",
            "leadtime_hour": ["1"],
            "variable": [
                "total_precipitation",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards"
            ],
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
            "data_type": "reanalysis"
        }
        
        logging.info(f"Downloading CERRA analysis data for {self.domain_name}...")
        c.retrieve("reanalysis-cerra-single-levels", analysis_req, str(af))
        
        logging.info(f"Downloading CERRA forecast data for {self.domain_name}...")
        c.retrieve("reanalysis-cerra-single-levels", forecast_req, str(ff))
        
        # 3. Process and Merge
        with xr.open_dataset(af) as dsa, xr.open_dataset(ff) as dsf:
            time_name_a = 'valid_time' if 'valid_time' in dsa.dims else 'time'
            time_name_f = 'valid_time' if 'valid_time' in dsf.dims else 'time'
            
            dsa = dsa.rename({time_name_a: 'time'})
            dsf = dsf.rename({time_name_f: 'time'})
            
            # Align forecast
            dsf['time'] = dsf['time'] - pd.Timedelta(hours=1)
            
            dsm = xr.merge([dsa, dsf], join='inner')
            
            # Spatial subsetting
            if hasattr(self, 'bbox') and self.bbox:
                lat = dsm.latitude.values
                lon = dsm.longitude.values
                mask = (
                    (lat >= self.bbox['lat_min']) & (lat <= self.bbox['lat_max']) &
                    (lon >= self.bbox['lon_min']) & (lon <= self.bbox['lon_max'])
                )
                y_idx, x_idx = np.where(mask)
                if len(y_idx) > 0:
                    dsm = dsm.isel(y=slice(y_idx.min(), y_idx.max() + 1), 
                                   x=slice(x_idx.min(), x_idx.max() + 1))
            
            # Rename and derived
            rename_map = {'t2m': 'airtemp', 'sp': 'airpres', 'si10': 'windspd', 'tp': 'pptrate', 'ssrd': 'SWRadAtm', 'strd': 'LWRadAtm'}
            dsm = dsm.rename({k: v for k, v in rename_map.items() if k in dsm.variables})
            
            if 'r2' in dsm:
                T = dsm['airtemp']; RH = dsm['r2']; P = dsm['airpres']
                es = 611.2 * np.exp(17.67 * (T - 273.15) / (T + 243.5))
                e = (RH / 100.0) * es
                dsm['spechum'] = (0.622 * e) / (P - 0.378 * e)
            
            # Unit conversions (CERRA tp/ssrd/strd also in J/m2 or kg/m2 per leadtime)
            if 'pptrate' in dsm: dsm['pptrate'] = (dsm['pptrate'] * 0.001) / 3600.0
            if 'SWRadAtm' in dsm: dsm['SWRadAtm'] = dsm['SWRadAtm'] / 3600.0
            if 'LWRadAtm' in dsm: dsm['LWRadAtm'] = dsm['LWRadAtm'] / 3600.0
            
            dsm = dsm.sel(time=slice(self.start_date, self.end_date))
            
            final_vars = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'windspd', 'spechum', 'LWRadAtm']
            available_vars = [v for v in final_vars if v in dsm.variables]
            
            final_f = output_dir / f"{self.domain_name}_CERRA_{self.start_date.year}-{self.end_date.year}.nc"
            dsm[available_vars].to_netcdf(final_f)
            
        for f in [af, ff]:
            if f.exists(): f.unlink()
            
        return final_f
