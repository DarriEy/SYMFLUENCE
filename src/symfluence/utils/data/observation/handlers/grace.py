import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('GRACE')
class GRACEHandler(BaseObservationHandler):
    """
    Handles GRACE Total Water Storage anomaly data.
    Implements adaptive extraction based on basin size.
    """
    
    # Basin size thresholds for extraction strategy
    STRATEGY_CONFIG = {
        'large_basin_threshold': 5000,      # > 5000 km²: bounding box
        'medium_basin_threshold': 1000,     # 1000-5000 km²: buffered bounding box
        'buffer_medium': 0.5,               # Buffer for medium basins (degrees)
    }

    def acquire(self) -> Path:
        """Locate GRACE data or download if possible (cloud path placeholder)."""
        grace_dir = Path(self.config.get('GRACE_DATA_DIR', self.project_dir / "observations" / "grace"))
        if not grace_dir.exists():
            grace_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"GRACE data directory not found: {grace_dir}")
        
        # In a future update, this would trigger a cloud download
        return grace_dir

    def process(self, input_path: Path) -> Path:
        """Process GRACE data for the current domain."""
        self.logger.info(f"Processing GRACE TWS for domain: {self.domain_name}")
        
        # Load basin shapefile
        catchment_path = Path(self.config.get('CATCHMENT_PATH', self.project_dir / "shapefiles" / "catchment"))
        catchment_name = self.config.get('CATCHMENT_SHP_NAME', f"{self.domain_name}_catchment.shp")
        if catchment_name == 'default':
             catchment_name = f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION', 'GRUs')}.shp"
        
        basin_shp = catchment_path / catchment_name
        if not basin_shp.exists():
            raise FileNotFoundError(f"Basin shapefile not found: {basin_shp}")
            
        basin_gdf = gpd.read_file(basin_shp)
        basin_area_km2 = self._calculate_area(basin_gdf)
        self.logger.info(f"Basin area: {basin_area_km2:.1f} km²")
        
        # Find GRACE files
        grace_files = self._find_grace_files(input_path)
        if not grace_files:
            self.logger.error("No GRACE NetCDF files found")
            return input_path
            
        results = {}
        for name, file_path in grace_files.items():
            with xr.open_dataset(file_path) as ds:
                ts = self._extract_for_basin(ds, basin_gdf, name, basin_area_km2)
                if ts is not None:
                    # Calculate anomalies (2003-2008 baseline as default)
                    ts_anomaly = self._calculate_anomalies(ts)
                    results[f'grace_{name}'] = ts
                    results[f'grace_{name}_anomaly'] = ts_anomaly
        
        if not results:
            self.logger.warning("No GRACE data could be extracted")
            return input_path
            
        # Save to CSV
        df = pd.DataFrame(results)
        output_dir = self.project_dir / "observations" / "grace" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_grace_tws_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"GRACE processing complete: {output_file}")
        return output_file

    def _calculate_area(self, gdf: gpd.GeoDataFrame) -> float:
        # Use equal area projection for calculation
        return gdf.to_crs('EPSG:6933').geometry.area.sum() / 1e6

    def _find_grace_files(self, grace_dir: Path) -> Dict[str, Path]:
        files = {}
        # Simple patterns based on common naming
        patterns = {'jpl': '*JPL*.nc', 'csr': '*CSR*.nc', 'gsfc': '*gsfc*.nc'}
        for name, pattern in patterns.items():
            found = list(grace_dir.rglob(pattern))
            if found: files[name] = found[0]
        return files

    def _extract_for_basin(self, ds: xr.Dataset, gdf: gpd.GeoDataFrame, name: str, area: float) -> Optional[pd.Series]:
        centroid = gdf.dissolve().centroid.iloc[0]
        
        # Adaptive strategy
        if area <= self.STRATEGY_CONFIG['medium_basin_threshold']:
            # Point sampling
            lons, lats = ds.lon.values, ds.lat.values
            c_lon = centroid.x + 360 if centroid.x < 0 and lons.max() > 180 else centroid.x
            idx_lon = np.argmin(np.abs(lons - c_lon))
            idx_lat = np.argmin(np.abs(lats - centroid.y))
            data = ds.lwe_thickness.isel(lon=idx_lon, lat=idx_lat)
        else:
            # Spatial averaging
            bounds = gdf.total_bounds
            if bounds[0] < 0 and ds.lon.values.max() > 180:
                bounds[0] += 360; bounds[2] += 360
            
            lon_mask = (ds.lon >= bounds[0]) & (ds.lon <= bounds[2])
            lat_mask = (ds.lat >= bounds[1]) & (ds.lat <= bounds[3])
            subset = ds.lwe_thickness.where(lon_mask & lat_mask, drop=True)
            data = subset.mean(dim=[d for d in subset.dims if d != 'time'])
            
        time_idx = self._get_time_index(ds, name)
        return pd.Series(data.values, index=time_idx).resample('MS').mean()

    def _get_time_index(self, ds: xr.Dataset, name: str) -> pd.DatetimeIndex:
        # Handle CSR/GSFC relative time if needed
        if 'units' in ds.time.attrs and 'days since' in ds.time.attrs['units']:
             return pd.to_datetime(ds.time.values, unit='D', origin=ds.time.attrs['units'].split('since ')[1])
        return pd.to_datetime(ds.time.values)

    def _calculate_anomalies(self, ts: pd.Series) -> pd.Series:
        baseline = ts.loc['2003-01-01':'2008-12-31']
        mean = baseline.mean() if not baseline.empty else ts.mean()
        return ts - mean
