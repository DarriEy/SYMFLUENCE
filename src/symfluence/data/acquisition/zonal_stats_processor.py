"""
Zonal statistics processor for raster-to-catchment attribute calculations.

Computes zonal statistics (elevation, soil class, land class) from raster
datasets within catchment boundaries using the rasterstats library.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore


class DataPreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        self.logger.info("Calculating elevation statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)

        dem_name = self.config.get('DEM_NAME')
        if dem_name == "default":
            dem_name = f"domain_{self.config.get('DOMAIN_NAME')}_elv.tif"

        dem_path = self._get_file_path('DEM_PATH', 'attributes/elevation/dem', dem_name)
        dem_name = self.config.get('INTERSECT_DEM_NAME')
        if dem_name == 'default':
            dem_name = 'catchment_with_dem.shp'
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem', dem_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(intersect_path), exist_ok=True)

        catchment_gdf = gpd.read_file(catchment_path)
        nodata_value = self.get_nodata_value(dem_path)

        with rasterio.open(dem_path) as src:
            affine = src.transform
            dem_data = src.read(1)

        stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
        result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
        
        if 'elev_mean' in catchment_gdf.columns:
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']
        else:
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']

        result_df = result_df.drop(columns=['elev_mean_new'])
        catchment_gdf.to_file(intersect_path)

    def calculate_soil_stats(self):
        self.logger.info("Calculating soil statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        soil_name = self.config.get('SOIL_CLASS_NAME')
        if soil_name == 'default':
            soil_name = f"domain_{self.config.get('DOMAIN_NAME')}_soil_classes.tif"
        soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass/', soil_name)
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids', intersect_soil_name)
        self.logger.info(f'processing landclasses: {soil_path}')

        if not intersect_path.exists() or self.config.get('FORCE_RUN_ALL_STEPS') == True:
            intersect_path.parent.mkdir(parents=True, exist_ok=True)

            catchment_gdf = gpd.read_file(catchment_path)
            nodata_value = self.get_nodata_value(soil_path)

            with rasterio.open(soil_path) as src:
                affine = src.transform
                soil_data = src.read(1)

            stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats)
            
            # Find the most common soil class (excluding 'count' column)
            soil_columns = [col for col in result_df.columns if col != 'count']
            most_common_soil = result_df[soil_columns].sum().idxmax()
            
            # Fill NaN values with the most common soil class (fallback in case very small HRUs)
            if result_df.isna().any().any():
                self.logger.warning("NaN values found in soil statistics. Filling with most common soil class. Please check HRU's size or use higher resolution land class raster")
                result_df = result_df.fillna({col: (0 if col == 'count' else most_common_soil) for col in result_df.columns})            

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            result_df = result_df.astype({col: int for col in result_df.columns if col != 'count'})

            # Merge with original GeoDataFrame
            for col in result_df.columns:
                if col != 'count':
                    catchment_gdf[col] = result_df[col]

            try:
                catchment_gdf.to_file(intersect_path)
                self.logger.info(f"Soil statistics calculated and saved to {intersect_path}")
            except Exception as e:
                self.logger.error(f"Failed to save file: {e}")
                raise

    def calculate_land_stats(self):
        self.logger.info("Calculating land statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        land_name = self.config.get('LAND_CLASS_NAME')
        if land_name == 'default':
            land_name = f"domain_{self.config.get('DOMAIN_NAME')}_land_classes.tif"
        land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass/', land_name)
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass', intersect_name)
        self.logger.info(f'processing landclasses: {land_path}')

        if not intersect_path.exists() or self.config.get('FORCE_RUN_ALL_STEPS') == True:
            intersect_path.parent.mkdir(parents=True, exist_ok=True)

            catchment_gdf = gpd.read_file(catchment_path)
            nodata_value = self.get_nodata_value(land_path)

            with rasterio.open(land_path) as src:
                affine = src.transform
                land_data = src.read(1)

            stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats)
            
            # Find the most common land class (excluding 'count' column)
            land_columns = [col for col in result_df.columns if col != 'count']
            most_common_land = result_df[land_columns].sum().idxmax()
            
            # Fill NaN values with the most common land class (fallback in case very small HRUs)
            if result_df.isna().any().any():
                self.logger.warning("NaN values found in land statistics. Filling with most common land class. Please check HRU's size or use higher resolution land class raster")
                result_df = result_df.fillna({col: (0 if col == 'count' else most_common_land) for col in result_df.columns})

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            result_df = result_df.astype({col: int for col in result_df.columns if col != 'count'})

            # Merge with original GeoDataFrame
            for col in result_df.columns:
                if col != 'count':
                    catchment_gdf[col] = result_df[col]

            try:
                catchment_gdf.to_file(intersect_path)
                self.logger.info(f"Land statistics calculated and saved to {intersect_path}")
            except Exception as e:
                self.logger.error(f"Failed to save file: {e}")
                raise

    def process_zonal_statistics(self):
        self.calculate_elevation_stats()
        self.calculate_soil_stats()
        self.calculate_land_stats()
        self.logger.info("All zonal statistics processed")

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
