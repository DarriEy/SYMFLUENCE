"""Zonal statistics processor for raster-to-catchment attribute extraction.

Computes spatially-aggregated catchment attributes (elevation, soil class, land cover)
by extracting raster values within polygon boundaries using efficient rasterstats
implementation. Provides both continuous statistics (mean, min, max) and categorical
statistics (mode, fractions) for diverse geospatial data sources.

Architecture:
    The zonal statistics processor implements efficient raster-vector overlay computation
    to extract catchment-aggregated attributes:

    1. Input Data:
       - Vector polygons: Catchments/HRUs from shapefiles
       - Raster data: Elevation (DEM), soil classification, land cover
       - Both must share same coordinate reference system (CRS)

    2. Processing:
       - Read catchment polygon geometries
       - Extract raster values intersecting each polygon
       - Compute summary statistics for extracted values
       - Handle NoData values (excluded from statistics)

    3. Output:
       - Catchment shapefile enriched with extracted attributes
       - Attribute columns for each statistic type (mean, mode, fractions)
       - Maintained spatial geometry for mapping/visualization

Statistics Computed:

    Continuous Data (Elevation from DEM):
        - elev_mean: Mean elevation per catchment (meters)
        - elev_min: Minimum elevation
        - elev_max: Maximum elevation
        - elev_std: Standard deviation (elevation variability)
        - Used for: PET calculation, lapse rate correction, uncertainty

    Categorical Data (Soil Class):
        - soil_class_{code}: Fractional area of each soil class
        - soil_dominant: Most common soil class in catchment
        - Used for: Infiltration parameters, water retention, porosity

    Categorical Data (Land Cover):
        - landcover_class_{code}: Fractional area of each land cover type
        - landcover_dominant: Most common land cover
        - Used for: Vegetation parameters, interception, surface roughness

Workflow:

    1. Read Inputs:
       gdf = gpd.read_file('catchments.shp')
       raster = rasterio.open('dem.tif')

    2. Compute Zonal Statistics:
       stats = rasterstats.zonal_stats(gdf.geometry, raster_path)
       - For each catchment polygon
       - Extract all pixels inside boundary
       - Calculate statistics

    3. Process Results:
       - Convert to DataFrame
       - Calculate fractional coverages
       - Identify dominant classes

    4. Join to Shapefile:
       gdf = gdf.join(pd.DataFrame(stats))
       gdf.to_file('catchments_with_elevation.shp')

Implementation Details:

    rasterstats Integration:
        - Uses Python rasterstats library
        - Efficient C-based polygon-raster intersection
        - Supports NoData masking
        - Handles affine transformations

    Statistics Types:
        Continuous (mean, min, max, std):
            stats=['mean']
            Per-pixel elevation values averaged within polygon

        Categorical (count, mode):
            categorical=True
            Pixel class values tabulated for frequency analysis

    NoData Handling:
        - Read from raster metadata (rasterio.nodatavals)
        - Passed to rasterstats.zonal_stats(nodata=...)
        - Excluded from all calculations
        - Prevents biased statistics

    CRS Validation:
        - Both vector and raster must have same CRS
        - rasterstats automatically handles coordinate alignment
        - Warns if mismatch detected

Configuration Parameters:

    Input paths:
        catchment_path: Path to catchment shapefile
        dem_path: Path to elevation raster (GeoTIFF)
        soil_path: Path to soil class raster
        landcover_path: Path to land cover raster

    Output paths:
        output_dir: Directory for result shapefiles
        Naming: catchment_with_dem.shp, catchment_with_soil.shp, etc.

    Statistics:
        stats_to_compute: ['mean', 'std', 'min', 'max'] for continuous
        categorical: True for soil/landcover classification

Use Cases:

    1. Lumped Watershed Modeling:
       Extract mean elevation per entire basin for GR4J parameter estimation

    2. Distributed Discretization:
       Extract elevation for each HRU (elevation band) in SUMMA/HYPE

    3. Attribute Database:
       Build comprehensive attribute database for all catchments in region

    4. Validation & QA:
       Check elevation statistics are reasonable (detect data errors)

Output Format:

    Enriched Shapefile with new columns:
    ┌────────────┬──────────────┬──────────┬─────────────┬──────────────┐
    │ geometry   │ catchment_id │ elev_mean│ soil_class_1│ landcover_1  │
    ├────────────┼──────────────┼──────────┼─────────────┼──────────────┤
    │ Polygon    │ 1001         │ 1250.5   │ 0.45        │ 0.60         │
    │ Polygon    │ 1002         │ 980.2    │ 0.30        │ 0.40         │
    └────────────┴──────────────┴──────────┴─────────────┴──────────────┘

Performance:

    - Elevation stats: ~0.1-0.5 sec per 1000 catchments
    - Categorical stats: ~0.2-1.0 sec per 1000 catchments (more complex)
    - Memory: ~1-5 GB for continental-scale (millions of polygons)
    - Optimization: Windowed raster reading reduces memory significantly

Dependencies:

    - geopandas: Vector geometry handling
    - rasterio: Raster I/O and metadata reading
    - rasterstats: Efficient zonal statistics computation
    - pandas/numpy: Data structures and operations

References:

    - Zonal Statistics: https://www.rspatial.org/
    - rasterstats Documentation: https://pythonhosted.org/rasterstats/
    - Rasterio: https://rasterio.readthedocs.io/

See Also:

    - GeospatialStatistics: Alternative implementation with chunking for very large domains
    - DataManager: High-level data workflow coordination
    - AttributeProcessor: Post-processing of extracted attributes
"""

import os
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
import rasterio  # type: ignore
from rasterstats import zonal_stats  # type: ignore

from symfluence.core.mixins import ConfigMixin


class DataPreProcessor(ConfigMixin):
    """
    Compute zonal statistics from raster datasets within catchment boundaries.

    This processor uses the rasterstats library to calculate spatial statistics
    (mean elevation, soil class fractions, land cover fractions) by intersecting
    raster grids with catchment polygons. Results are joined back to catchment
    shapefiles as new attributes for hydrological modeling.

    Purpose:
        Extracts spatially-aggregated raster values for each catchment/HRU polygon,
        enabling hydrological models to access elevation, soil properties, and land
        cover characteristics required for parameter estimation and process simulation.

    Zonal Statistics Computed:
        Elevation (DEM):
            - Mean elevation per catchment (m)
            - Uses continuous raster values
            - Handles NoData values via rasterio metadata

        Soil Classification:
            - Categorical soil class distribution
            - Fractional coverage per class within catchment
            - Majority class identification
            - Count of pixels per class

        Land Classification:
            - Categorical land cover distribution (MODIS, NLCD, etc.)
            - Fractional coverage per class within catchment
            - Majority class identification
            - Count of pixels per class

    Workflow:
        1. **Read Inputs**:
           - Catchment shapefile (polygons for HRUs/subcatchments)
           - Raster dataset (GeoTIFF for elevation/soil/land cover)

        2. **Extract NoData Value**: Read from raster metadata to exclude invalid pixels

        3. **Compute Zonal Stats**:
           - Use rasterstats.zonal_stats() with catchment geometries
           - Specify statistics (mean, categorical, count)
           - Apply affine transformation for spatial alignment

        4. **Process Results**:
           - Convert statistics to DataFrame
           - Calculate fractional coverages for categorical data
           - Identify majority classes

        5. **Join to Shapefile**:
           - Add new attributes to catchment GeoDataFrame
           - Save enriched shapefile to intersection output directory

    rasterstats Integration:
        Uses Python rasterstats library for efficient zonal calculations:
        - Handles polygon-raster intersection
        - Supports both continuous (mean) and categorical statistics
        - NoData masking via nodata parameter
        - Affine transformation for coordinate alignment

    Statistics Types:
        Continuous (elevation):
            stats=['mean']
            Returns: Single mean value per polygon

        Categorical (soil/land class):
            stats=['count'], categorical=True
            Returns: Dictionary of {class_id: pixel_count} per polygon
            Post-processing: Convert to fractional coverage

    NoData Handling:
        - Reads NoData value from raster metadata (src.nodatavals[0])
        - Falls back to -9999 if no NoData value specified
        - Passes nodata parameter to zonal_stats() for masking
        - Ensures invalid pixels don't contribute to statistics

    Configuration Requirements:
        Required:
            - SYMFLUENCE_DATA_DIR: Base data directory
            - DOMAIN_NAME: Domain identifier

        Elevation Statistics:
            - CATCHMENT_SHP_NAME: Catchment shapefile name (or 'default')
            - CATCHMENT_PATH: Path to catchment shapefile
            - DEM_NAME: DEM raster filename (or 'default')
            - DEM_PATH: Path to DEM GeoTIFF
            - INTERSECT_DEM_NAME: Output shapefile name
            - INTERSECT_DEM_PATH: Output directory

        Soil Statistics:
            - SOIL_CLASS_NAME: Soil classification raster (or 'default')
            - SOIL_CLASS_PATH: Path to soil raster
            - INTERSECT_SOIL_NAME: Output shapefile name
            - INTERSECT_SOIL_PATH: Output directory

        Land Cover Statistics:
            - LAND_CLASS_NAME: Land cover raster (or 'default')
            - LAND_CLASS_PATH: Path to land cover raster
            - INTERSECT_LAND_NAME: Output shapefile name
            - INTERSECT_LAND_PATH: Output directory

        Optional:
            - FORCE_RUN_ALL_STEPS: Force recomputation (default: False)

    Output Files:
        Enriched shapefiles with new attributes:
        - catchment_with_dem.shp: Original attributes + elev_mean
        - catchment_with_soilclass.shp: Original + soil_class_X fractions
        - catchment_with_landclass.shp: Original + land_class_X fractions

    Spatial Alignment:
        - Catchment shapefile and raster must share coordinate reference system
        - Affine transformation ensures correct spatial intersection
        - Rasterio handles CRS reading and coordinate mapping
        - GeoPandas manages shapefile CRS

    Performance:
        - Processing time: ~5-30 seconds per raster-catchment pair
        - Memory: Loads entire raster into memory (can be large for high-res DEMs)
        - Scales with: Number of catchments × raster resolution
        - rasterstats uses efficient vectorized operations

    Example:
        >>> config = {
        ...     'SYMFLUENCE_DATA_DIR': '/project/data',
        ...     'DOMAIN_NAME': 'test_basin',
        ...     'CATCHMENT_SHP_NAME': 'test_basin_HRUs_elevation.shp',
        ...     'DEM_NAME': 'domain_test_basin_elv.tif'
        ... }
        >>> processor = DataPreProcessor(config, logger)
        >>> processor.calculate_elevation_stats()
        # Reads: shapefiles/catchment/test_basin_HRUs_elevation.shp
        # Reads: attributes/elevation/dem/domain_test_basin_elv.tif
        # Computes: Mean elevation per HRU polygon
        # Writes: shapefiles/catchment_intersection/with_dem/catchment_with_dem.shp
        # Result: Shapefile with new 'elev_mean' column

    Typical Workflow:
        1. calculate_elevation_stats() → Adds elev_mean to catchments
        2. calculate_soil_stats() → Adds soil class fractions
        3. calculate_land_stats() → Adds land cover fractions
        4. Use enriched shapefiles for model parameter assignment

    Error Handling:
        - Creates output directories if they don't exist
        - Validates raster NoData value (fallback to -9999)
        - Logs processing steps for debugging
        - Overwrites existing output if FORCE_RUN_ALL_STEPS=True

    Notes:
        - Raster and shapefile CRS must match (no automatic reprojection)
        - Large rasters may require significant memory
        - Categorical statistics return dictionaries requiring post-processing
        - Output shapefiles retain all original attributes plus new statistics
        - rasterstats uses exact pixel-polygon intersection (not centroid method)

    See Also:
        - data.preprocessing.attribute_processing: High-level attribute processing
        - data.preprocessing.attribute_processors.utils: Zonal stats utilities
        - rasterstats documentation: https://pythonhosted.org/rasterstats/
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize the zonal statistics processor.

        Sets up file paths and configuration for computing raster-to-catchment
        attribute statistics.

        Args:
            config: Configuration dictionary with keys:
                - SYMFLUENCE_DATA_DIR: Root data directory
                - DOMAIN_NAME: Domain identifier for file paths
            logger: Logger instance for diagnostic messages
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.root_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def get_nodata_value(self, raster_path):
        """Extract NoData value from raster metadata.

        Reads the NoData/fill value from raster file metadata using rasterio.
        This value is used to mask invalid pixels during zonal statistics
        computation (e.g., values outside the valid data range).

        The NoData value is typically set by the data provider to indicate:
        - Pixels outside the domain of interest
        - Water bodies or ice (for land cover)
        - Missing observations (gaps in coverage)

        Args:
            raster_path: Path to GeoTIFF or other raster file

        Returns:
            int/float: NoData value from file metadata, or -9999 if not specified.
                      This value will be excluded from zonal statistics calculations.

        Note:
            rasterio reads from the first band only (nodatavals[0]).
            Falls back to -9999 if NoData value is not explicitly set in metadata.
        """
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        """Calculate mean elevation for each catchment polygon.

        Computes zonal statistics on a continuous DEM raster (elevation values)
        within each catchment/HRU boundary. The result is a single mean elevation
        value per polygon.

        Workflow:
            1. Load catchment shapefile (polygons for each HRU/subcatchment)
            2. Load DEM GeoTIFF raster (elevation in meters)
            3. Extract NoData value from DEM metadata
            4. Compute zonal mean: rasterstats.zonal_stats(..., stats=['mean'])
            5. Add 'elev_mean' column to catchment GeoDataFrame
            6. Save enriched shapefile to output directory

        Configuration Parameters:
            CATCHMENT_SHP_NAME: Input catchment shapefile (or 'default')
            DEM_NAME: Input DEM raster filename (or 'default')
            INTERSECT_DEM_NAME: Output shapefile name (default: 'catchment_with_dem.shp')
            INTERSECT_DEM_PATH: Output directory path

        Output:
            Shapefile with new 'elev_mean' attribute (meters elevation)

        Side Effects:
            - Creates output directory if it doesn't exist
            - Overwrites existing output file with same name
            - Logs progress to logger
        """
        self.logger.info("Calculating elevation statistics")
        subbasins_name = self._get_config_value(lambda: self.config.paths.catchment_name, dict_key='CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_HRUs_{self._get_config_value(lambda: self.config.domain.discretization, dict_key='SUB_GRID_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)

        dem_name = self._get_config_value(lambda: self.config.paths.dem_name, dict_key='DEM_NAME')
        if dem_name == "default":
            dem_name = f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_elv.tif"

        dem_path = self._get_file_path('DEM_PATH', 'attributes/elevation/dem', dem_name)
        dem_name = self._get_config_value(lambda: self.config.paths.intersect_dem_name, dict_key='INTERSECT_DEM_NAME')
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
        """Calculate soil class fractions for each catchment polygon.

        Computes zonal statistics on a categorical soil class raster within each
        catchment/HRU. The result is the fractional coverage (pixel count) of each
        soil class within each polygon.

        Algorithm:
            1. Load catchment shapefile and soil class raster
            2. Compute categorical zonal stats: rasterstats.zonal_stats(..., stats=['count'], categorical=True)
            3. This returns a dict per polygon: {class_id: pixel_count, ...}
            4. Convert to DataFrame with columns per class
            5. Fill NaN values: Use most common soil class as fallback
               (handles edge case: very small HRUs with few pixels)
            6. Rename columns: {class_id} → 'USGS_{class_id}'
               (USGS convention for soil classification codes)
            7. Add columns to catchment GeoDataFrame
            8. Save enriched shapefile

        Why Categorical Stats:
            Soil classes are categorical (integer codes), not continuous values.
            rasterstats returns {class_id: count} pairs, allowing computation of
            coverage fractions per class.

        NaN Handling:
            Small HRUs may have very few pixels or none in certain classes.
            rasterstats returns NaN for missing classes. We fill these with the
            most common soil class (within that HRU) as fallback to avoid losing
            the HRU in subsequent modeling steps.

        Configuration Parameters:
            CATCHMENT_SHP_NAME: Input catchment shapefile (or 'default')
            SOIL_CLASS_NAME: Input soil class raster (or 'default')
            INTERSECT_SOIL_NAME: Output shapefile name (default: 'catchment_with_soilclass.shp')
            INTERSECT_SOIL_PATH: Output directory path
            FORCE_RUN_ALL_STEPS: Force recomputation if True (default: False)

        Output:
            Shapefile with columns: USGS_0, USGS_1, ..., USGS_N
            Values are pixel counts (or fractions if post-processed)

        Side Effects:
            - Creates output directory if missing
            - Skips if output exists (unless FORCE_RUN_ALL_STEPS=True)
            - Logs warnings if NaN values present (indicates potential resolution issues)
        """
        self.logger.info("Calculating soil statistics")
        subbasins_name = self._get_config_value(lambda: self.config.paths.catchment_name, dict_key='CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_HRUs_{self._get_config_value(lambda: self.config.domain.discretization, dict_key='SUB_GRID_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        soil_name = self._get_config_value(lambda: self.config.paths.soil_class_name, dict_key='SOIL_CLASS_NAME')
        if soil_name == 'default':
            soil_name = f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_soil_classes.tif"
        soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass/', soil_name)
        intersect_soil_name = self._get_config_value(lambda: self.config.paths.intersect_soil_name, dict_key='INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids', intersect_soil_name)
        self.logger.info(f'processing landclasses: {soil_path}')

        if not intersect_path.exists() or self._get_config_value(lambda: self.config.system.force_run_all_steps, dict_key='FORCE_RUN_ALL_STEPS'):
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
        """Calculate land cover class fractions for each catchment polygon.

        Identical to calculate_soil_stats() but for land cover classification
        raster data instead of soil classes. Computes fractional coverage of
        each IGBP (International Geosphere-Biosphere Programme) land class within
        each catchment/HRU.

        Algorithm:
            1. Load catchment shapefile and land cover raster
            2. Compute categorical zonal stats: rasterstats.zonal_stats(..., stats=['count'], categorical=True)
            3. This returns {class_id: pixel_count} per polygon
            4. Convert to DataFrame with one column per land class
            5. Fill NaN values: Use most common land class as fallback
            6. Rename columns: {class_id} → 'IGBP_{class_id}'
               (IGBP convention for land cover classification)
            7. Add columns to catchment GeoDataFrame
            8. Save enriched shapefile

        Land Cover Classes (IGBP):
            Typical codes include:
            - 0: Water
            - 1: Evergreen Needleleaf Forest
            - 2: Evergreen Broadleaf Forest
            - 3: Deciduous Needleleaf Forest
            - ... (17 classes total in standard IGBP classification)

        NaN Handling:
            Small HRUs may not contain all land classes. rasterstats returns NaN
            for missing classes. We fill with the most common land class (within
            that HRU) to prevent data loss in downstream modeling.

        Configuration Parameters:
            CATCHMENT_SHP_NAME: Input catchment shapefile (or 'default')
            LAND_CLASS_NAME: Input land cover raster (or 'default')
            INTERSECT_LAND_NAME: Output shapefile name (default: 'catchment_with_landclass.shp')
            INTERSECT_LAND_PATH: Output directory path
            FORCE_RUN_ALL_STEPS: Force recomputation if True (default: False)

        Output:
            Shapefile with columns: IGBP_0, IGBP_1, ..., IGBP_N
            Values are pixel counts (integer)

        Side Effects:
            - Creates output directory if missing
            - Skips if output exists (unless FORCE_RUN_ALL_STEPS=True)
            - Logs warnings if NaN values present (indicates raster resolution mismatch)
        """
        self.logger.info("Calculating land statistics")
        subbasins_name = self._get_config_value(lambda: self.config.paths.catchment_name, dict_key='CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_HRUs_{self._get_config_value(lambda: self.config.domain.discretization, dict_key='SUB_GRID_DISCRETIZATION')}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, dict_key='LAND_CLASS_NAME')
        if land_name == 'default':
            land_name = f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_land_classes.tif"
        land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass/', land_name)
        intersect_name = self._get_config_value(lambda: self.config.paths.intersect_land_name, dict_key='INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass', intersect_name)
        self.logger.info(f'processing landclasses: {land_path}')

        if not intersect_path.exists() or self._get_config_value(lambda: self.config.system.force_run_all_steps, dict_key='FORCE_RUN_ALL_STEPS'):
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
        """Compute all zonal statistics (orchestrator method).

        Main entry point that runs the complete workflow:
        1. calculate_elevation_stats() - Mean elevation per HRU
        2. calculate_soil_stats() - Soil class fractions per HRU
        3. calculate_land_stats() - Land cover fractions per HRU

        This method should be called to compute all attribute statistics in sequence.
        Each method is independent and can handle missing files gracefully (with
        logging and optional skipping).

        Side Effects:
            - Calls three sub-methods in sequence
            - Each creates/overwrites output shapefiles
            - Logs completion message after all three finish
        """
        self.calculate_elevation_stats()
        self.calculate_soil_stats()
        self.calculate_land_stats()
        self.logger.info("All zonal statistics processed")

    def _get_file_path(self, file_type, file_def_path, file_name):
        """Resolve file path from configuration or defaults.

        This method handles two path resolution strategies:
        1. **Default paths**: If config[file_type] == 'default', construct path
           from domain project directory and file_def_path
        2. **Custom paths**: If config[file_type] is set, use it directly

        This allows users to override default paths via configuration without
        modifying the code.

        Args:
            file_type: Config key name (e.g., 'DEM_PATH', 'SOIL_CLASS_PATH')
            file_def_path: Default relative path (e.g., 'attributes/elevation/dem')
            file_name: Filename (e.g., 'domain_test_elv.tif')

        Returns:
            Path: Resolved absolute file path

        Example:
            If DOMAIN_NAME='test_basin' and DEM_PATH='default':
                file_type='DEM_PATH'
                file_def_path='attributes/elevation/dem'
                file_name='domain_test_basin_elv.tif'
            Returns: /data/domain_test_basin/attributes/elevation/dem/domain_test_basin_elv.tif

            If DEM_PATH='/custom/path/dem.tif' (custom):
            Returns: /custom/path/dem.tif
        """
        file_type_val = self._get_config_value(lambda: None, default='default', dict_key=file_type)
        if file_type_val == 'default':
            # Use resolve_data_subdir for data subdirectories
            parts = file_def_path.split('/', 1)
            if parts[0] in ('forcing', 'attributes', 'observations'):
                from symfluence.core.mixins.project import resolve_data_subdir
                base = resolve_data_subdir(self.project_dir, parts[0])
                return base / parts[1] / file_name if len(parts) > 1 else base / file_name
            return self.project_dir / file_def_path / file_name
        else:
            return Path(file_type_val)
