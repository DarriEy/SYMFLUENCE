from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore
import gc
import warnings
from rasterio.windows import from_bounds
from shapely.geometry import box

from symfluence.utils.data.path_manager import PathManager


class GeospatialStatistics:
    """
    Calculates geospatial statistics for catchments (elevation, soil, land cover).

    Uses PathManager for standardized path resolution.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # Use PathManager for path resolution
        self.paths = PathManager(config)
        self.project_dir = self.paths.project_dir

        # Resolve paths using PathManager
        self.catchment_path = self.paths.resolve('CATCHMENT_PATH', 'shapefiles/catchment')

        self.catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')
        if self.catchment_name == 'default':
            discretization = str(config.get('DOMAIN_DISCRETIZATION', '')).replace(',', '_')
            self.catchment_name = f"{self.paths.domain_name}_HRUs_{discretization}.shp"

        dem_name = config.get('DEM_NAME', 'default')
        if dem_name == "default":
            dem_name = f"domain_{self.paths.domain_name}_elv.tif"

        self.dem_path = self.paths.resolve('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self.paths.resolve('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self.paths.resolve('LAND_CLASS_PATH', 'attributes/landclass')

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        """
        Calculate elevation statistics with chunked processing for memory efficiency.
        
        This method processes catchments in spatial tiles or index chunks to avoid 
        OOM on large domains like NWM North America (~2.7M catchments).
        """
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        output_file = intersect_path / intersect_name
        checkpoint_dir = intersect_path / 'checkpoints'
        
        # Check if output already exists
        if output_file.exists():
            try:
                gdf = gpd.read_file(output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics file already exists: {output_file}. Skipping calculation.")
                    return
            except Exception as e:
                self.logger.warning(f"Error checking existing elevation statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating elevation statistics (memory-optimized chunked mode)")
        
        # Load catchment shapefile
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        n_catchments = len(catchment_gdf)
        self.logger.info(f"Loaded {n_catchments:,} catchments")
        
        # Ensure we have a unique ID column for tracking
        if 'chunk_idx' not in catchment_gdf.columns:
            catchment_gdf['chunk_idx'] = catchment_gdf.index
        
        try:
            # Get DEM info
            with rasterio.open(self.dem_path) as src:
                dem_crs = src.crs
                dem_bounds = src.bounds
                self.logger.info(f"DEM CRS: {dem_crs}")
            
            shapefile_crs = catchment_gdf.crs
            self.logger.info(f"Catchment shapefile CRS: {shapefile_crs}")
            
            # Reproject if needed
            if dem_crs != shapefile_crs:
                self.logger.info(f"Reprojecting catchments from {shapefile_crs} to {dem_crs}")
                catchment_gdf_projected = catchment_gdf.to_crs(dem_crs)
            else:
                self.logger.info("CRS match - no reprojection needed")
                catchment_gdf_projected = catchment_gdf
            
            # Initialize results array
            elev_means = np.full(n_catchments, np.nan, dtype=np.float32)
            
            # === CHUNKING STRATEGY ===
            # Choose strategy based on catchment count
            if n_catchments > 500_000:
                # Very large: use spatial tiling
                self.logger.info("Using SPATIAL TILING strategy for very large domain")
                elev_means = self._process_elevation_spatial_tiles(
                    catchment_gdf_projected, elev_means, checkpoint_dir
                )
            elif n_catchments > 50_000:
                # Medium-large: use index-based chunking
                chunk_size = self.config.get('ELEV_CHUNK_SIZE', 10_000)
                self.logger.info(f"Using INDEX CHUNKING strategy ({chunk_size:,} catchments/chunk)")
                elev_means = self._process_elevation_index_chunks(
                    catchment_gdf_projected, elev_means, chunk_size, checkpoint_dir
                )
            else:
                # Small enough to process at once
                self.logger.info("Processing all catchments in single batch")
                stats = zonal_stats(
                    catchment_gdf_projected.geometry,
                    str(self.dem_path),
                    stats=['mean'],
                    nodata=-9999
                )
                for i, stat in enumerate(stats):
                    if stat['mean'] is not None:
                        elev_means[i] = stat['mean']
            
            # Add results to original GeoDataFrame
            catchment_gdf['elev_mean'] = elev_means
            
            # Report statistics
            valid_count = np.sum(~np.isnan(elev_means))
            self.logger.info(f"Computed elevation for {valid_count:,}/{n_catchments:,} catchments ({100*valid_count/n_catchments:.1f}%)")
            
            # Save output
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            self.logger.info(f"Elevation statistics saved to {output_file}")
            
            # Clean up checkpoints on success
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
                self.logger.info("Cleaned up checkpoint files")
            
        except Exception as e:
            self.logger.error(f"Error calculating elevation statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _process_elevation_spatial_tiles(self, gdf, elev_means, checkpoint_dir):
        """
        Process catchments using spatial tiles for memory efficiency.
        
        Divides the domain into a grid and processes each tile independently,
        reading only the required DEM window for each tile.
        """
        from shapely.geometry import box
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get domain bounds
        total_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        
        # Create tile grid - aim for ~50k catchments per tile
        n_catchments = len(gdf)
        target_per_tile = self.config.get('ELEV_TILE_TARGET', 50_000)
        n_tiles_approx = max(1, n_catchments // target_per_tile)
        
        # Calculate grid dimensions
        aspect = (total_bounds[2] - total_bounds[0]) / max(0.001, total_bounds[3] - total_bounds[1])
        n_cols = max(1, int(np.sqrt(n_tiles_approx * aspect)))
        n_rows = max(1, int(n_tiles_approx / n_cols))
        
        self.logger.info(f"Creating {n_rows}x{n_cols} tile grid ({n_rows * n_cols} tiles)")
        
        tile_width = (total_bounds[2] - total_bounds[0]) / n_cols
        tile_height = (total_bounds[3] - total_bounds[1]) / n_rows
        buffer = max(tile_width, tile_height) * 0.01
        
        # Build spatial index
        self.logger.info("Building spatial index...")
        sindex = gdf.sindex
        
        processed_count = 0
        tile_num = 0
        total_tiles = n_rows * n_cols
        
        for row in range(n_rows):
            for col in range(n_cols):
                tile_num += 1
                
                # Calculate tile bounds with buffer
                minx = total_bounds[0] + col * tile_width - buffer
                maxx = total_bounds[0] + (col + 1) * tile_width + buffer
                miny = total_bounds[1] + row * tile_height - buffer
                maxy = total_bounds[1] + (row + 1) * tile_height + buffer
                
                tile_box = box(minx, miny, maxx, maxy)
                
                # Find catchments using spatial index
                possible_matches_idx = list(sindex.intersection(tile_box.bounds))
                
                if not possible_matches_idx:
                    continue
                
                # Filter to catchments whose centroid is in this tile (avoid duplicates)
                tile_gdf = gdf.iloc[possible_matches_idx]
                centroids = tile_gdf.geometry.centroid
                mask = centroids.within(tile_box)
                tile_gdf = tile_gdf[mask]
                tile_indices = tile_gdf['chunk_idx'].values
                
                if len(tile_gdf) == 0:
                    continue
                
                self.logger.info(f"Tile {tile_num}/{total_tiles}: {len(tile_gdf):,} catchments")
                
                # Check for checkpoint
                checkpoint_file = checkpoint_dir / f"tile_{row}_{col}.npy"
                if checkpoint_file.exists():
                    self.logger.debug(f"  Loading from checkpoint")
                    tile_elevs = np.load(checkpoint_file)
                    for i, idx in enumerate(tile_indices):
                        elev_means[idx] = tile_elevs[i]
                    processed_count += len(tile_gdf)
                    continue
                
                try:
                    # Compute with windowed DEM reading
                    tile_elevs = self._compute_elevation_windowed(tile_gdf, tile_box.bounds)
                    
                    for i, idx in enumerate(tile_indices):
                        elev_means[idx] = tile_elevs[i]
                    
                    np.save(checkpoint_file, np.array(tile_elevs, dtype=np.float32))
                    processed_count += len(tile_gdf)
                    
                except Exception as e:
                    self.logger.warning(f"  Error: {str(e)}")
                
                del tile_gdf
                gc.collect()
                
                if tile_num % 20 == 0:
                    self.logger.info(f"Progress: {processed_count:,}/{len(gdf):,} catchments")
        
        return elev_means

    def _process_elevation_index_chunks(self, gdf, elev_means, chunk_size, checkpoint_dir):
        """
        Process catchments in index-based chunks.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        n_catchments = len(gdf)
        n_chunks = (n_catchments + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Processing {n_catchments:,} catchments in {n_chunks} chunks")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_catchments)
            
            checkpoint_file = checkpoint_dir / f"chunk_{chunk_idx}.npy"
            if checkpoint_file.exists():
                self.logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: loading checkpoint")
                elev_means[start_idx:end_idx] = np.load(checkpoint_file)
                continue
            
            self.logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: indices {start_idx:,}-{end_idx:,}")
            
            chunk_gdf = gdf.iloc[start_idx:end_idx]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stats = zonal_stats(
                        chunk_gdf.geometry,
                        str(self.dem_path),
                        stats=['mean'],
                        nodata=-9999,
                        all_touched=False
                    )
                
                chunk_elevs = np.array([
                    s['mean'] if s['mean'] is not None else np.nan 
                    for s in stats
                ], dtype=np.float32)
                
                elev_means[start_idx:end_idx] = chunk_elevs
                np.save(checkpoint_file, chunk_elevs)
                
            except Exception as e:
                self.logger.warning(f"  Error: {str(e)}")
            
            del chunk_gdf
            gc.collect()
        
        return elev_means

    def _compute_elevation_windowed(self, gdf, tile_bounds):
        """
        Compute zonal statistics using windowed DEM reading.
        
        Only reads the portion of the DEM needed for the current tile.
        """
        with rasterio.open(self.dem_path) as src:
            pad = 0.001
            window = from_bounds(
                tile_bounds[0] - pad,
                tile_bounds[1] - pad,
                tile_bounds[2] + pad,
                tile_bounds[3] + pad,
                src.transform
            )
            
            dem_data = src.read(1, window=window)
            dem_transform = src.window_transform(window)
            dem_nodata = src.nodata if src.nodata is not None else -9999
        
        stats = zonal_stats(
            gdf.geometry,
            dem_data,
            affine=dem_transform,
            stats=['mean'],
            nodata=dem_nodata,
            all_touched=False
        )
        
        elevs = [s['mean'] if s['mean'] is not None else np.nan for s in stats]
        
        del dem_data
        gc.collect()
        
        return elevs

    def calculate_soil_stats(self):
        """Calculate soil statistics with output file checking and CRS alignment"""
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one USGS soil class column
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing soil statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing soil statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self.config.get('SOIL_CLASS_NAME')
        if soil_name == 'default':
            soil_name = f"domain_{self.config.get('DOMAIN_NAME')}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name
        
        try:
            # Get CRS information
            with rasterio.open(soil_raster) as src:
                soil_crs = src.crs
                self.logger.info(f"Soil raster CRS: {soil_crs}")
            
            shapefile_crs = catchment_gdf.crs
            self.logger.info(f"Catchment shapefile CRS: {shapefile_crs}")
            
            # Check if CRS match and reproject if needed
            if soil_crs != shapefile_crs:
                self.logger.info(f"CRS mismatch detected. Reprojecting catchment from {shapefile_crs} to {soil_crs}")
                try:
                    catchment_gdf_projected = catchment_gdf.to_crs(soil_crs)
                    self.logger.info("CRS reprojection successful")
                except Exception as e:
                    self.logger.error(f"Failed to reproject CRS: {str(e)}")
                    self.logger.warning("Using original CRS - calculation may fail")
                    catchment_gdf_projected = catchment_gdf.copy()
            else:
                self.logger.info("CRS match - no reprojection needed")
                catchment_gdf_projected = catchment_gdf.copy()

            # Use rasterstats with the raster file path directly
            stats = zonal_stats(
                catchment_gdf_projected.geometry, 
                str(soil_raster),  # Use file path instead of array
                stats=['count'], 
                categorical=True, 
                nodata=-9999
            )
            
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Soil statistics saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error calculating soil statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_land_stats(self):
        """Calculate land statistics with output file checking and CRS alignment"""
        # Get the output path and check if the file already exists
        intersect_path = self.paths.resolve('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one IGBP land class column
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing land statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing land statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self.config.get('LAND_CLASS_NAME')
        if land_name == 'default':
            land_name = f"domain_{self.config.get('DOMAIN_NAME')}_land_classes.tif"
        land_raster = self.land_path / land_name
        
        try:
            # Get CRS information
            with rasterio.open(land_raster) as src:
                land_crs = src.crs
                self.logger.info(f"Land raster CRS: {land_crs}")
            
            shapefile_crs = catchment_gdf.crs
            self.logger.info(f"Catchment shapefile CRS: {shapefile_crs}")
            
            # Check if CRS match and reproject if needed
            if land_crs != shapefile_crs:
                self.logger.info(f"CRS mismatch detected. Reprojecting catchment from {shapefile_crs} to {land_crs}")
                try:
                    catchment_gdf_projected = catchment_gdf.to_crs(land_crs)
                    self.logger.info("CRS reprojection successful")
                except Exception as e:
                    self.logger.error(f"Failed to reproject CRS: {str(e)}")
                    self.logger.warning("Using original CRS - calculation may fail")
                    catchment_gdf_projected = catchment_gdf.copy()
            else:
                self.logger.info("CRS match - no reprojection needed")
                catchment_gdf_projected = catchment_gdf.copy()

            # Use rasterstats with the raster file path directly
            stats = zonal_stats(
                catchment_gdf_projected.geometry, 
                str(land_raster),  # Use file path instead of array
                stats=['count'], 
                categorical=True, 
                nodata=-9999
            )
            
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Land statistics saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error calculating land statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def run_statistics(self):
        """Run all geospatial statistics with checks for existing outputs"""
        self.logger.info("Starting geospatial statistics calculation")
        
        # Count how many steps we're skipping
        skipped = 0
        total = 3  # Total number of statistics operations
        
        # Check soil stats
        intersect_soil_path = self.paths.resolve('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'

        soil_output_file = intersect_soil_path / intersect_soil_name
        
        if soil_output_file.exists():
            try:
                gdf = gpd.read_file(soil_output_file)
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics already calculated: {soil_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 1:
            self.calculate_soil_stats()
        
        # Check land stats
        intersect_land_path = self.paths.resolve('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_land_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_land_name == 'default':
            intersect_land_name = 'catchment_with_landclass.shp'

        land_output_file = intersect_land_path / intersect_land_name
        
        if land_output_file.exists():
            try:
                gdf = gpd.read_file(land_output_file)
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics already calculated: {land_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 2:
            self.calculate_land_stats()
        
        # Check elevation stats
        intersect_dem_path = self.paths.resolve('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_dem_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_dem_name == 'default':
            intersect_dem_name = 'catchment_with_dem.shp'

        dem_output_file = intersect_dem_path / intersect_dem_name
        
        if dem_output_file.exists():
            try:
                gdf = gpd.read_file(dem_output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics already calculated: {dem_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 3:
            self.calculate_elevation_stats()
        
        self.logger.info(f"Geospatial statistics completed: {skipped}/{total} steps skipped, {total-skipped}/{total} steps executed")
