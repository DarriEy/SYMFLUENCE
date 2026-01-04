import os
import re
from pathlib import Path
import easymore # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import shutil
from rasterio.mask import mask # type: ignore
from shapely.geometry import Polygon # type: ignore
import rasterstats # type: ignore
from pyproj import CRS, Transformer # type: ignore
import pyproj # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore
import multiprocessing as mp
import time
import uuid
import sys
from datetime import datetime

import gc
from rasterio.windows import from_bounds
import warnings

from symfluence.utils.common.path_resolver import PathResolverMixin

# Add the path to dataset handlers if not already in sys.path
try:
    from symfluence.utils.data.preprocessing.dataset_handlers import DatasetRegistry
except ImportError:
    # Fallback for development/testing
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils' / 'data' / 'preprocessing'))
        from dataset_handlers import DatasetRegistry
    except ImportError as e:
        raise ImportError(
            f"Cannot import DatasetRegistry. Please ensure dataset handlers are installed. Error: {e}"
        )


def _create_easymore_instance():
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")


class ForcingResampler(PathResolverMixin):
    def __init__(self, config, logger):
        self.config = config
        self.config_dict = config  # For PathResolverMixin compatibility
        self.logger = logger
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        dem_name = self.config.get('DEM_NAME')
        if dem_name == "default":
            dem_name = f"domain_{self.config.get('DOMAIN_NAME')}_elv.tif"

        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{str(self.config.get('DOMAIN_DISCRETIZATION')).replace(',','_')}.shp"
        self.forcing_dataset = self.config.get('FORCING_DATASET').lower()
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/raw_data')
        
        # Initialize dataset-specific handler
        try:
            self.dataset_handler = DatasetRegistry.get_handler(
                self.forcing_dataset, 
                self.config, 
                self.logger, 
                self.project_dir
            )
            self.logger.info(f"Initialized {self.forcing_dataset.upper()} dataset handler")
        except ValueError as e:
            self.logger.error(f"Failed to initialize dataset handler: {str(e)}")
            raise
        
        # Merge forcings if required by dataset
        if self.dataset_handler.needs_merging():
            self.logger.info(f"{self.forcing_dataset.upper()} requires merging of raw files")
            # Ensure the directory exists before calling merge_forcings
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
            self.merge_forcings()

    def run_resampling(self):
        self.logger.info("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.info("Forcing data resampling process completed")

    def merge_forcings(self):
        """
        Merge forcing data files into monthly files using dataset-specific handler.

        This method delegates to the appropriate dataset handler which contains
        all dataset-specific logic for variable mapping, unit conversions, and merging.

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If there are issues with data processing or merging.
            IOError: If there are issues writing output files.
        """
        # Extract year range from configuration
        start_year = int(self.config.get('EXPERIMENT_TIME_START').split('-')[0])
        end_year = int(self.config.get('EXPERIMENT_TIME_END').split('-')[0])
        
        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'
        
        # Delegate to dataset handler
        self.dataset_handler.merge_forcings(
            raw_forcing_path=raw_forcing_path,
            merged_forcing_path=merged_forcing_path,
            start_year=start_year,
            end_year=end_year
        )

    def create_shapefile(self):
        """Create forcing shapefile using dataset-specific handler with check for existing files"""
        self.logger.info(f"Creating {self.forcing_dataset.upper()} shapefile")
        
        # Check if shapefile already exists
        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
        
        if output_shapefile.exists():
            try:
                # Verify the shapefile is valid
                gdf = gpd.read_file(output_shapefile)
                expected_columns = [self.config.get('FORCING_SHAPE_LAT_NAME'), 
                                    self.config.get('FORCING_SHAPE_LON_NAME'), 
                                    'ID', 'elev_m']
                
                if all(col in gdf.columns for col in expected_columns) and len(gdf) > 0:
                    # If bbox is larger than existing shapefile bounds, recreate
                    bbox_str = self.config.get("BOUNDING_BOX_COORDS")
                    if isinstance(bbox_str, str) and "/" in bbox_str:
                        try:
                            lat_max, lon_min, lat_min, lon_max = [float(v) for v in bbox_str.split("/")]
                            lat_min, lat_max = sorted([lat_min, lat_max])
                            lon_min, lon_max = sorted([lon_min, lon_max])
                            minx, miny, maxx, maxy = gdf.total_bounds
                            tol = 1e-6
                            if (lon_min < minx - tol or lon_max > maxx + tol or
                                    lat_min < miny - tol or lat_max > maxy + tol):
                                self.logger.info("Existing forcing shapefile bounds do not cover current bbox. Recreating.")
                            else:
                                self.logger.info(f"Forcing shapefile already exists: {output_shapefile}. Skipping creation.")
                                return output_shapefile
                        except Exception as e:
                            self.logger.warning(f"Error checking bbox vs shapefile bounds: {e}. Recreating.")
                    else:
                        self.logger.info(f"Forcing shapefile already exists: {output_shapefile}. Skipping creation.")
                        return output_shapefile
                else:
                    self.logger.info("Existing forcing shapefile missing expected columns. Recreating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing forcing shapefile: {str(e)}. Recreating.")
        
        # Delegate shapefile creation to dataset handler
        return self.dataset_handler.create_shapefile(
            shapefile_path=self.shapefile_path,
            merged_forcing_path=self.merged_forcing_path,
            dem_path=self.dem_path,
            elevation_calculator=self._calculate_elevation_stats_safe
        )

    def _calculate_elevation_stats_safe(self, gdf, dem_path, batch_size=50):
        """
        Safely calculate elevation statistics with CRS alignment and batching.
        
        Args:
            gdf: GeoDataFrame containing geometries
            dem_path: Path to DEM raster
            batch_size: Number of geometries to process per batch
            
        Returns:
            List of elevation values corresponding to each geometry
        """
        self.logger.info(f"Calculating elevation statistics for {len(gdf)} geometries")
        
        # Initialize elevation column with default value
        elevations = [-9999] * len(gdf)
        
        try:
            # Get CRS information
            with rasterio.open(dem_path) as src:
                dem_crs = src.crs
                self.logger.info(f"DEM CRS: {dem_crs}")
            
            shapefile_crs = gdf.crs
            self.logger.info(f"Shapefile CRS: {shapefile_crs}")
            
            # Check if CRS match and reproject if needed
            if dem_crs != shapefile_crs:
                self.logger.info(f"CRS mismatch detected. Reprojecting geometries from {shapefile_crs} to {dem_crs}")
                try:
                    gdf_projected = gdf.to_crs(dem_crs)
                    self.logger.info("CRS reprojection successful")
                except Exception as e:
                    self.logger.error(f"Failed to reproject CRS: {str(e)}")
                    self.logger.warning("Using original CRS - elevation calculation may fail")
                    gdf_projected = gdf.copy()
            else:
                self.logger.info("CRS match - no reprojection needed")
                gdf_projected = gdf.copy()
            
            # Process in batches to manage memory
            num_batches = (len(gdf_projected) + batch_size - 1) // batch_size
            self.logger.info(f"Processing elevation in {num_batches} batches of {batch_size}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gdf_projected))
                
                self.logger.info(f"Processing elevation batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                
                try:
                    # Get batch of geometries
                    batch_gdf = gdf_projected.iloc[start_idx:end_idx]
                    
                    # Use rasterstats with the raster file path directly (more efficient and handles CRS properly)
                    zs = rasterstats.zonal_stats(
                        batch_gdf.geometry, 
                        str(dem_path),  # Use file path instead of array
                        stats=['mean'],
                        nodata=-9999  # Explicit nodata value
                    )
                    
                    # Update elevation values
                    for i, item in enumerate(zs):
                        idx = start_idx + i
                        elevations[idx] = item['mean'] if item['mean'] is not None else -9999
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating elevations for batch {batch_idx+1}: {str(e)}")
                    # Continue with next batch - elevations remain -9999 for failed batch
            
            valid_elevations = sum(1 for elev in elevations if elev != -9999)
            self.logger.info(f"Successfully calculated elevation for {valid_elevations}/{len(elevations)} geometries")
            
        except Exception as e:
            self.logger.error(f"Error in elevation calculation: {str(e)}")
            # Return all -9999 values on error
            elevations = [-9999] * len(gdf)
        
        return elevations

    def remap_forcing(self):
        self.logger.info("Starting forcing remapping process")
        
        # Check for point-scale bypass
        if self.config.get('DOMAIN_DEFINITION_METHOD', '').lower() == 'point':
            self.logger.info("Point-scale domain detected. Using simplified extraction instead of EASYMORE remapping.")
            self._process_point_scale_forcing()
        else:
            self._create_parallelized_weighted_forcing()
            
        self.logger.info("Forcing remapping process completed")

    def _process_point_scale_forcing(self):
        """
        Simplified extraction for point-scale models.
        Just takes the single grid cell available in the merged forcing files.
        """
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)
        
        forcing_files = sorted([f for f in self.merged_forcing_path.glob('*.nc')])
        
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        # Create minimal intersected CSV for SUMMA preprocessor
        case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
        intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"
        
        if not intersect_csv.exists():
            self.logger.info(f"Creating minimal intersection artifact: {intersect_csv.name}")
            # Get HRU info
            target_shp_path = self.catchment_path / self.catchment_name
            target_gdf = gpd.read_file(target_shp_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID')
            
            # Create a 1-to-1 mapping for the point
            # SUMMA expects specific names like S_1_elev_m, S_2_elev_m if lapse rates enabled
            # and S_1_GRU_ID / S_1_HRU_ID for grouping
            hru_id_field_val = target_gdf[hru_id_field].values
            df_int = pd.DataFrame({
                hru_id_field: hru_id_field_val,
                'S_1_HRU_ID': hru_id_field_val,
                'S_1_GRU_ID': target_gdf['GRU_ID'].values if 'GRU_ID' in target_gdf.columns else [1],
                'ID': [1] * len(target_gdf),
                'weight': [1.0] * len(target_gdf),
                'S_1_elev_m': target_gdf['elev_mean'].values if 'elev_mean' in target_gdf.columns else [1600.0],
                'S_2_elev_m': [1600.0] * len(target_gdf) # Forcing elevation
            })
            df_int.to_csv(intersect_csv, index=False)

        for file in forcing_files:
            output_file = self._determine_output_filename(file)
            if output_file.exists() and not self.config.get('FORCE_RUN_ALL_STEPS', False):
                continue
                
            self.logger.info(f"Extracting point forcing: {file.name}")
            with xr.open_dataset(file) as ds:
                # Instead of mean(), use isel to pick the first cell if it's a grid
                # This is safer for preserving all data variables
                spatial_dims = {d: 0 for d in ds.dims if d not in ['time', 'hru']}

                # Check for empty spatial dimensions
                for dim_name, idx in spatial_dims.items():
                    if dim_name in ds.dims and ds.sizes[dim_name] == 0:
                        raise ValueError(
                            f"Cannot extract point forcing from {file.name}: dimension '{dim_name}' has size 0. "
                            f"This typically happens when the forcing file was downloaded for a bounding box smaller "
                            f"than the dataset's resolution. Please use a larger bounding box or a higher-resolution "
                            f"forcing dataset for small domains."
                        )

                ds_point = ds.isel(spatial_dims)
                
                # Add HRU dimension and variable if missing (required by model runners)
                if 'hru' not in ds_point.dims:
                    ds_point = ds_point.expand_dims(hru=[1])
                
                if 'hruId' not in ds_point.data_vars:
                    # Use the first HRU ID from the intersection if available, or default to 1
                    hru_ids = [1]
                    if intersect_csv.exists():
                        try:
                            df_int = pd.read_csv(intersect_csv)
                            hru_ids = df_int[self.config.get('CATCHMENT_SHP_HRUID')].values.astype('int32')
                        except Exception:
                            pass
                    ds_point['hruId'] = (('hru',), hru_ids)
                
                # Ensure correct dimension order (time, hru)
                # Important: some variables might be (hru, time) if they were 1D originally
                # We want everything to be (time, hru)
                for var in ds_point.data_vars:
                    if 'time' in ds_point[var].dims and 'hru' in ds_point[var].dims:
                        ds_point[var] = ds_point[var].transpose('time', 'hru')
                
                # Drop coordinates that are no longer relevant to avoid EASYMORE/SUMMA confusion
                coords_to_drop = ['latitude', 'longitude', 'lat', 'lon', 'expver']
                ds_point = ds_point.drop_vars([c for c in coords_to_drop if c in ds_point.coords or c in ds_point.data_vars])

                # Clear encoding and potentially conflicting attributes
                for var in ds_point.variables:
                    ds_point[var].encoding = {}
                    if 'missing_value' in ds_point[var].attrs:
                        del ds_point[var].attrs['missing_value']
                    if '_FillValue' in ds_point[var].attrs:
                        del ds_point[var].attrs['_FillValue']

                ds_point.to_netcdf(output_file)
                self.logger.info(f"✓ Created point forcing: {output_file.name}")

    def _determine_output_filename(self, input_file):
        """
        Determine the expected output filename for a given input file.
        This handles different forcing datasets with their specific naming patterns.
        
        Args:
            input_file (Path): Input forcing file path
        
        Returns:
            Path: Expected output file path
        """
        # Extract base information
        domain_name = self.config.get('DOMAIN_NAME')
        forcing_dataset = self.config.get('FORCING_DATASET')
        input_stem = input_file.stem
        
        # Handle RDRS and CASR specific naming patterns
        if forcing_dataset.lower() == 'rdrs':
            # For files like "RDRS_monthly_198001.nc", output should be "Canada_RDRS_remapped_RDRS_monthly_198001.nc"
            if input_stem.startswith('RDRS_monthly_'):
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
            else:
                # General fallback for other RDRS files
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        elif forcing_dataset.lower() == 'casr':
            # For files like "CASR_monthly_198001.nc", output should be "Canada_CASR_remapped_CASR_monthly_198001.nc"
            if input_stem.startswith('CASR_monthly_'):
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
            else:
                # General fallback for other CASR files
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        elif forcing_dataset.lower() in ('carra', 'cerra'):
            start_str = self.config.get('EXPERIMENT_TIME_START')
            try:
                dt_start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
                time_tag = dt_start.strftime("%Y-%m-%d-%H-%M-%S")
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{time_tag}.nc"
            except Exception:
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        elif forcing_dataset.lower() == 'era5':
            # ERA5 merged files often end with YYYYMM or YYYYMMDD (e.g., *_200401)
            date_match = re.search(r"(\d{8})$", input_stem) or re.search(r"(\d{6})$", input_stem)
            if date_match:
                date_str = date_match.group(1)
                if len(date_str) == 6:
                    dt = datetime.strptime(date_str, "%Y%m")
                    time_tag = dt.strftime("%Y-%m-01-00-00-00")
                else:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    time_tag = dt.strftime("%Y-%m-%d-00-00-00")
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{time_tag}.nc"
            else:
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        else:
            # General pattern for other datasets (ERA5, etc.)
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        
        return self.forcing_basin_path / output_filename

    def _create_parallelized_weighted_forcing(self):
        """Create weighted forcing files with proper serial/parallel handling for HPC environments"""
        self.logger.info("Creating weighted forcing files")
        
        # Create output directories if they don't exist
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of forcing files
        forcing_path = self.merged_forcing_path
        forcing_files = sorted([f for f in forcing_path.glob('*.nc')])
        
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return
        
        self.logger.info(f"Found {len(forcing_files)} forcing files to process")
        
        # STEP 1: Create remapping weights ONCE (not per file)
        remap_file = self._create_remapping_weights_once(forcing_files[0], intersect_path)
        
        # STEP 2: Filter out already processed files
        remaining_files = self._filter_processed_files(forcing_files)
        
        if not remaining_files:
            self.logger.info("All files have already been processed")
            return
        
        # STEP 3: Apply remapping weights to all files
        requested_cpus = int(self.config.get('MPI_PROCESSES', 1))
        max_available_cpus = mp.cpu_count()
        use_parallel = requested_cpus > 1 and max_available_cpus > 1
        
        if use_parallel:
            # Remove the artificial 4-core limit
            num_cpus = min(requested_cpus, max_available_cpus)
            
            # Practical limit based on I/O
            if num_cpus > 20:
                num_cpus = 20
                self.logger.warning(f"Limiting to {num_cpus} CPUs to avoid I/O bottleneck")
            
            # Don't spawn more workers than files
            num_cpus = min(num_cpus, len(remaining_files))
            
            self.logger.info(f"Using parallel processing with {num_cpus} CPUs")
            success_count = self._process_files_parallel(remaining_files, num_cpus, remap_file)
        else:
            self.logger.info("Using serial processing (no multiprocessing)")
            success_count = self._process_files_serial(remaining_files, remap_file)
        
        # Report final results
        already_processed = len(forcing_files) - len(remaining_files)
        self.logger.info(f"Processing complete: {success_count} files processed successfully out of {len(remaining_files)}")
        self.logger.info(f"Total files processed or skipped: {success_count + already_processed} out of {len(forcing_files)}")

    def _filter_processed_files(self, forcing_files):
        """Filter out already processed files"""
        remaining_files = []
        already_processed = 0
        
        for file in forcing_files:
            output_file = self._determine_output_filename(file)
            
            if output_file.exists():
                try:
                    file_size = output_file.stat().st_size
                    if file_size > 1000:
                        self.logger.debug(f"Skipping already processed file: {file.name}")
                        already_processed += 1
                        continue
                    else:
                        self.logger.warning(f"Found potentially corrupted output file {output_file}. Will reprocess.")
                except Exception as e:
                    self.logger.warning(f"Error checking output file {output_file}: {str(e)}. Will reprocess.")
            
            remaining_files.append(file)
        
        self.logger.info(f"Found {already_processed} already processed files")
        self.logger.info(f"Found {len(remaining_files)} files that need processing")
        
        return remaining_files


    def _create_remapping_weights_once(self, sample_forcing_file, intersect_path):
        """
        Create the remapping weights file once using a sample forcing file.
        This is the expensive GIS operation that only needs to be done once.
        
        Returns:
            Path to the remapping netCDF file
        """
        # Ensure shapefiles are in WGS84
        source_shp_path = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
        target_shp_path = self.catchment_path / self.catchment_name
        
        source_shp_wgs84 = self._ensure_shapefile_wgs84(source_shp_path, "_wgs84")
        target_result = self._ensure_shapefile_wgs84(target_shp_path, "_wgs84")
        
        # Handle tuple return from target shapefile
        if isinstance(target_result, tuple):
            target_shp_wgs84, actual_hru_field = target_result
        else:
            target_shp_wgs84 = target_result
            actual_hru_field = self.config.get('CATCHMENT_SHP_HRUID')
        
        # Define remap file path
        case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
        remap_file = intersect_path / f"{case_name}_{actual_hru_field}_remapping.csv"
        
        # Check if remap file already exists
        if remap_file.exists():
            intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"
            intersect_shp = intersect_path / f"{case_name}_intersected_shapefile.shp"
            if intersect_csv.exists() or intersect_shp.exists():
                self.logger.info(f"Remapping weights file already exists: {remap_file}")
                return remap_file
            self.logger.info("Remapping weights found but intersected shapefile missing. Recreating.")
        
        self.logger.info("Creating remapping weights (this is done only once)...")
        
        # Create temporary directory for this operation
        temp_dir = self.project_dir / 'forcing' / 'temp_easymore_weights'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Align target longitudes to 0-360 if the source grid uses that frame
            target_shp_for_easymore = target_shp_wgs84
            disable_lon_correction = False
            try:
                source_gdf = gpd.read_file(source_shp_wgs84)
                source_lon_field = self.config.get('FORCING_SHAPE_LON_NAME')
                if source_lon_field in source_gdf.columns:
                    source_lon_max = float(source_gdf[source_lon_field].max())
                    if source_lon_max > 180:
                        target_gdf = gpd.read_file(target_shp_wgs84)
                        minx, _, maxx, _ = target_gdf.total_bounds
                        if minx < 0 or maxx < 0:
                            from shapely.affinity import translate
                            target_gdf = target_gdf.copy()
                            target_gdf["geometry"] = target_gdf["geometry"].apply(
                                lambda geom: translate(geom, xoff=360) if geom is not None else geom
                            )
                            target_lon_field = self.config.get('CATCHMENT_SHP_LON')
                            if target_lon_field in target_gdf.columns:
                                target_gdf[target_lon_field] = target_gdf[target_lon_field].apply(
                                    lambda v: v + 360 if v < 0 else v
                                )
                            shifted_path = temp_dir / f"{target_shp_wgs84.stem}_lon360.shp"
                            target_gdf.to_file(shifted_path)
                            target_shp_for_easymore = shifted_path
                            disable_lon_correction = True
                            self.logger.info("Shifted target shapefile longitudes to 0-360 for easymore.")
            except Exception as e:
                self.logger.warning(f"Failed to align target longitudes for easymore: {e}")

            # Setup easymore for weight creation only
            esmr = _create_easymore_instance()
            
            esmr.author_name = 'SUMMA public workflow scripts'
            esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
            esmr.case_name = case_name
            # Disable easymore's internal longitude correction which is buggy with recent pandas
            esmr.correction_shp_lon = False
            if disable_lon_correction:
                # Already handled manually
                pass
            
            # Shapefile configuration
            esmr.source_shp = str(source_shp_wgs84)
            esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
            esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')
            esmr.source_shp_ID = self.config.get('FORCING_SHAPE_ID_NAME', 'ID')  # Default to 'ID' if not specified

            esmr.target_shp = str(target_shp_for_easymore)
            esmr.target_shp_ID = actual_hru_field
            esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
            esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')
            
            # NetCDF configuration - use sample file
            # Get coordinate names from dataset handler
            var_lat, var_lon = self.dataset_handler.get_coordinate_names()

            # Detect which SUMMA variables actually exist in the forcing file
            import xarray as xr
            with xr.open_dataset(sample_forcing_file) as ds:
                all_summa_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                available_vars = [v for v in all_summa_vars if v in ds.data_vars]

                if not available_vars:
                    raise ValueError(f"No SUMMA forcing variables found in {sample_forcing_file}. "
                                   f"Available variables: {list(ds.data_vars)}")

                self.logger.info(f"Detected {len(available_vars)}/{len(all_summa_vars)} SUMMA variables in forcing file: {available_vars}")

                # Store detected variables for use in weight application
                self.detected_forcing_vars = available_vars

                # Calculate grid resolution for small grids (needed by EASYMORE)
                lat_vals = ds[var_lat].values
                lon_vals = ds[var_lon].values

                # Handle 1D or 2D coordinate arrays
                if lat_vals.ndim == 1:
                    lat_size = len(lat_vals)
                    lon_size = len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size = lat_vals.shape[0]
                    lon_size = lat_vals.shape[1]
                else:
                    lat_size = 1
                    lon_size = 1

                # Calculate resolution if grid is small
                source_nc_resolution = None
                if lat_size == 1 or lon_size == 1:
                    # For small grids, estimate resolution from the data or use default
                    if lat_vals.ndim == 1:
                        if len(lat_vals) > 1:
                            res_lat = abs(float(lat_vals[1] - lat_vals[0]))
                        else:
                            res_lat = 0.25  # Default for ERA5
                        if len(lon_vals) > 1:
                            res_lon = abs(float(lon_vals[1] - lon_vals[0]))
                        else:
                            res_lon = 0.25  # Default for ERA5
                    else:
                        # 2D arrays - estimate from first row/column
                        res_lat = 0.25
                        res_lon = 0.25

                    source_nc_resolution = max(res_lat, res_lon)
                    self.logger.info(f"Small grid detected ({lat_size}x{lon_size}), setting source_nc_resolution={source_nc_resolution}")

            esmr.source_nc = str(sample_forcing_file)
            esmr.var_names = available_vars
            esmr.var_lat = var_lat
            esmr.var_lon = var_lon
            esmr.var_time = 'time'

            # Set resolution for small grids
            if source_nc_resolution is not None:
                esmr.source_nc_resolution = source_nc_resolution
            
            # Directories
            esmr.temp_dir = str(temp_dir) + '/'
            esmr.output_dir = str(self.forcing_basin_path) + '/'
            
            # Output configuration
            esmr.remapped_dim_id = 'hru'
            esmr.remapped_var_id = 'hruId'
            esmr.format_list = ['f4']
            esmr.fill_value_list = ['-9999']
            
            # Critical: Tell easymore to ONLY create the remapping weights
            esmr.only_create_remap_csv = True
            esmr.save_csv = True
            esmr.sort_ID = False
            
            # Create the weights
            self.logger.info("Running easymore to create remapping weights...")
            esmr.nc_remapper()
            
            # Move the remap file to the final location
            temp_remap = temp_dir / f"{case_name}_remapping.csv"
            candidate_paths = [
                temp_remap,
                self.forcing_basin_path / f"{case_name}_remapping.csv",
            ]
            if not any(path.exists() for path in candidate_paths):
                fallback = list(temp_dir.glob("*remapping*.csv")) + list(self.forcing_basin_path.glob("*remapping*.csv"))
                if fallback:
                    candidate_paths.extend(fallback)

            remap_source = next((path for path in candidate_paths if path.exists()), None)
            if remap_source is not None:
                remap_source.replace(remap_file)
                self.logger.info(f"Remapping weights created: {remap_file}")
            else:
                raise FileNotFoundError(f"Expected remapping file not created: {temp_remap}")
            
            # Move shapefile files
            for shp_file in temp_dir.glob(f"{case_name}_intersected_shapefile.*"):
                shutil.move(str(shp_file), str(intersect_path / shp_file.name))
            
            return remap_file
            
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_files_serial(self, files, remap_file):
        """Process files in serial mode applying pre-computed weights"""
        self.logger.info(f"Processing {len(files)} files in serial mode")
        
        success_count = 0
        total_files = len(files)
        
        for idx, file in enumerate(files):
            self.logger.info(f"Processing file {idx+1}/{total_files}: {file.name}")
            
            try:
                success = self._apply_remapping_weights(file, remap_file)
                if success:
                    success_count += 1
                    self.logger.info(f"✓ Successfully processed {file.name} ({idx+1}/{total_files})")
                else:
                    self.logger.error(f"✗ Failed to process {file.name} ({idx+1}/{total_files})")
            except Exception as e:
                self.logger.error(f"✗ Error processing {file.name}: {str(e)}")
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Progress: {idx+1}/{total_files} files processed ({success_count} successful)")
        
        return success_count

    def _apply_remapping_weights_worker(self, file, remap_file, worker_id):
        """Worker function for parallel processing"""
        try:
            return self._apply_remapping_weights(file, remap_file, worker_id)
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: Error processing {file.name}: {str(e)}")
            return False

    def _apply_remapping_weights(self, file, remap_file, worker_id=None):
        """
        Apply pre-computed remapping weights to a forcing file.
        This is the fast operation that reads weights and applies them.
        
        Args:
            file: Path to forcing file to process
            remap_file: Path to pre-computed remapping weights CSV
            worker_id: Optional worker ID for logging
        
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        worker_str = f"Worker {worker_id}: " if worker_id is not None else ""
        
        try:
            output_file = self._determine_output_filename(file)
            
            # Double-check output doesn't already exist
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    self.logger.debug(f"{worker_str}Output already exists: {file.name}")
                    return True
            
            # Create unique temp directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = self.project_dir / 'forcing' / f'temp_apply_{unique_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Setup easymore to APPLY weights only
                esmr = _create_easymore_instance()
                
                esmr.author_name = 'SUMMA public workflow scripts'
                esmr.case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
                esmr.correction_shp_lon = False
                
                target_shp_path = self.catchment_path / self.catchment_name
                target_result = self._ensure_shapefile_wgs84(target_shp_path, "_wgs84")
                if isinstance(target_result, tuple):
                    target_shp_wgs84, actual_hru_field = target_result
                else:
                    target_shp_wgs84 = target_result
                    actual_hru_field = self.config.get('CATCHMENT_SHP_HRUID')
                
                esmr.target_shp = str(target_shp_wgs84)
                esmr.target_shp_ID = actual_hru_field
                esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
                esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')
                
                # Coordinate variables
                # Get coordinate names from dataset handler
                var_lat, var_lon = self.dataset_handler.get_coordinate_names()
                
                # NetCDF file configuration
                esmr.source_nc = str(file)
                # Use the same detected variables from weight creation
                if hasattr(self, 'detected_forcing_vars') and self.detected_forcing_vars:
                    esmr.var_names = self.detected_forcing_vars
                else:
                    # Fallback to detecting variables from this file
                    import xarray as xr
                    with xr.open_dataset(file) as ds:
                        all_summa_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                        esmr.var_names = [v for v in all_summa_vars if v in ds.data_vars]

                esmr.var_lat = var_lat
                esmr.var_lon = var_lon
                esmr.var_time = 'time'
                
                # Directories
                esmr.temp_dir = str(temp_dir) + '/'
                esmr.output_dir = str(self.forcing_basin_path) + '/'
                
                # Output configuration
                esmr.remapped_dim_id = 'hru'
                esmr.remapped_var_id = 'hruId'
                esmr.format_list = ['f4']
                esmr.fill_value_list = ['-9999']
                
                # Critical: Point to pre-computed weights file
                esmr.remap_csv = str(remap_file)
                esmr.save_csv = False
                esmr.sort_ID = False
                
                # Apply the remapping
                self.logger.debug(f"{worker_str}Applying remapping weights to {file.name}")
                
                # Store list of files before remapping to detect the new one
                files_before = set(self.forcing_basin_path.glob("*.nc"))
                
                esmr.nc_remapper()
                
                # Detect the new file
                files_after = set(self.forcing_basin_path.glob("*.nc"))
                new_files = files_after - files_before
                
                if not output_file.exists() and new_files:
                    # Move/rename the new file to our expected output name
                    actual_output = list(new_files)[0]
                    actual_output.rename(output_file)
                    self.logger.debug(f"{worker_str}Renamed {actual_output.name} to {output_file.name}")
                
            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Verify output
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    elapsed_time = time.time() - start_time
                    self.logger.debug(f"{worker_str}Successfully processed {file.name} in {elapsed_time:.2f} seconds")
                    return True
                else:
                    self.logger.error(f"{worker_str}Output file corrupted (size: {file_size})")
                    return False
            else:
                self.logger.error(f"{worker_str}Output file not created: {output_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"{worker_str}Error processing {file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _process_files_parallel(self, files, num_cpus, remap_file):
        """Process files in parallel mode applying pre-computed weights"""
        self.logger.info(f"Processing {len(files)} in parallel with {num_cpus} CPUs")
        
        batch_size = min(10, len(files))
        total_batches = (len(files) + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {total_batches} batches of up to {batch_size} files each")
        
        success_count = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(files))
            batch_files = files[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(batch_files)} files")
            
            try:
                with mp.Pool(processes=num_cpus) as pool:
                    # Pass remap_file to each worker
                    worker_args = [(file, remap_file, i % num_cpus) for i, file in enumerate(batch_files)]
                    results = pool.starmap(self._apply_remapping_weights_worker, worker_args)
                
                batch_success = sum(1 for r in results if r)
                success_count += batch_success
                
                self.logger.info(f"Batch {batch_num+1}/{total_batches} complete: {batch_success}/{len(batch_files)} successful")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
            
            import gc
            gc.collect()
        
        return success_count

    def _ensure_unique_hru_ids(self, shapefile_path, hru_id_field):
        """
        Ensure HRU IDs are unique in the shapefile. Create new unique IDs if needed.
        
        Args:
            shapefile_path: Path to the shapefile
            hru_id_field: Name of the HRU ID field
            
        Returns:
            tuple: (updated_shapefile_path, actual_hru_id_field_used)
        """
        try:
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            self.logger.info(f"Checking HRU ID uniqueness in {shapefile_path.name}")
            self.logger.info(f"Available fields: {list(gdf.columns)}")
            
            # Check if the HRU ID field exists
            if hru_id_field not in gdf.columns:
                self.logger.error(f"HRU ID field '{hru_id_field}' not found in shapefile.")
                raise ValueError(f"HRU ID field '{hru_id_field}' not found in shapefile")
            
            # Check for uniqueness
            original_count = len(gdf)
            unique_count = gdf[hru_id_field].nunique()
            
            self.logger.info(f"Shapefile has {original_count} rows, {unique_count} unique {hru_id_field} values")
            
            if unique_count == original_count:
                self.logger.info(f"All {hru_id_field} values are unique")
                return shapefile_path, hru_id_field
            
            # Handle duplicate IDs
            self.logger.warning(f"Found {original_count - unique_count} duplicate {hru_id_field} values")
            
            # Create new unique ID field with shorter name (shapefile 10-char limit)
            new_hru_field = "hru_id_new"  # 10 characters max for shapefile compatibility
            
            # Check if we already have a unique field
            if new_hru_field in gdf.columns:
                if gdf[new_hru_field].nunique() == len(gdf):
                    self.logger.info(f"Using existing unique field: {new_hru_field}")
                    gdf_updated = gdf.copy()
                    actual_field = new_hru_field
                else:
                    # Create new unique IDs
                    self.logger.info(f"Creating new unique IDs in field: {new_hru_field}")
                    gdf_updated = gdf.copy()
                    gdf_updated[new_hru_field] = range(1, len(gdf_updated) + 1)
                    actual_field = new_hru_field
            else:
                # Create new unique IDs
                self.logger.info(f"Creating new unique IDs in field: {new_hru_field}")
                gdf_updated = gdf.copy()
                gdf_updated[new_hru_field] = range(1, len(gdf_updated) + 1)
                actual_field = new_hru_field
            
            # Create output path for the fixed shapefile
            output_path = shapefile_path.parent / f"{shapefile_path.stem}_unique_ids.shp"
            
            # Save the updated shapefile
            gdf_updated.to_file(output_path)
            self.logger.info(f"Updated shapefile with unique IDs saved to: {output_path}")
            
            # Verify the fix worked - check what fields actually exist
            verify_gdf = gpd.read_file(output_path)
            self.logger.info(f"Fields in saved shapefile: {list(verify_gdf.columns)}")
            
            # Find the actual field name (may be truncated by shapefile format)
            possible_fields = [col for col in verify_gdf.columns if col.startswith('hru_id')]
            if not possible_fields:
                self.logger.error(f"No hru_id fields found in saved shapefile. Available: {list(verify_gdf.columns)}")
                raise ValueError("Could not find unique HRU ID field in saved shapefile")
            
            # Use the first matching field (should be our new unique field)
            actual_saved_field = possible_fields[0]
            self.logger.info(f"Using field '{actual_saved_field}' from saved shapefile")
            
            if verify_gdf[actual_saved_field].nunique() == len(verify_gdf):
                self.logger.info(f"Verification successful: All {actual_saved_field} values are unique")
                return output_path, actual_saved_field
            else:
                self.logger.error("Verification failed: Still have duplicate IDs after fix")
                raise ValueError("Could not create unique HRU IDs")
                
        except Exception as e:
            self.logger.error(f"Error ensuring unique HRU IDs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _ensure_shapefile_wgs84(self, shapefile_path, output_suffix="_wgs84"):
        """
        Ensure shapefile is in WGS84 (EPSG:4326) for easymore compatibility.
        Creates a WGS84 version if needed and ensures unique HRU IDs.
        
        Args:
            shapefile_path: Path to the shapefile
            output_suffix: Suffix to add to WGS84 version filename
            
        Returns:
            tuple: (wgs84_shapefile_path, hru_id_field_used) for target shapefiles,
                just wgs84_shapefile_path for source shapefiles
        """
        shapefile_path = Path(shapefile_path)
        is_target_shapefile = 'catchment' in str(shapefile_path).lower()
        
        try:
            # Read the shapefile and check its CRS
            gdf = gpd.read_file(shapefile_path)
            current_crs = gdf.crs
            
            self.logger.info(f"Checking CRS for {shapefile_path.name}: {current_crs}")
            
            # For target shapefiles, ensure unique HRU IDs first
            if is_target_shapefile:
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID')
                try:
                    shapefile_path, actual_hru_field = self._ensure_unique_hru_ids(shapefile_path, hru_id_field)
                    # Re-read the potentially updated shapefile
                    gdf = gpd.read_file(shapefile_path)
                    current_crs = gdf.crs
                except Exception as e:
                    self.logger.error(f"Failed to ensure unique HRU IDs: {str(e)}")
                    raise
            
            # Check if already in WGS84
            if current_crs is not None and current_crs.to_epsg() == 4326:
                self.logger.info(f"Shapefile {shapefile_path.name} already in WGS84")
                if is_target_shapefile:
                    return shapefile_path, actual_hru_field
                else:
                    return shapefile_path
            
            # Create WGS84 version
            wgs84_shapefile = shapefile_path.parent / f"{shapefile_path.stem}{output_suffix}.shp"
            
            # Check if WGS84 version already exists and is valid
            if wgs84_shapefile.exists():
                try:
                    wgs84_gdf = gpd.read_file(wgs84_shapefile)
                    if wgs84_gdf.crs is not None and wgs84_gdf.crs.to_epsg() == 4326:
                        # For target shapefiles, also check if it has unique IDs
                        if is_target_shapefile:
                            # Check if the unique field exists (might be truncated)
                            possible_fields = [col for col in wgs84_gdf.columns if col.startswith('hru_id')]
                            if possible_fields and wgs84_gdf[possible_fields[0]].nunique() == len(wgs84_gdf):
                                self.logger.info(f"WGS84 version with unique IDs already exists: {wgs84_shapefile.name}")
                                return wgs84_shapefile, possible_fields[0]
                            else:
                                self.logger.warning(f"Existing WGS84 file missing unique ID field. Recreating.")
                        else:
                            self.logger.info(f"WGS84 version already exists: {wgs84_shapefile.name}")
                            return wgs84_shapefile
                    else:
                        self.logger.warning(f"Existing WGS84 file has wrong CRS: {wgs84_gdf.crs}. Recreating.")
                except Exception as e:
                    self.logger.warning(f"Error reading existing WGS84 file: {str(e)}. Recreating.")
            
            # Convert to WGS84
            self.logger.info(f"Converting {shapefile_path.name} from {current_crs} to WGS84")
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
            
            # Save WGS84 version
            gdf_wgs84.to_file(wgs84_shapefile)
            self.logger.info(f"WGS84 shapefile created: {wgs84_shapefile}")
            
            if is_target_shapefile:
                # Re-read to get the actual field name (may be truncated)
                saved_gdf = gpd.read_file(wgs84_shapefile)
                possible_fields = [col for col in saved_gdf.columns if col.startswith('hru_id')]
                if possible_fields:
                    actual_saved_field = possible_fields[0]
                    self.logger.info(f"Using field '{actual_saved_field}' from WGS84 shapefile")
                    return wgs84_shapefile, actual_saved_field
                else:
                    self.logger.error(f"No hru_id field found in WGS84 shapefile")
                    return wgs84_shapefile, actual_hru_field  # fallback
            else:
                return wgs84_shapefile
            
        except Exception as e:
            self.logger.error(f"Error ensuring WGS84 for {shapefile_path}: {str(e)}")
            raise

    def _process_single_forcing_file_serial(self, file):
        """Process a single forcing file in serial mode with proper WGS84 and unique ID handling"""
        try:
            start_time = time.time()
            
            # Check output file first
            output_file = self._determine_output_filename(file)
            
            if output_file.exists():
                try:
                    file_size = output_file.stat().st_size
                    if file_size > 1000:
                        self.logger.debug(f"Skipping already processed file {file.name}")
                        return True
                except Exception:
                    pass
            
            # Process the file
            file_to_process = file
            
            # Define paths
            intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
            intersect_path.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique temp directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = self.project_dir / 'forcing' / f'temp_easymore_serial_{unique_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Ensure shapefiles are in WGS84 for easymore
                source_shp_path = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
                target_shp_path = self.catchment_path / self.catchment_name
                
                # Convert to WGS84 if needed
                source_shp_wgs84 = self._ensure_shapefile_wgs84(source_shp_path, "_wgs84")
                target_shp_wgs84, actual_hru_field = self._ensure_shapefile_wgs84(target_shp_path, "_wgs84")
                
                # Setup easymore configuration with WGS84 shapefiles
                esmr = _create_easymore_instance()
                
                esmr.author_name = 'SUMMA public workflow scripts'
                esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
                esmr.case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
                
                # Use WGS84 shapefiles
                esmr.source_shp = str(source_shp_wgs84)
                esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
                esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')
                esmr.source_shp_ID = self.config.get('FORCING_SHAPE_ID_NAME', 'ID')  # Default to 'ID' if not specified

                esmr.target_shp = str(target_shp_wgs84)
                esmr.target_shp_ID = actual_hru_field  # Use the actual unique field name
                esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
                esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')
                
                # Set coordinate variable names based on forcing dataset
                if self.forcing_dataset in ['rdrs', 'casr']:
                    var_lat = 'lat' 
                    var_lon = 'lon'
                else:  # era5, carra, etc.
                    var_lat = 'latitude'
                    var_lon = 'longitude'
                
                esmr.source_nc = str(file_to_process)
                esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                esmr.var_lat = var_lat
                esmr.var_lon = var_lon
                esmr.var_time = 'time'
                
                esmr.temp_dir = str(temp_dir) + '/'
                esmr.output_dir = str(self.forcing_basin_path) + '/'
                
                esmr.remapped_dim_id = 'hru'
                esmr.remapped_var_id = 'hruId'
                esmr.format_list = ['f4']
                esmr.fill_value_list = ['-9999']
                
                esmr.save_csv = False
                esmr.remap_csv = ''
                esmr.sort_ID = False
                
                # Handle remap file creation/reuse - include unique field in filename
                remap_file = f"{esmr.case_name}_{actual_hru_field}_remapping.csv"
                remap_path = intersect_path / remap_file
                
                if not remap_path.exists():
                    self.logger.info(f"Creating new remap file for {file.name} using field {actual_hru_field}")
                    esmr.nc_remapper()
                    
                    # Move the remap file to the intersection path
                    temp_remap = Path(esmr.temp_dir) / f"{esmr.case_name}_remapping.csv"
                    if temp_remap.exists():
                        shutil.move(str(temp_remap), str(remap_path))
                        
                    # Move the shapefile files
                    for shp_file in Path(esmr.temp_dir).glob(f"{esmr.case_name}_intersected_shapefile.*"):
                        shutil.move(str(shp_file), str(intersect_path / shp_file.name))
                else:
                    self.logger.debug(f"Using existing remap file for {file.name}")
                    esmr.remap_csv = str(remap_path)
                    esmr.nc_remapper()
                
            finally:
                # Clean up temporary files
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp files: {str(e)}")
            
            # Verify output file exists and is valid
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    elapsed_time = time.time() - start_time
                    self.logger.debug(f"Successfully processed {file.name} in {elapsed_time:.2f} seconds")
                    return True
                else:
                    self.logger.error(f"Output file {output_file} exists but may be corrupted (size: {file_size} bytes)")
                    return False
            else:
                self.logger.error(f"Expected output file {output_file} was not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _process_forcing_file(self, file, worker_id):
        """Process a single forcing file - tuple handling fix"""
        try:
            start_time = time.time()
            
            # Check output file first
            output_file = self._determine_output_filename(file)
            if output_file.exists():
                try:
                    file_size = output_file.stat().st_size
                    if file_size > 1000:
                        self.logger.info(f"Worker {worker_id}: Skipping already processed file {file.name}")
                        return True
                except Exception:
                    pass
            
            self.logger.info(f"Worker {worker_id}: Processing file {file.name}")
            
            # Save current working directory
            original_cwd = os.getcwd()
            
            # For CASR and RDRS, files are already processed during merging
            file_to_process = file
            
            # Define paths
            intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
            intersect_path.mkdir(parents=True, exist_ok=True)
            
            # Generate unique temp directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = self.project_dir / 'forcing' / f'temp_easymore_{unique_id}_{worker_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Get shapefile paths
                source_shp_path = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
                target_shp_path = self.catchment_path / self.catchment_name
                
                # Verify files exist
                if not source_shp_path.exists():
                    self.logger.error(f"Worker {worker_id}: Source shapefile missing: {source_shp_path}")
                    return False
                if not target_shp_path.exists():
                    self.logger.error(f"Worker {worker_id}: Target shapefile missing: {target_shp_path}")
                    return False
                
                # Convert to WGS84 and handle potential tuple returns
                source_result = self._ensure_shapefile_wgs84(source_shp_path, "_wgs84")
                target_result = self._ensure_shapefile_wgs84(target_shp_path, "_wgs84")
                
                # Handle tuple returns - extract just the path
                if isinstance(source_result, tuple):
                    source_shp_wgs84 = Path(source_result[0]).resolve()
                else:
                    source_shp_wgs84 = Path(source_result).resolve()
                    
                if isinstance(target_result, tuple):
                    target_shp_wgs84 = Path(target_result[0]).resolve()
                else:
                    target_shp_wgs84 = Path(target_result).resolve()
                
                # Verify WGS84 files exist
                if not source_shp_wgs84.exists():
                    self.logger.error(f"Worker {worker_id}: WGS84 source shapefile missing: {source_shp_wgs84}")
                    return False
                if not target_shp_wgs84.exists():
                    self.logger.error(f"Worker {worker_id}: WGS84 target shapefile missing: {target_shp_wgs84}")
                    return False
                
                # Change to temp directory to avoid any relative path issues
                os.chdir(temp_dir)
                self.logger.info(f"Worker {worker_id}: Working in temp directory: {temp_dir}")
                
                # Setup easymore configuration with absolute paths
                esmr = _create_easymore_instance()
                
                esmr.author_name = 'SUMMA public workflow scripts'
                esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
                esmr.case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
                
                # Use absolute paths
                esmr.source_shp = str(source_shp_wgs84)
                esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
                esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')
                esmr.source_shp_ID = self.config.get('FORCING_SHAPE_ID_NAME', 'ID')  # Default to 'ID' if not specified

                esmr.target_shp = str(target_shp_wgs84)
                esmr.target_shp_ID = self.config.get('CATCHMENT_SHP_HRUID')
                esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
                esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')
                
                # Set coordinate variable names
                if self.forcing_dataset in ['rdrs', 'casr']:
                    var_lat = 'lat' 
                    var_lon = 'lon'
                else:
                    var_lat = 'latitude'
                    var_lon = 'longitude'
                
                # Use absolute path for NetCDF file
                esmr.source_nc = str(Path(file_to_process).resolve())
                esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                esmr.var_lat = var_lat
                esmr.var_lon = var_lon
                esmr.var_time = 'time'
                
                # Use current directory (temp_dir) for easymore operations
                esmr.temp_dir = './'
                esmr.output_dir = str(self.forcing_basin_path.resolve()) + '/'
                
                esmr.remapped_dim_id = 'hru'
                esmr.remapped_var_id = 'hruId'
                esmr.format_list = ['f4']
                esmr.fill_value_list = ['-9999']
                
                esmr.save_csv = False
                esmr.remap_csv = ''
                esmr.sort_ID = False
                
                # Check for existing remap file
                remap_file = f"{esmr.case_name}_remapping.csv"
                remap_final_path = intersect_path / remap_file
                
                if not remap_final_path.exists():
                    try:
                        self.logger.info(f"Worker {worker_id}: Creating new remap file...")
                        esmr.nc_remapper()
                        
                        # Move files from current directory to final locations
                        if Path(remap_file).exists():
                            shutil.move(remap_file, remap_final_path)
                            self.logger.info(f"Worker {worker_id}: Moved remap file to {remap_final_path}")
                        
                        # Move shapefile files
                        for shp_file in Path('.').glob(f"{esmr.case_name}_intersected_shapefile.*"):
                            shutil.move(shp_file, intersect_path / shp_file.name)
                            self.logger.info(f"Worker {worker_id}: Moved {shp_file.name}")
                            
                    except Exception as e:
                        self.logger.error(f"Worker {worker_id}: Error creating remap file: {str(e)}")
                        import traceback
                        self.logger.error(f"Worker {worker_id}: Traceback: {traceback.format_exc()}")
                        return False
                else:
                    # Use existing remap file
                    self.logger.info(f"Worker {worker_id}: Using existing remap file")
                    esmr.remap_csv = str(remap_final_path.resolve())
                    esmr.nc_remapper()
                    
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
                
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    self.logger.warning(f"Worker {worker_id}: Failed to clean temp files: {e}")
            
            # Verify output
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Worker {worker_id}: Successfully processed {file.name} in {elapsed_time:.2f} seconds")
                    return True
                else:
                    self.logger.error(f"Worker {worker_id}: Output file corrupted (size: {file_size})")
                    return False
            else:
                self.logger.error(f"Worker {worker_id}: Output file not created: {output_file}")
                return False
                    
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: Error processing {file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
