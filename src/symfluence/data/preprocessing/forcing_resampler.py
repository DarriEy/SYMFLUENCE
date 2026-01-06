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
import logging
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import gc
import netCDF4 as nc4
from rasterio.windows import from_bounds
import warnings
from tqdm import tqdm

from symfluence.core.path_resolver import PathResolverMixin

# Add the path to dataset handlers if not already in sys.path
try:
    from symfluence.data.preprocessing.dataset_handlers import DatasetRegistry
except ImportError:
    # Fallback for development/testing
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils' / 'data' / 'preprocessing'))
        from dataset_handlers import DatasetRegistry
    except ImportError as e:
        raise ImportError(
            f"Cannot import DatasetRegistry. Please ensure dataset handlers are installed. Error: {e}"
        )

# Suppress verbose easmore logging
logging.getLogger('easymore').setLevel(logging.WARNING)
logging.getLogger('easymorepy').setLevel(logging.WARNING)

# Suppress numpy deprecation warnings from EASMORE
warnings.filterwarnings('ignore', category=DeprecationWarning, module='easymore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='easymore')


def _run_easmore_with_suppressed_output(esmr, logger):
    """
    Run EASMORE's nc_remapper while suppressing its verbose print output.
    EASMORE prints directly to stdout/stderr, not through logging.

    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    try:
        # Suppress all warnings from EASMORE during execution
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            # Capture both stdout and stderr
            captured_output = StringIO()
            captured_error = StringIO()

            with redirect_stdout(captured_output), redirect_stderr(captured_error):
                esmr.nc_remapper()

            # Get full captured output
            stdout_text = captured_output.getvalue().strip()
            stderr_text = captured_error.getvalue().strip()

            # Check for error indicators in output
            has_error = any(indicator in stderr_text.lower() for indicator in ['error', 'failed', 'exception', 'traceback'])

            if stdout_text:
                # Log first 200 chars at debug level for normal output
                logger.debug(f"EASMORE stdout: {stdout_text[:200]}")

            if stderr_text:
                if has_error:
                    # Log FULL stderr if it contains error indicators
                    logger.warning(f"EASMORE stderr (errors detected):\n{stderr_text}")
                else:
                    # Just first 200 chars for normal stderr
                    logger.debug(f"EASMORE stderr: {stderr_text[:200]}")

            return (True, stdout_text, stderr_text)

    except Exception as e:
        logger.error(f"Error running EASMORE: {str(e)}")
        raise


def _create_easymore_instance():
    """Create an Easymore instance while suppressing initialization output."""
    # EASMORE prints initialization message to stdout when instantiated
    # Capture and suppress this output
    captured_output = StringIO()
    with redirect_stdout(captured_output):
        if hasattr(easymore, "Easymore"):
            instance = easymore.Easymore()
        elif hasattr(easymore, "easymore"):
            instance = easymore.easymore()
        else:
            raise AttributeError("easymore module does not expose an Easymore class")
    return instance


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
            self.logger.debug(f"Initialized {self.forcing_dataset.upper()} dataset handler")
        except ValueError as e:
            self.logger.error(f"Failed to initialize dataset handler: {str(e)}")
            raise
        
        # Merge forcings if required by dataset
        if self.dataset_handler.needs_merging():
            self.logger.debug(f"{self.forcing_dataset.upper()} requires merging of raw files")
            # Ensure the directory exists before calling merge_forcings
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
            self.merge_forcings()

        # Cache for processed shapefile path and HRU field - populated during weight creation
        # and reused during weight application to avoid re-processing (which can cause segfaults)
        self._cached_target_shp_wgs84 = None
        self._cached_hru_field = None

    def run_resampling(self):
        self.logger.debug("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.debug("Forcing data resampling process completed")

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
        self.logger.debug(f"Creating {self.forcing_dataset.upper()} shapefile")
        
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
                                self.logger.debug("Existing forcing shapefile bounds do not cover current bbox. Recreating.")
                            else:
                                self.logger.debug(f"Forcing shapefile already exists. Skipping creation.")
                                return output_shapefile
                        except Exception as e:
                            self.logger.warning(f"Error checking bbox vs shapefile bounds: {e}. Recreating.")
                    else:
                        self.logger.debug(f"Forcing shapefile already exists. Skipping creation.")
                        return output_shapefile
                else:
                    self.logger.debug("Existing forcing shapefile missing expected columns. Recreating.")
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
        Uses rasterio directly to avoid segmentation faults in rasterstats.
        
        Args:
            gdf: GeoDataFrame containing geometries
            dem_path: Path to DEM raster
            batch_size: Number of geometries to process per batch (unused in new implementation but kept for API compatibility)
            
        Returns:
            List of elevation values corresponding to each geometry
        """
        self.logger.info(f"Calculating elevation statistics for {len(gdf)} geometries")
        
        # Initialize elevation column with default value
        elevations = [-9999.0] * len(gdf)
        
        try:
            # Open the raster once
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
                
                self.logger.info(f"Processing elevation for {len(gdf_projected)} geometries using rasterio")
                
                # Iterate over geometries
                # Using tqdm for progress if available, otherwise just iterate
                try:
                    iterator = tqdm(gdf_projected.geometry.iloc, total=len(gdf_projected), desc="Calculating Elevation")
                except ImportError:
                    iterator = gdf_projected.geometry.iloc

                for idx, geom in enumerate(iterator):
                    try:
                        # Skip invalid empty geometries
                        if geom is None or geom.is_empty:
                            continue

                        # rasterio.mask.mask expects a list of geometries (GeoJSON-like)
                        # crop=True creates a smaller array for the masked area
                        out_image, out_transform = mask(src, [geom], crop=True, nodata=-9999)
                        
                        # out_image is (bands, rows, cols)
                        data = out_image[0]
                        
                        # Filter for valid data
                        valid_data = data[data != -9999]
                        
                        if valid_data.size > 0:
                            elevations[idx] = float(np.mean(valid_data))
                        
                    except ValueError:
                        # Usually means geometry is outside raster bounds
                        pass
                    except Exception as e:
                        # Log specific error but continue
                        # Only log first few errors to avoid spamming
                        if idx < 5:
                            self.logger.debug(f"Error calculating elevation for geometry {idx}: {str(e)}")

            valid_elevations = sum(1 for elev in elevations if elev != -9999.0)
            self.logger.info(f"Successfully calculated elevation for {valid_elevations}/{len(elevations)} geometries")
            
        except Exception as e:
            self.logger.error(f"Error in elevation calculation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return elevations

    def remap_forcing(self):
        self.logger.debug("Starting forcing remapping process")

        # Check for point-scale bypass
        if self.config.get('DOMAIN_DEFINITION_METHOD', '').lower() == 'point':
            self.logger.debug("Point-scale domain detected. Using simplified extraction instead of EASYMORE remapping.")
            self._process_point_scale_forcing()
        elif self._should_use_point_scale_extraction():
            # Bypass EASYMORE for tiny grids (1x1) to avoid intersection errors
            self.logger.info("Tiny forcing grid detected (1x1). Using simplified extraction instead of EASYMORE remapping.")
            self._process_point_scale_forcing()
        else:
            self._create_parallelized_weighted_forcing()

        self.logger.debug("Forcing remapping process completed")

    def _should_use_point_scale_extraction(self) -> bool:
        """
        Check if the forcing grid is too small for EASYMORE remapping.

        For 1x1 grids (single forcing cell), EASYMORE intersection often fails
        with 'max() arg is an empty sequence' because there's no meaningful
        spatial intersection to compute. In these cases, point-scale extraction
        is more appropriate and reliable.

        Returns:
            True if point-scale extraction should be used instead of EASYMORE
        """
        try:
            # Find a sample forcing file
            forcing_path = self.merged_forcing_path
            exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']
            all_nc_files = list(forcing_path.glob('*.nc'))
            forcing_files = [
                f for f in all_nc_files
                if not any(pattern in f.name.lower() for pattern in exclude_patterns)
            ]

            if not forcing_files:
                return False

            sample_file = forcing_files[0]
            var_lat, var_lon = self.dataset_handler.get_coordinate_names()

            with xr.open_dataset(sample_file) as ds:
                lat_vals = ds[var_lat].values
                lon_vals = ds[var_lon].values

                # Determine grid size
                if lat_vals.ndim == 1:
                    lat_size = len(lat_vals)
                    lon_size = len(lon_vals)
                elif lat_vals.ndim == 2:
                    lat_size, lon_size = lat_vals.shape
                else:
                    lat_size = lon_size = 1

                # Check if grid is tiny (1x1 or 1xN or Nx1)
                is_tiny = (lat_size <= 1 and lon_size <= 1) or (lat_size * lon_size <= 1)

                if is_tiny:
                    self.logger.info(f"Detected tiny forcing grid: {lat_size}x{lon_size}")
                    return True

                return False

        except Exception as e:
            self.logger.warning(f"Could not check forcing grid size: {e}")
            return False

    def _validate_and_repair_geometries(self, gdf):
        """
        Validate and repair geometries in a GeoDataFrame.

        Dissolve operations can create invalid geometries (self-intersections,
        invalid rings, etc.) that cause bus errors or crashes when processed
        by EASYMORE or other GIS operations.

        Args:
            gdf: GeoDataFrame with potentially invalid geometries

        Returns:
            GeoDataFrame with validated and repaired geometries
        """
        from shapely.validation import make_valid

        invalid_count = 0
        repaired_count = 0

        def repair_geometry(geom):
            nonlocal invalid_count, repaired_count

            if geom is None or geom.is_empty:
                return geom

            if not geom.is_valid:
                invalid_count += 1
                try:
                    # Try buffer(0) first - simple and effective for most cases
                    repaired = geom.buffer(0)
                    if repaired.is_valid and not repaired.is_empty:
                        repaired_count += 1
                        return repaired

                    # Fall back to make_valid for more complex issues
                    repaired = make_valid(geom)
                    if repaired.is_valid and not repaired.is_empty:
                        repaired_count += 1
                        return repaired

                    # If still invalid, log warning but return original
                    self.logger.warning(f"Could not repair geometry: {geom.geom_type}")
                    return geom

                except Exception as e:
                    self.logger.warning(f"Error repairing geometry: {e}")
                    return geom

            return geom

        # Apply repair to all geometries
        gdf = gdf.copy()
        gdf['geometry'] = gdf['geometry'].apply(repair_geometry)

        if invalid_count > 0:
            self.logger.info(
                f"Found {invalid_count} invalid geometries, repaired {repaired_count}"
            )
        else:
            self.logger.debug("All geometries are valid")

        return gdf

    def _process_point_scale_forcing(self):
        """
        Simplified extraction for point-scale models.
        Just takes the single grid cell available in the merged forcing files.
        """
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Get forcing files, excluding non-temporal files
        exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']
        all_nc_files = list(self.merged_forcing_path.glob('*.nc'))
        forcing_files = sorted([
            f for f in all_nc_files
            if not any(pattern in f.name.lower() for pattern in exclude_patterns)
        ])
        
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
        
        # Standardize naming by attempting to extract a date first
        # This prevents duplicates if raw files have different naming schemes for the same date
        import re
        date_tag = None
        
        # Pattern 1: YYYY-MM-DD-HH-MM-SS
        match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", input_stem)
        if match:
            date_tag = match.group(1)
        else:
            # Pattern 2: YYYYMMDD or YYYYMM (look for 6 or 8 digits, potentially preceded by underscore or at start)
            # We look for digits that are likely a year (starting with 19 or 20)
            match = re.search(r"(19|20)(\d{4,6})", input_stem)
            if match:
                date_str = match.group(0)
                try:
                    if len(date_str) == 6:
                        dt = datetime.strptime(date_str, "%Y%m")
                        date_tag = dt.strftime("%Y-%m-01-00-00-00")
                    elif len(date_str) == 8:
                        dt = datetime.strptime(date_str, "%Y%m%d")
                        date_tag = dt.strftime("%Y-%m-%d-00-00-00")
                except ValueError:
                    pass

        if date_tag:
            # Clean, standardized filename
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{date_tag}.nc"
        else:
            # Fallback logic: prevent redundant prefixing
            # If the input_stem already contains the domain_name, don't prepend it again
            clean_stem = input_stem
            if input_stem.startswith(f"domain_{domain_name}"):
                clean_stem = input_stem.replace(f"domain_{domain_name}_", "")
            elif input_stem.startswith(domain_name):
                clean_stem = input_stem.replace(f"{domain_name}_", "")
            
            # Remove forcing dataset from stem if it's there
            clean_stem = clean_stem.replace(f"{forcing_dataset}_", "").replace(f"{forcing_dataset.lower()}_", "")
            # Remove existing "remapped" or "merged" tags
            clean_stem = clean_stem.replace("remapped_", "").replace("merged_", "")
            
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{clean_stem}.nc"
        
        return self.forcing_basin_path / output_filename

    def _create_parallelized_weighted_forcing(self):
        """Create weighted forcing files with proper serial/parallel handling for HPC environments"""
        # Create output directories if they don't exist
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of forcing files (exclude non-temporal files like attributes, metadata, etc.)
        forcing_path = self.merged_forcing_path
        exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params']

        all_nc_files = list(forcing_path.glob('*.nc'))
        forcing_files = sorted([
            f for f in all_nc_files
            if not any(pattern in f.name.lower() for pattern in exclude_patterns)
        ])

        excluded_count = len(all_nc_files) - len(forcing_files)
        if excluded_count > 0:
            excluded_files = [f.name for f in all_nc_files if f not in forcing_files]
            self.logger.debug(f"Excluded {excluded_count} non-forcing files: {excluded_files}")
        
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return
        
        self.logger.debug(f"Found {len(forcing_files)} forcing files to process")
        
        # STEP 1: Create remapping weights ONCE (not per file)
        remap_file = self._create_remapping_weights_once(forcing_files[0], intersect_path)
        
        # STEP 2: Filter out already processed files
        remaining_files = self._filter_processed_files(forcing_files)
        
        if not remaining_files:
            self.logger.debug("All files have already been processed")
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
            
            self.logger.debug(f"Using parallel processing with {num_cpus} CPUs")
            success_count = self._process_files_parallel(remaining_files, num_cpus, remap_file)
        else:
            self.logger.debug("Using serial processing (no multiprocessing)")
            success_count = self._process_files_serial(remaining_files, remap_file)
        
        # Report final results
        already_processed = len(forcing_files) - len(remaining_files)
        self.logger.debug(f"Processing complete: {success_count} files processed successfully out of {len(remaining_files)}")
        self.logger.debug(f"Total files processed or skipped: {success_count + already_processed} out of {len(forcing_files)}")

    def _filter_processed_files(self, forcing_files):
        """Filter out already processed and valid files"""
        remaining_files = []
        already_processed = 0
        corrupted_files = 0

        for file in forcing_files:
            output_file = self._determine_output_filename(file)

            if output_file.exists():
                # Validate the existing file
                is_valid = self._validate_forcing_file(output_file)

                if is_valid:
                    self.logger.debug(f"Skipping already processed file: {file.name}")
                    already_processed += 1
                    continue
                else:
                    # File exists but is corrupted - delete and reprocess
                    self.logger.warning(f"Found corrupted output file {output_file}. Deleting and will reprocess.")
                    try:
                        output_file.unlink()
                        corrupted_files += 1
                    except Exception as e:
                        self.logger.warning(f"Error deleting corrupted file {output_file}: {str(e)}")

            remaining_files.append(file)

        self.logger.debug(f"Found {already_processed} already processed files")
        if corrupted_files > 0:
            self.logger.info(f"Deleted {corrupted_files} corrupted files that will be reprocessed")
        self.logger.debug(f"Found {len(remaining_files)} files that need processing")

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

        # Cache the processed shapefile path and HRU field for reuse during weight application
        # This avoids re-calling _ensure_shapefile_wgs84 which can cause segfaults with dissolved geometries
        self._cached_target_shp_wgs84 = target_shp_wgs84
        self._cached_hru_field = actual_hru_field
        self.logger.debug(f"Cached target shapefile: {target_shp_wgs84}, HRU field: {actual_hru_field}")

        # Define remap file path
        case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
        remap_file = intersect_path / f"{case_name}_{actual_hru_field}_remapping.csv"
        remap_nc = remap_file.with_suffix('.nc')
        
        # Check if remap files already exist
        if remap_file.exists() and remap_nc.exists():
            intersect_csv = intersect_path / f"{case_name}_intersected_shapefile.csv"
            intersect_shp = intersect_path / f"{case_name}_intersected_shapefile.shp"
            if intersect_csv.exists() or intersect_shp.exists():
                self.logger.debug(f"Remapping weights files (.csv and .nc) already exist. Skipping creation.")
                return remap_file
            self.logger.debug("Remapping weights found but intersected shapefile missing. Recreating.")
        elif remap_file.exists():
            self.logger.debug("Remapping CSV found but NetCDF weights missing (required for EASYMORE 2.0). Recreating.")
        
        self.logger.debug("Creating remapping weights...")

        
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
                            self.logger.debug("Shifted target shapefile longitudes to 0-360 for easymore.")
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

            # Set HDF5 file locking to FALSE to avoid potential crashes on some filesystems
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

            # Detect which SUMMA variables actually exist in the forcing file
            # Use netCDF4 directly for lightweight variable detection
            available_vars = []
            source_nc_resolution = None
            try:
                with nc4.Dataset(sample_forcing_file, 'r') as ncid:
                    all_summa_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                    available_vars = [v for v in all_summa_vars if v in ncid.variables]

                    if not available_vars:
                        raise ValueError(f"No SUMMA forcing variables found in {sample_forcing_file}. "
                                       f"Available variables: {list(ncid.variables.keys())}")

                    self.logger.info(f"Detected {len(available_vars)}/{len(all_summa_vars)} SUMMA variables in forcing file: {available_vars}")

                    # Store detected variables for use in weight application
                    self.detected_forcing_vars = available_vars

                    # Calculate grid resolution for small grids (needed by EASYMORE)
                    if var_lat not in ncid.variables:
                        raise KeyError(f"Latitude variable '{var_lat}' not found in {sample_forcing_file}")
                    if var_lon not in ncid.variables:
                        raise KeyError(f"Longitude variable '{var_lon}' not found in {sample_forcing_file}")
                        
                    lat_var = ncid.variables[var_lat]
                    lon_var = ncid.variables[var_lon]
                    
                    lat_vals = lat_var[:]
                    lon_vals = lon_var[:]

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
            except Exception as e:
                self.logger.error(f"Error detecting variables in {sample_forcing_file}: {e}")
                raise
            finally:
                # Force garbage collection to ensure file handles are closed
                gc.collect()

            esmr.source_nc = str(sample_forcing_file)
            esmr.var_names = available_vars
            esmr.var_lat = var_lat
            esmr.var_lon = var_lon
            esmr.var_time = 'time'

            # Set resolution for small grids
            if source_nc_resolution is not None:
                esmr.source_nc_resolution = source_nc_resolution
            
            # Directories - use temp_dir for BOTH to avoid polluting basin_averaged_data
            # during weight creation (even with only_create_remap_csv=True, EASYMORE 2.0
            # may still create output files)
            esmr.temp_dir = str(temp_dir) + '/'
            esmr.output_dir = str(temp_dir) + '/'
            
            # Output configuration
            esmr.remapped_dim_id = 'hru'
            esmr.remapped_var_id = 'hruId'
            esmr.format_list = ['f4']
            esmr.fill_value_list = ['-9999']
            
            # Critical: Tell easymore to ONLY create the remapping weights
            # Use attributes supported by both older and newer EASYMORE versions
            esmr.only_create_remap_csv = True
            if hasattr(esmr, 'only_create_remap_nc'):
                esmr.only_create_remap_nc = True
            
            esmr.save_csv = True
            esmr.sort_ID = False
            
            # Enable saving temporary shapefiles so SUMMA can find the intersected geometries/elevations
            esmr.save_temp_shp = True
            
            # Set numcpu to 1 to avoid internal multiprocessing in EASYMORE 2.0+

            # which can cause bus errors on macOS
            esmr.numcpu = 1
            
            # Create the weights
            self.logger.info("Running easymore to create remapping weights...")
            _run_easmore_with_suppressed_output(esmr, self.logger)
            
            # Move the remap file to the final location
            # With output_dir=temp_dir, EASYMORE should create files there
            case_remap_csv = temp_dir / f"{case_name}_remapping.csv"
            case_remap_nc = temp_dir / f"{case_name}_remapping.nc"
            case_attr_nc = temp_dir / f"{case_name}_attributes.nc"
            
            # If CSV is missing but NC exists (EASYMORE 2.0 with only_create_remap_nc=True),
            # convert NC to CSV as SYMFLUENCE expects the CSV as the primary weight file
            if not case_remap_csv.exists() and case_remap_nc.exists():
                self.logger.debug("EASYMORE 2.0: Converting NetCDF weights to CSV...")
                try:
                    with xr.open_dataset(case_remap_nc) as ds:
                        ds.to_dataframe().to_csv(case_remap_csv)
                except Exception as e:
                    self.logger.warning(f"Failed to convert NetCDF weights to CSV: {e}")

            candidate_paths = [case_remap_csv]

            if not any(path.exists() for path in candidate_paths):
                # Search for any CSV file that looks like a mapping file
                # EASYMORE 2.0.0 often creates files named 'Mapping_*.csv' for each variable
                # or with a hash in the name
                mapping_patterns = ["*remapping*.csv", "*_remapping.csv", "Mapping_*.csv"]
                fallback = []
                for pattern in mapping_patterns:
                    fallback.extend(list(temp_dir.glob(pattern)))

                if fallback:
                    self.logger.info(f"Using fallback mapping file: {fallback[0].name}")
                    candidate_paths.extend(fallback)

            remap_source = next((path for path in candidate_paths if path.exists()), None)
            if remap_source is not None:
                remap_source.replace(remap_file)
                self.logger.info(f"Remapping weights created: {remap_file}")
                
                # Also move the NetCDF versions if they exist (EASYMORE 2.0+)
                # These are required by EASYMORE 2.0 to skip re-calculating weights
                remap_nc = remap_file.with_suffix('.nc')
                attr_nc = remap_file.parent / f"{case_name}_attributes.nc"
                
                if case_remap_nc.exists():
                    shutil.move(str(case_remap_nc), str(remap_nc))
                    self.logger.debug(f"Moved NetCDF remapping file to {remap_nc}")
                
                if case_attr_nc.exists():
                    shutil.move(str(case_attr_nc), str(attr_nc))
                    self.logger.debug(f"Moved NetCDF attributes file to {attr_nc}")
            else:
                self.logger.error(f"Remapping file not found. Checked: {candidate_paths}")

                self.logger.error(f"Contents of {temp_dir}:")
                for p in temp_dir.glob("*"):
                    self.logger.error(f"  {p.name}")
                raise FileNotFoundError(f"Expected remapping file not created: {case_remap_csv}")

            
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

        with tqdm(total=len(files), desc="Remapping forcing files", unit="file") as pbar:
            for file in files:
                try:
                    success = self._apply_remapping_weights(file, remap_file)
                    if success:
                        success_count += 1
                    else:
                        self.logger.error(f"✗ Failed to process {file.name}")
                except Exception as e:
                    self.logger.error(f"✗ Error processing {file.name}: {str(e)}")

                pbar.update(1)

        self.logger.info(f"Serial processing complete: {success_count}/{len(files)} successful")
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

                # Use cached shapefile path and HRU field from weight creation phase
                # This avoids re-calling _ensure_shapefile_wgs84 which triggers dissolve
                # operations that can cause segfaults with certain geometries
                if self._cached_target_shp_wgs84 is not None and self._cached_hru_field is not None:
                    target_shp_wgs84 = self._cached_target_shp_wgs84
                    actual_hru_field = self._cached_hru_field
                    self.logger.debug(f"{worker_str}Using cached shapefile: {target_shp_wgs84}")
                else:
                    # Fallback to processing if cache not available (shouldn't happen in normal flow)
                    self.logger.warning(f"{worker_str}Shapefile cache not available, re-processing")
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

                # Detect variables in THIS specific file (not just from weight creation)
                # to ensure we're not trying to remap variables that don't exist
                available_vars = []
                try:
                    with nc4.Dataset(file, 'r') as ncid:
                        all_summa_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                        available_vars = [v for v in all_summa_vars if v in ncid.variables]

                        if not available_vars:
                            self.logger.error(
                                f"{worker_str}No SUMMA variables found in {file.name}. "
                                f"Available variables: {list(ncid.variables.keys())}"
                            )
                            return False

                        # Check if time dimension exists
                        if 'time' not in ncid.dimensions:
                            self.logger.error(f"{worker_str}Input file {file.name} has no time dimension!")
                            return False

                        esmr.var_names = available_vars
                        self.logger.debug(f"{worker_str}Detected {len(available_vars)} variables in {file.name}: {available_vars}")
                except Exception as e:
                    self.logger.error(f"{worker_str}Error opening {file.name} for variable detection: {e}")
                    return False
                finally:
                    gc.collect()

                esmr.var_lat = var_lat
                esmr.var_lon = var_lon
                esmr.var_time = 'time'
                
                # Directories - use temp_dir for output to avoid race conditions in parallel processing
                esmr.temp_dir = str(temp_dir) + '/'
                esmr.output_dir = str(temp_dir) + '/'  # Output to isolated temp directory

                # Output configuration
                esmr.remapped_dim_id = 'hru'
                esmr.remapped_var_id = 'hruId'
                esmr.format_list = ['f4']
                esmr.fill_value_list = ['-9999']

                # Critical: Point to pre-computed weights file
                esmr.remap_csv = str(remap_file)
                
                # EASYMORE 2.0 Support: Provide NetCDF version of weights if available
                # This is CRITICAL to avoid re-calculating weights which can cause bus errors
                remap_nc = remap_file.with_suffix('.nc')
                case_name = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}"
                attr_nc = remap_file.parent / f"{case_name}_attributes.nc"
                
                if remap_nc.exists():
                    esmr.remap_nc = str(remap_nc)
                    self.logger.debug(f"{worker_str}Using NetCDF remapping file: {remap_nc}")
                
                if attr_nc.exists():
                    esmr.attr_nc = str(attr_nc)
                    self.logger.debug(f"{worker_str}Using NetCDF attributes file: {attr_nc}")

                esmr.save_csv = False
                esmr.sort_ID = False
                
                # Disable saving temporary shapefiles
                esmr.save_temp_shp = False
                
                # Set numcpu to 1 to avoid internal multiprocessing in EASYMORE 2.0+

                esmr.numcpu = 1

                # Apply the remapping
                self.logger.debug(f"{worker_str}Applying remapping weights to {file.name}")
                self.logger.debug(f"{worker_str}EASYMORE configured to remap variables: {esmr.var_names}")

                success, stdout, stderr = _run_easmore_with_suppressed_output(esmr, self.logger)

                # Log any concerning patterns in output
                if 'no data' in stdout.lower() or 'no data' in stderr.lower():
                    self.logger.warning(f"{worker_str}EASYMORE reported 'no data' for {file.name}")
                if 'empty' in stdout.lower() or 'empty' in stderr.lower():
                    self.logger.warning(f"{worker_str}EASYMORE reported 'empty' for {file.name}")

                # Find the output file in temp directory (no race condition since each worker has its own temp dir)
                # Exclude metadata/auxiliary files that EASYMORE creates (not the actual forcing data)
                exclude_patterns = ['attributes', 'metadata', 'static', 'constants', 'params', 'remapping']
                all_temp_files = list(temp_dir.glob("*.nc"))
                temp_output_files = [
                    f for f in all_temp_files
                    if not any(pattern in f.name.lower() for pattern in exclude_patterns)
                ]

                if temp_output_files:
                    # Move the file from temp to final location with correct name
                    temp_output = temp_output_files[0]

                    # Validate file before moving to prevent corrupted files
                    is_valid = self._validate_forcing_file(temp_output, worker_str)

                    if not is_valid:
                        self.logger.error(
                            f"{worker_str}EASYMORE created invalid output for input {file.name}. "
                            f"Output file: {temp_output.name}. Check if EASYMORE is processing correctly."
                        )
                        # Log what files EASYMORE actually created in temp_dir
                        all_created = list(temp_dir.glob("*"))
                        self.logger.error(f"{worker_str}Files created by EASYMORE: {[f.name for f in all_created]}")
                        return False

                    # Ensure output directory exists
                    self.forcing_basin_path.mkdir(parents=True, exist_ok=True)

                    # Move to final location
                    shutil.move(str(temp_output), str(output_file))
                    self.logger.debug(f"{worker_str}Moved {temp_output.name} to {output_file.name}")
                elif output_file.exists():
                    # File already exists at final location (shouldn't happen, but handle gracefully)
                    self.logger.debug(f"{worker_str}Output file already exists: {output_file.name}")
                else:
                    self.logger.error(
                        f"{worker_str}EASYMORE created NO valid output files for input {file.name}. "
                        f"Files in temp dir: {[f.name for f in all_temp_files]}"
                    )
                    return False
                
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

    def _validate_forcing_file(self, file_path, worker_str=""):
        """
        Validate that a forcing file has proper structure (time dimension and forcing variables).

        Args:
            file_path: Path to the NetCDF file to validate
            worker_str: Optional worker identifier for logging

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            with xr.open_dataset(file_path) as ds:
                # Check 1: Must have time dimension
                if 'time' not in ds.dims:
                    self.logger.warning(f"{worker_str}File {file_path.name} missing time dimension")
                    return False

                # Check 2: Time dimension should have reasonable size (at least 1 timestep)
                time_size = ds.sizes.get('time', 0)  # Use sizes instead of dims
                if time_size < 1:
                    self.logger.warning(f"{worker_str}File {file_path.name} has empty time dimension")
                    return False

                # Check 3: Should have at least one forcing variable
                expected_vars = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
                has_forcing_var = any(var in ds.data_vars for var in expected_vars)

                if not has_forcing_var:
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} missing forcing variables. "
                        f"Has: {list(ds.data_vars)}"
                    )
                    return False

                # Check 4: File should be larger than just metadata (>= 100KB for real data)
                file_size = file_path.stat().st_size
                if file_size < 100000:  # 100KB
                    self.logger.warning(
                        f"{worker_str}File {file_path.name} suspiciously small ({file_size} bytes). "
                        f"Likely contains only metadata."
                    )
                    return False

                self.logger.debug(f"{worker_str}File {file_path.name} validated successfully")
                return True

        except Exception as e:
            self.logger.warning(f"{worker_str}Error validating file {file_path.name}: {str(e)}")
            return False

    def _process_files_parallel(self, files, num_cpus, remap_file):
        """Process files in parallel mode applying pre-computed weights"""
        self.logger.debug(f"Processing {len(files)} in parallel with {num_cpus} CPUs")
        
        batch_size = min(10, len(files))
        total_batches = (len(files) + batch_size - 1) // batch_size
        
        self.logger.debug(f"Processing {total_batches} batches of up to {batch_size} files each")
        
        success_count = 0

        with tqdm(total=len(files), desc="Remapping forcing files", unit="file") as pbar:
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(files))
                batch_files = files[start_idx:end_idx]
                
                try:
                    with mp.Pool(processes=num_cpus) as pool:
                        worker_args = [(file, remap_file, i % num_cpus) for i, file in enumerate(batch_files)]
                        results = pool.starmap(self._apply_remapping_weights_worker, worker_args)
                    
                    batch_success = sum(1 for r in results if r)
                    success_count += batch_success
                    pbar.update(len(batch_files))
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_num+1}: {str(e)}")
                    pbar.update(len(batch_files))
                
                import gc
                gc.collect()

        self.logger.debug(f"Parallel processing complete: {success_count}/{len(files)} successful")
        return success_count

    def _ensure_unique_hru_ids(self, shapefile_path, hru_id_field):
        """
        Ensure HRU IDs are unique in the shapefile. For lumped catchments with duplicate HRU_IDs,
        dissolve features by HRU_ID to create a single polygon per HRU. Otherwise, create new unique IDs.

        Args:
            shapefile_path: Path to the shapefile
            hru_id_field: Name of the HRU ID field

        Returns:
            tuple: (updated_shapefile_path, actual_hru_id_field_used)
        """
        try:
            shapefile_path = Path(shapefile_path)

            # Check if a dissolved or unique_ids version already exists and is valid
            dissolved_path = shapefile_path.parent / f"{shapefile_path.stem}_dissolved.shp"
            unique_ids_path = shapefile_path.parent / f"{shapefile_path.stem}_unique_ids.shp"

            for existing_path in [dissolved_path, unique_ids_path]:
                if existing_path.exists():
                    try:
                        existing_gdf = gpd.read_file(existing_path)
                        if hru_id_field in existing_gdf.columns:
                            if existing_gdf[hru_id_field].nunique() == len(existing_gdf):
                                self.logger.debug(
                                    f"Using existing processed shapefile: {existing_path.name}"
                                )
                                return existing_path, hru_id_field
                    except Exception as e:
                        self.logger.debug(f"Could not use existing {existing_path.name}: {e}")

            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            self.logger.debug(f"Checking HRU ID uniqueness in {shapefile_path.name}")
            self.logger.debug(f"Available fields: {list(gdf.columns)}")

            # Check if the HRU ID field exists
            if hru_id_field not in gdf.columns:
                self.logger.error(f"HRU ID field '{hru_id_field}' not found in shapefile.")
                raise ValueError(f"HRU ID field '{hru_id_field}' not found in shapefile")

            # Check for uniqueness
            original_count = len(gdf)
            unique_count = gdf[hru_id_field].nunique()

            self.logger.debug(f"Shapefile has {original_count} rows, {unique_count} unique {hru_id_field} values")

            if unique_count == original_count:
                self.logger.debug(f"All {hru_id_field} values are unique")
                return shapefile_path, hru_id_field

            # Handle duplicate IDs
            self.logger.info(f"Found {original_count - unique_count} duplicate {hru_id_field} values")

            # For lumped catchments (small number of unique HRUs), dissolve features by HRU_ID
            # This aggregates multiple spatial features with the same HRU_ID into a single polygon
            if unique_count <= 10:  # Threshold for lumped/semi-distributed models
                self.logger.info(f"Detected lumped/semi-distributed catchment ({unique_count} unique HRUs)")
                self.logger.info(f"Dissolving {original_count} features into {unique_count} HRUs by {hru_id_field}")

                # Aggregate numeric fields (mean), preserve first value for others
                # Note: Don't include the grouping field (hru_id_field) in agg_dict
                # as it will become the index and reset_index() will handle it
                agg_dict = {}
                for col in gdf.columns:
                    if col == 'geometry':
                        continue
                    elif col == hru_id_field:
                        continue  # Skip the grouping field - it becomes the index
                    elif col in ['GRU_ID', 'gru_to_seg']:
                        agg_dict[col] = 'first'  # Keep first value for ID fields
                    elif gdf[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        agg_dict[col] = 'mean'  # Average numeric fields
                    else:
                        agg_dict[col] = 'first'  # Keep first value for text fields

                # Dissolve features by HRU_ID
                gdf_dissolved = gdf.dissolve(by=hru_id_field, aggfunc=agg_dict)
                gdf_dissolved = gdf_dissolved.reset_index()

                self.logger.info(f"Dissolved into {len(gdf_dissolved)} features")

                # Validate and repair geometries after dissolve to prevent bus errors
                # Dissolve operations can sometimes create invalid geometries
                gdf_dissolved = self._validate_and_repair_geometries(gdf_dissolved)

                # Create output path for the dissolved shapefile
                output_path = shapefile_path.parent / f"{shapefile_path.stem}_dissolved.shp"

                # Save the dissolved shapefile
                gdf_dissolved.to_file(output_path)
                self.logger.info(f"Dissolved shapefile saved to: {output_path}")

                # Verify the result
                verify_gdf = gpd.read_file(output_path)
                if verify_gdf[hru_id_field].nunique() == len(verify_gdf):
                    self.logger.info(f"Verification successful: {len(verify_gdf)} unique HRUs")
                    return output_path, hru_id_field
                else:
                    self.logger.error("Verification failed: Dissolve did not create unique HRU IDs")
                    raise ValueError("Could not dissolve features by HRU_ID")

            # For distributed models with many HRUs, create unique sequential IDs
            else:
                self.logger.info(f"Detected distributed model ({unique_count} unique HRUs, {original_count} features)")
                self.logger.info("Creating new unique sequential IDs for each feature")

                # Create new unique ID field with shorter name (shapefile 10-char limit)
                new_hru_field = "hru_id_new"

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
            
            self.logger.debug(f"Checking CRS for {shapefile_path.name}: {current_crs}")
            
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
                self.logger.debug(f"Shapefile {shapefile_path.name} already in WGS84")
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
                    _run_easmore_with_suppressed_output(esmr, self.logger)
                    
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
                    _run_easmore_with_suppressed_output(esmr, self.logger)
                
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
                        _run_easmore_with_suppressed_output(esmr, self.logger)
                        
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
                    _run_easmore_with_suppressed_output(esmr, self.logger)
                    
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
