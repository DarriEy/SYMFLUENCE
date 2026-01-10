"""
Forcing Resampler

Orchestrates forcing data remapping from source grids to model catchments.
Delegates to specialized components for GIS operations, file processing, and weight management.
"""

import logging
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import geopandas as gpd

from symfluence.core.path_resolver import PathResolverMixin

from .resampling import (
    ElevationCalculator,
    FileProcessor,
    PointScaleForcingExtractor,
    RemappingWeightApplier,
    RemappingWeightGenerator,
    ShapefileProcessor,
)

try:
    from symfluence.data.preprocessing.dataset_handlers import DatasetRegistry
except ImportError:
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

warnings.filterwarnings('ignore', category=DeprecationWarning, module='easymore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='easymore')


class ForcingResampler(PathResolverMixin):
    """
    Orchestrates forcing data remapping using EASYMORE.

    Delegates specialized operations to:
    - RemappingWeightGenerator: Creates intersection weights (expensive, one-time)
    - RemappingWeightApplier: Applies weights to forcing files (fast, per-file)
    - FileProcessor: Handles parallel/serial file processing
    - PointScaleForcingExtractor: Simplified extraction for small grids
    - ElevationCalculator: DEM-based elevation statistics
    - ShapefileProcessor: CRS conversion and HRU ID handling
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'

        dem_name = self._get_config_value(lambda: self.config.paths.dem_name)
        if dem_name == "default":
            dem_name = f"domain_{self._get_config_value(lambda: self.config.domain.name)}_elv.tif"

        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self._get_config_value(lambda: self.config.paths.catchment_shp_name)
        if self.catchment_name == 'default':
            self.catchment_name = f"{self._get_config_value(lambda: self.config.domain.name)}_HRUs_{str(self._get_config_value(lambda: self.config.domain.discretization)).replace(',','_')}.shp"
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
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
            self.merge_forcings()

        # Lazy-initialized components
        self._elevation_calculator = None
        self._file_processor = None
        self._weight_generator = None
        self._weight_applier = None
        self._point_scale_extractor = None
        self._shapefile_processor = None

    # Lazy initialization properties
    @property
    def elevation_calculator(self) -> ElevationCalculator:
        if self._elevation_calculator is None:
            self._elevation_calculator = ElevationCalculator(self.logger)
        return self._elevation_calculator

    @property
    def file_processor(self) -> FileProcessor:
        if self._file_processor is None:
            self._file_processor = FileProcessor(
                self.config,
                self.forcing_basin_path,
                self.logger
            )
        return self._file_processor

    @property
    def weight_generator(self) -> RemappingWeightGenerator:
        if self._weight_generator is None:
            self._weight_generator = RemappingWeightGenerator(
                self.config,
                self.project_dir,
                self.dataset_handler,
                self.logger
            )
        return self._weight_generator

    @property
    def weight_applier(self) -> RemappingWeightApplier:
        if self._weight_applier is None:
            self._weight_applier = RemappingWeightApplier(
                self.config,
                self.project_dir,
                self.forcing_basin_path,
                self.dataset_handler,
                self.logger
            )
        return self._weight_applier

    @property
    def point_scale_extractor(self) -> PointScaleForcingExtractor:
        if self._point_scale_extractor is None:
            self._point_scale_extractor = PointScaleForcingExtractor(
                self.config,
                self.project_dir,
                self.dataset_handler,
                self.logger
            )
        return self._point_scale_extractor

    @property
    def shapefile_processor(self) -> ShapefileProcessor:
        if self._shapefile_processor is None:
            self._shapefile_processor = ShapefileProcessor(self.config, self.logger)
        return self._shapefile_processor

    def run_resampling(self):
        """Run the complete forcing resampling process."""
        self.logger.debug("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.debug("Forcing data resampling process completed")

    def merge_forcings(self):
        """Merge forcing data files using dataset-specific handler."""
        start_year = int(self._get_config_value(lambda: self.config.domain.time_start).split('-')[0])
        end_year = int(self._get_config_value(lambda: self.config.domain.time_end).split('-')[0])

        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'

        self.dataset_handler.merge_forcings(
            raw_forcing_path=raw_forcing_path,
            merged_forcing_path=merged_forcing_path,
            start_year=start_year,
            end_year=end_year
        )

    def create_shapefile(self):
        """Create forcing shapefile using dataset-specific handler."""
        self.logger.debug(f"Creating {self.forcing_dataset.upper()} shapefile")

        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset)}.shp"

        if output_shapefile.exists():
            if self._validate_existing_shapefile(output_shapefile):
                return output_shapefile

        return self.dataset_handler.create_shapefile(
            shapefile_path=self.shapefile_path,
            merged_forcing_path=self.merged_forcing_path,
            dem_path=self.dem_path,
            elevation_calculator=self.elevation_calculator.calculate
        )

    def _validate_existing_shapefile(self, output_shapefile: Path) -> bool:
        """Check if existing shapefile is valid and covers current bbox."""
        try:
            gdf = gpd.read_file(output_shapefile)
            expected_columns = [
                self._get_config_value(lambda: self.config.forcing.shape_lat_name),
                self._get_config_value(lambda: self.config.forcing.shape_lon_name),
                'ID', 'elev_m'
            ]

            if not all(col in gdf.columns for col in expected_columns) or len(gdf) == 0:
                self.logger.debug("Existing forcing shapefile missing expected columns. Recreating.")
                return False

            bbox_str = self._get_config_value(lambda: self.config.domain.bounding_box_coords)
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
                        return False
                except Exception as e:
                    self.logger.warning(f"Error checking bbox vs shapefile bounds: {e}. Recreating.")
                    return False

            self.logger.debug(f"Forcing shapefile already exists. Skipping creation.")
            return True

        except Exception as e:
            self.logger.warning(f"Error checking existing forcing shapefile: {str(e)}. Recreating.")
            return False

    def remap_forcing(self):
        """Remap forcing data to catchment HRUs."""
        self.logger.debug("Starting forcing remapping process")

        # Check for point-scale bypass conditions
        if (self._get_config_value(lambda: self.config.domain.definition_method, default='') or '').lower() == 'point':
            self.logger.debug("Point-scale domain detected. Using simplified extraction.")
            self._process_point_scale_forcing()
        elif self.point_scale_extractor.should_use_point_scale(self.merged_forcing_path):
            self.logger.info("Tiny forcing grid detected. Using simplified extraction.")
            self._process_point_scale_forcing()
        else:
            self._create_parallelized_weighted_forcing()

        self.logger.debug("Forcing remapping process completed")

    def _process_point_scale_forcing(self):
        """Process forcing files using point-scale extraction."""
        forcing_files = self.file_processor.get_forcing_files(self.merged_forcing_path)
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        self.point_scale_extractor.process(
            forcing_files=forcing_files,
            output_dir=self.forcing_basin_path,
            catchment_path=self.catchment_path,
            catchment_name=self.catchment_name,
            output_filename_func=self.file_processor.determine_output_filename
        )

    def _create_parallelized_weighted_forcing(self):
        """Create weighted forcing files with parallel/serial processing."""
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Get forcing files
        forcing_files = self.file_processor.get_forcing_files(self.merged_forcing_path)
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        self.logger.debug(f"Found {len(forcing_files)} forcing files to process")

        # STEP 1: Create remapping weights once
        source_shp_path = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self._get_config_value(lambda: self.config.forcing.dataset)}.shp"
        target_shp_path = self.catchment_path / self.catchment_name

        remap_file = self.weight_generator.create_weights(
            sample_forcing_file=forcing_files[0],
            intersect_path=intersect_path,
            source_shp_path=source_shp_path,
            target_shp_path=target_shp_path
        )

        # Transfer cached shapefile info to weight applier
        self.weight_applier.set_shapefile_cache(
            self.weight_generator.cached_target_shp_wgs84,
            self.weight_generator.cached_hru_field
        )

        # STEP 2: Filter already processed files
        remaining_files = self.file_processor.filter_processed_files(forcing_files)
        if not remaining_files:
            self.logger.debug("All files have already been processed")
            return

        # STEP 3: Apply remapping weights
        requested_cpus = int(self._get_config_value(lambda: self.config.system.mpi_processes, default=1))
        max_available_cpus = mp.cpu_count()
        use_parallel = requested_cpus > 1 and max_available_cpus > 1

        if use_parallel:
            num_cpus = min(requested_cpus, max_available_cpus, 20, len(remaining_files))
            self.logger.debug(f"Using parallel processing with {num_cpus} CPUs")
            success_count = self._process_files_parallel(remaining_files, num_cpus, remap_file)
        else:
            self.logger.debug("Using serial processing")
            success_count = self._process_files_serial(remaining_files, remap_file)

        already_processed = len(forcing_files) - len(remaining_files)
        self.logger.debug(
            f"Processing complete: {success_count} files processed successfully "
            f"out of {len(remaining_files)}"
        )
        self.logger.debug(
            f"Total files processed or skipped: {success_count + already_processed} "
            f"out of {len(forcing_files)}"
        )

    def _process_files_serial(self, files, remap_file):
        """Process files in serial mode."""
        def process_func(file):
            output_file = self.file_processor.determine_output_filename(file)
            return self.weight_applier.apply_weights(file, remap_file, output_file)

        return self.file_processor.process_serial(files, process_func)

    def _process_files_parallel(self, files, num_cpus, remap_file):
        """Process files in parallel mode."""
        def process_func(file, worker_id):
            output_file = self.file_processor.determine_output_filename(file)
            return self.weight_applier.apply_weights(file, remap_file, output_file, worker_id)

        return self.file_processor.process_parallel(files, num_cpus, process_func)
