"""
ForcingResampler - Orchestrator for forcing data remapping.

This module coordinates the forcing remapping workflow:
1. Merge raw forcing files (via dataset handlers)
2. Create forcing grid shapefile
3. Generate remapping weights (once)
4. Apply weights to all forcing files

This is a refactored version that delegates to focused modules:
- ShapefileManager: CRS alignment and HRU ID management
- ElevationCalculator: DEM-based elevation statistics
- RemappingWeightGenerator: EASYMORE weight creation
- RemappingWeightApplier: Weight application to forcing files
- BatchProcessor: Serial/parallel processing orchestration
"""

from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp
import sys

from symfluence.utils.common.path_resolver import PathResolverMixin
from symfluence.utils.data.path_manager import PathManager

# Import dataset handlers
try:
    from symfluence.utils.data.preprocessing.dataset_handlers import DatasetRegistry
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_handlers import DatasetRegistry

# Import refactored components
from .shapefile_manager import ShapefileManager
from .elevation_calculator import ElevationCalculator
from .remapping_weights import (
    RemappingWeightGenerator,
    RemappingWeightApplier,
    BatchProcessor
)


class ForcingResampler(PathResolverMixin):
    """
    Orchestrates forcing data remapping from gridded to catchment-averaged format.

    This class coordinates the workflow but delegates actual work to specialized
    components for better testability and maintainability.
    """

    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize ForcingResampler.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Initialize path manager
        self.paths = PathManager(config)
        self.domain_name = self.paths.domain_name
        self.project_dir = self.paths.project_dir

        # Derived paths
        self.shapefile_path = self.paths.forcing_shapefile_dir
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.catchment_path = self.paths.resolve('CATCHMENT_PATH', 'shapefiles/catchment')

        # DEM path
        dem_name = config.get('DEM_NAME', 'default')
        if dem_name == "default":
            dem_name = f"domain_{self.domain_name}_elv.tif"
        self.dem_path = self.paths.resolve('DEM_PATH', f"attributes/elevation/dem/{dem_name}")

        # Catchment shapefile name
        self.catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')
        if self.catchment_name == 'default':
            discretization = str(config.get('DOMAIN_DISCRETIZATION', '')).replace(',', '_')
            self.catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"

        # Forcing configuration
        self.forcing_dataset = config.get('FORCING_DATASET', 'era5').lower()
        self.merged_forcing_path = self.paths.resolve('FORCING_PATH', 'forcing/raw_data')

        # Initialize dataset handler
        self._init_dataset_handler()

        # Initialize helper components
        self.shapefile_manager = ShapefileManager(config, logger)
        self.elevation_calculator = ElevationCalculator(logger)

        # Merge forcings if required
        if self.dataset_handler.needs_merging():
            self.logger.info(f"{self.forcing_dataset.upper()} requires merging of raw files")
            self.merge_forcings()
            self.merged_forcing_path = self.paths.resolve('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)

    def _init_dataset_handler(self) -> None:
        """Initialize the dataset-specific handler."""
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

    def run_resampling(self) -> None:
        """
        Execute the complete forcing resampling workflow.

        Steps:
        1. Create forcing grid shapefile (if needed)
        2. Generate remapping weights (once)
        3. Apply weights to all forcing files
        """
        self.logger.info("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.info("Forcing data resampling process completed")

    def merge_forcings(self) -> None:
        """
        Merge forcing data files using dataset-specific handler.

        Delegates to the appropriate dataset handler which contains
        all dataset-specific logic for variable mapping, unit conversions, and merging.
        """
        start_year = int(self.config.get('EXPERIMENT_TIME_START').split('-')[0])
        end_year = int(self.config.get('EXPERIMENT_TIME_END').split('-')[0])

        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'

        self.dataset_handler.merge_forcings(
            raw_forcing_path=raw_forcing_path,
            merged_forcing_path=merged_forcing_path,
            start_year=start_year,
            end_year=end_year
        )

    def create_shapefile(self) -> Optional[Path]:
        """
        Create forcing shapefile using dataset-specific handler.

        Checks for existing valid shapefile before creating a new one.

        Returns:
            Path to the created or existing shapefile
        """
        import geopandas as gpd

        self.logger.info(f"Creating {self.forcing_dataset.upper()} shapefile")

        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config.get('FORCING_DATASET')}.shp"

        # Check for existing valid shapefile
        if output_shapefile.exists():
            if self._is_shapefile_valid(output_shapefile):
                self.logger.info(f"Forcing shapefile already exists: {output_shapefile}. Skipping.")
                return output_shapefile
            self.logger.info("Existing forcing shapefile invalid. Recreating.")

        # Delegate shapefile creation to dataset handler
        return self.dataset_handler.create_shapefile(
            shapefile_path=self.shapefile_path,
            merged_forcing_path=self.merged_forcing_path,
            dem_path=self.dem_path,
            elevation_calculator=self.elevation_calculator.calculate_mean_elevation
        )

    def _is_shapefile_valid(self, shapefile_path: Path) -> bool:
        """Check if an existing shapefile is valid for the current configuration."""
        import geopandas as gpd

        try:
            gdf = gpd.read_file(shapefile_path)
            expected_columns = [
                self.config.get('FORCING_SHAPE_LAT_NAME'),
                self.config.get('FORCING_SHAPE_LON_NAME'),
                'ID', 'elev_m'
            ]

            if not all(col in gdf.columns for col in expected_columns) or len(gdf) == 0:
                return False

            # Check bbox coverage
            bbox_str = self.config.get("BOUNDING_BOX_COORDS")
            if isinstance(bbox_str, str) and "/" in bbox_str:
                lat_max, lon_min, lat_min, lon_max = [float(v) for v in bbox_str.split("/")]
                lat_min, lat_max = sorted([lat_min, lat_max])
                lon_min, lon_max = sorted([lon_min, lon_max])
                minx, miny, maxx, maxy = gdf.total_bounds
                tol = 1e-6
                if (lon_min < minx - tol or lon_max > maxx + tol or
                        lat_min < miny - tol or lat_max > maxy + tol):
                    self.logger.info("Existing shapefile bounds don't cover current bbox.")
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Error checking existing shapefile: {str(e)}")
            return False

    def remap_forcing(self) -> None:
        """
        Remap forcing data from grid to catchments.

        Orchestrates weight generation (once) and application (per file).
        """
        self.logger.info("Starting forcing remapping process")

        # Setup directories
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        intersect_path.mkdir(parents=True, exist_ok=True)

        # Get forcing files
        forcing_files = sorted(self.merged_forcing_path.glob('*.nc'))
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return

        self.logger.info(f"Found {len(forcing_files)} forcing files to process")

        # Prepare shapefiles in WGS84
        source_shp = self.shapefile_path / f"forcing_{self.config.get('FORCING_DATASET')}.shp"
        target_shp = self.catchment_path / self.catchment_name

        source_wgs84 = self.shapefile_manager.ensure_wgs84(source_shp)
        target_wgs84, actual_hru_field = self.shapefile_manager.ensure_wgs84(
            target_shp,
            ensure_unique_ids=True,
            hru_id_field=self.config.get('CATCHMENT_SHP_HRUID')
        )

        # Generate remapping weights (once)
        weight_generator = RemappingWeightGenerator(
            self.config, self.logger, self.project_dir, self.shapefile_manager
        )
        remap_file, detected_vars = weight_generator.create_weights(
            source_shapefile=source_wgs84,
            target_shapefile=target_wgs84,
            sample_forcing_file=forcing_files[0],
            output_dir=intersect_path,
            dataset_handler=self.dataset_handler,
            hru_id_field=actual_hru_field
        )

        # Setup weight applier
        applier = RemappingWeightApplier(
            self.config, self.logger, self.project_dir,
            self.forcing_basin_path, self.dataset_handler
        )
        applier.set_detected_variables(detected_vars)

        # Setup batch processor
        processor = BatchProcessor(applier, self.logger)

        # Filter already processed files
        remaining_files = processor.filter_unprocessed(forcing_files)
        if not remaining_files:
            self.logger.info("All files have already been processed")
            return

        # Determine processing mode
        requested_cpus = int(self.config.get('MPI_PROCESSES', 1))
        max_cpus = mp.cpu_count()
        use_parallel = requested_cpus > 1 and max_cpus > 1

        if use_parallel:
            num_cpus = min(requested_cpus, max_cpus, 20, len(remaining_files))
            self.logger.info(f"Using parallel processing with {num_cpus} CPUs")
            success_count = processor.process_parallel(remaining_files, remap_file, num_cpus)
        else:
            self.logger.info("Using serial processing")
            success_count = processor.process_serial(remaining_files, remap_file)

        # Report results
        already_processed = len(forcing_files) - len(remaining_files)
        self.logger.info(
            f"Processing complete: {success_count} files processed "
            f"({success_count + already_processed}/{len(forcing_files)} total)"
        )

        self.logger.info("Forcing remapping process completed")
