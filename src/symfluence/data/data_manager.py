"""
Data management facade for SYMFLUENCE hydrological modeling.

Coordinates all data acquisition and preprocessing operations through
specialized services for forcing data, observations, and attribute processing.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.core.exceptions import (
    DataAcquisitionError,
    symfluence_error_handler
)
from symfluence.data.acquisition.acquisition_service import AcquisitionService
from symfluence.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.data.observation.registry import ObservationRegistry
from symfluence.data.preprocessing.em_earth_integrator import EMEarthIntegrator
from symfluence.data.preprocessing.forcing_resampler import ForcingResampler
from symfluence.data.preprocessing.geospatial_statistics import GeospatialStatistics
from symfluence.data.utils.variable_utils import VariableHandler

if TYPE_CHECKING:
    pass


class DataManager(BaseManager):
    """
    Orchestrates all data acquisition and preprocessing for hydrological modeling.

    This manager acts as a facade that coordinates specialized data services for
    forcing acquisition, attribute extraction, observation processing, and spatial
    preprocessing. It delegates to specialized services while providing a unified
    interface for the SYMFLUENCE workflow orchestrator.

    Architecture:
        Facade Pattern - Delegates to specialized services:
        - **AcquisitionService**: Cloud/HPC data downloads (forcings, attributes)
        - **ObservedDataProcessor**: Streamflow and validation data processing
        - **ObservationRegistry**: Extensible observation handlers (GRACE, MODIS, SNOTEL)
        - **GeospatialStatistics**: Zonal statistics (elevation, soil, land cover)
        - **ForcingResampler**: Spatial averaging and temporal resampling
        - **EMEarthIntegrator**: EM-Earth supplementary forcing integration
        - **VariableHandler**: Dataset-specific variable mapping

    Data Acquisition Workflow:
        1. **acquire_attributes()**: Download DEM, soil class, land cover
        2. **acquire_forcings()**: Download meteorological forcing (ERA5/RDRS/CARRA/etc.)
        3. **acquire_observations()**: Download streamflow, snow, GRACE data
        4. **acquire_em_earth_forcings()**: Download EM-Earth supplementary data (optional)

    Preprocessing Workflow:
        5. **process_observed_data()**: Process and standardize all observations
        6. **run_model_agnostic_preprocessing()**: Spatial averaging and resampling

    Supported Forcing Datasets:
        Cloud-based (via AcquisitionService):
            - ERA5: Global reanalysis, 0.25°, 1979-present
            - RDRS: Canadian reanalysis, 10 km, 1980-2018
            - CARRA: Arctic reanalysis, 2.5 km, 1990-present
            - CERRA: European reanalysis, 5.5 km, 1984-present
            - AORC: CONUS, 1 km, 1979-present (AWS S3)
            - CONUS404: CONUS WRF, 4 km, 1979-present (HyTEST)
            - HRRR: CONUS operational, 3 km, 2014-present (AWS S3)

        HPC-based (via datatool/MAF):
            - ERA5, RDRS, CASR (from local HPC archives)

    Supported Observation Types:
        Streamflow:
            - USGS_STREAMFLOW: US Geological Survey
            - WSC_STREAMFLOW: Water Survey of Canada
            - SMHI_STREAMFLOW: Swedish Meteorological Institute
            - LAMAH_ICE_STREAMFLOW: Iceland basins
            - CARAVANS: Global multi-source dataset
            - VI: Iceland Meteorological Office

        Snow/SWE:
            - SNOTEL: Snow telemetry network (USA)
            - MODIS_SNOW: Satellite snow cover/SWE

        Other Variables:
            - GRACE: Terrestrial water storage anomalies
            - USGS_GW: Groundwater levels
            - FLUXNET: Eddy covariance flux towers
            - ISMN: International Soil Moisture Network

    Geospatial Attributes:
        Acquired via gistool or cloud sources:
        - Elevation: DEM (mean, min, max, slope)
        - Soil classification: Categorical classes (SoilGrids, national datasets)
        - Land cover: MODIS, NLCD, CCI-LC, etc.
        - Fractional coverages and majority classes computed via zonal statistics

    Preprocessing Operations:
        Geospatial Statistics (via GeospatialStatistics):
            - Zonal statistics: Raster-to-catchment aggregation
            - Elevation statistics per HRU
            - Soil/land class fractions per HRU
            - Shapefile enrichment with attributes

        Forcing Resampling (via ForcingResampler):
            - Spatial averaging: Grid-to-catchment using EASYMORE
            - Temporal resampling: Hourly/3-hourly/daily alignment
            - Unit conversions: Dataset-specific transformations
            - NetCDF output: CF-compliant forcing files

        EM-Earth Integration (via EMEarthIntegrator):
            - Gap-filling: Fill missing forcing periods
            - Variable supplementation: Add missing variables
            - Blending: Weight-based combination with primary dataset

    Configuration Dependencies:
        Required:
            - SYMFLUENCE_DATA_DIR: Base data directory
            - DOMAIN_NAME: Basin identifier
            - FORCING_DATASET: Primary forcing source (ERA5/RDRS/etc.)
            - EXPERIMENT_TIME_START: Simulation start date
            - EXPERIMENT_TIME_END: Simulation end date

        Observations:
            - STREAMFLOW_DATA_PROVIDER: Streamflow source (USGS/WSC/etc.)
            - DOWNLOAD_SNOTEL: Enable SNOTEL acquisition (True/False)
            - DOWNLOAD_MODIS_SNOW: Enable MODIS snow (True/False)
            - ADDITIONAL_OBSERVATIONS: List of observation types

        Preprocessing:
            - FORCING_TIME_STEP_SIZE: Model timestep (seconds)
            - SUPPLEMENT_FORCING: Enable EM-Earth integration (True/False)

    Service Initialization:
        Services initialized in _initialize_services():
        - AcquisitionService: Handles all downloads
        - EMEarthIntegrator: EM-Earth processing
        - VariableHandler: Variable mapping (ERA5↔SUMMA)

        Lazy initialization for:
        - ObservedDataProcessor: Created when needed
        - GeospatialStatistics: Created during preprocessing
        - ForcingResampler: Created during preprocessing

    Registry-Based Extensibility:
        Observation handlers registered via ObservationRegistry:
        - Decoupled design: Add new observation types without modifying core
        - Handler interface: acquire() → process() → validate()
        - Automatic discovery: Registry-based lookup

    Error Handling:
        Uses symfluence_error_handler for consistent error management:
        - Raises DataAcquisitionError on failures
        - Logs detailed error messages
        - Provides context for debugging
        - Continues workflow where possible (graceful degradation)

    Visualization Integration:
        If reporting_manager available:
        - Data distribution plots for observations
        - Spatial coverage maps for forcings/attributes
        - Visualization after each preprocessing stage

    Example Workflow:
        >>> from symfluence.data.data_manager import DataManager
        >>> config = load_config('config.yaml')
        >>> logger = setup_logger()
        >>> reporting_mgr = ReportingManager(config, logger)
        >>>
        >>> # Initialize manager
        >>> data_mgr = DataManager(config, logger, reporting_mgr)
        >>>
        >>> # Acquisition phase
        >>> data_mgr.acquire_attributes()      # Download DEM, soil, land cover
        >>> data_mgr.acquire_forcings()        # Download ERA5 forcing
        >>> data_mgr.acquire_observations()    # Download USGS streamflow
        >>>
        >>> # Preprocessing phase
        >>> data_mgr.process_observed_data()   # Process streamflow to CSV
        >>> data_mgr.run_model_agnostic_preprocessing()  # Spatial averaging
        >>>
        >>> # Validation
        >>> status = data_mgr.get_data_status()
        >>> print(status)
        # {'forcings': True, 'observations': True, 'attributes': True}

    Output Structure:
        Data organized in project directory:
        project_dir/
        ├── forcing/
        │   ├── raw_data/              # Downloaded forcing NetCDF
        │   ├── basin_averaged_data/   # Spatially averaged forcing
        │   └── merged_data/           # Model-ready forcing
        ├── observations/
        │   ├── streamflow/
        │   │   ├── raw/               # Downloaded observations
        │   │   └── preprocessed/      # Processed CSV
        │   ├── snow/                  # SNOTEL, MODIS snow
        │   └── grace/                 # TWS anomalies
        ├── attributes/
        │   ├── elevation/             # DEM GeoTIFFs
        │   ├── soilclass/             # Soil classification
        │   └── landclass/             # Land cover
        └── shapefiles/
            ├── catchment/             # HRU polygons
            └── catchment_intersection/ # Enriched with attributes

    Performance:
        - Parallel downloads: AcquisitionService uses concurrent downloads
        - Lazy loading: Services initialized only when needed
        - Caching: Avoids redundant downloads (checks for existing files)
        - Memory efficient: Streaming for large NetCDF files

    Notes:
        - Facade pattern simplifies complex data operations
        - Delegates to specialized services for separation of concerns
        - Extensible via ObservationRegistry for new data types
        - Reporting integration provides automatic visualization
        - Graceful degradation: Non-critical observations fail safely

    See Also:
        - data.acquisition.acquisition_service.AcquisitionService: Download coordinator
        - data.acquisition.observed_processor.ObservedDataProcessor: Obs processing
        - data.observation.registry.ObservationRegistry: Extensible obs handlers
        - data.preprocessing.geospatial_statistics.GeospatialStatistics: Zonal stats
        - data.preprocessing.forcing_resampler.ForcingResampler: Spatial averaging
    """

    def _initialize_services(self) -> None:
        """Initialize data management services."""
        self.acquisition_service = self._get_service(
            AcquisitionService,
            self.config,
            self.logger,
            self.reporting_manager
        )
        self.em_earth_integrator = self._get_service(
            EMEarthIntegrator,
            self.config,
            self.logger
        )
        self.variable_handler = self._get_service(
            VariableHandler,
            self.config_dict,
            self.logger,
            'ERA5',
            'SUMMA'
        )

    def acquire_attributes(self):
        """
        Acquire geospatial attributes (DEM, soil classes, land cover) for the domain.

        Downloads and processes required geospatial data layers including elevation,
        soil classification, and land cover data from configured data sources.
        """
        self.acquisition_service.acquire_attributes()

    def acquire_forcings(self):
        """
        Acquire meteorological forcing data for the simulation period.

        Downloads forcing variables (precipitation, temperature, radiation, etc.)
        from the configured forcing dataset (ERA5, RDRS, CARRA, etc.) for the
        specified temporal domain.
        """
        self.acquisition_service.acquire_forcings()

    def acquire_observations(self):
        """
        Acquire observational data for model calibration and validation.

        Downloads streamflow observations, snow measurements, and other validation
        data from configured observation sources (USGS, WSC, SNOTEL, etc.).
        """
        self.acquisition_service.acquire_observations()

    def acquire_em_earth_forcings(self):
        """
        Acquire EM-Earth supplementary forcing data.

        Downloads and processes EM-Earth reanalysis data for gap-filling or
        supplementing primary forcing datasets.
        """
        self.acquisition_service.acquire_em_earth_forcings()

    def process_observed_data(self):
        """
        Process observed data including streamflow and additional variables.

        Raises:
            DataAcquisitionError: If data processing fails
        """
        self.logger.info("Processing observed data")
        self.acquire_observations()

        with symfluence_error_handler(
            "observed data processing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # 1. Parse observations to process
            additional_obs = self._get_config_value(
                lambda: self.config.data.additional_observations,
                []
            )
            if additional_obs is None:
                additional_obs = []
            elif isinstance(additional_obs, str):
                additional_obs = [o.strip() for o in additional_obs.split(',')]

            # 2. Check for primary streamflow provider and handle USGS/WSC migration
            streamflow_provider = str(self._get_config_value(
                lambda: self.config.data.streamflow_data_provider,
                ''
            )).upper()
            if streamflow_provider == 'USGS' and 'USGS_STREAMFLOW' not in additional_obs:
                # Automatically add USGS_STREAMFLOW if it's the primary provider but not in additional_obs
                additional_obs.append('USGS_STREAMFLOW')
            elif streamflow_provider == 'WSC' and 'WSC_STREAMFLOW' not in additional_obs:
                additional_obs.append('WSC_STREAMFLOW')
            elif streamflow_provider == 'SMHI' and 'SMHI_STREAMFLOW' not in additional_obs:
                additional_obs.append('SMHI_STREAMFLOW')
            elif streamflow_provider == 'LAMAH_ICE' and 'LAMAH_ICE_STREAMFLOW' not in additional_obs:
                additional_obs.append('LAMAH_ICE_STREAMFLOW')

            # Check for USGS Groundwater download and ensure it's in additional_obs
            download_usgs_gw = self._get_config_value(
                lambda: self.config.evaluation.usgs_gw.download,
                False
            )
            if isinstance(download_usgs_gw, str):
                download_usgs_gw = download_usgs_gw.lower() == 'true'

            if download_usgs_gw and 'USGS_GW' not in additional_obs:
                additional_obs.append('USGS_GW')

            # Check for MODIS Snow and ensure it's in additional_obs
            download_modis_snow = self._get_config_value(
                lambda: self.config.evaluation.modis_snow.download,
                False
            )
            if download_modis_snow and 'MODIS_SNOW' not in additional_obs:
                additional_obs.append('MODIS_SNOW')

            # Check for SNOTEL download and ensure it's in additional_obs
            download_snotel = self._get_config_value(
                lambda: self.config.evaluation.snotel.download,
                False
            )
            if isinstance(download_snotel, str):
                download_snotel = download_snotel.lower() == 'true'

            if download_snotel and 'SNOTEL' not in additional_obs:
                additional_obs.append('SNOTEL')

            # Check for ISMN download and ensure it's in additional_obs
            download_ismn = self._get_config_value(
                lambda: self.config.data.download_ismn,
                False
            )
            if isinstance(download_ismn, str):
                download_ismn = download_ismn.lower() == 'true'

            if download_ismn and 'ISMN' not in additional_obs:
                additional_obs.append('ISMN')

            # 3. Traditional streamflow processing (for providers not yet migrated)
            observed_data_processor = ObservedDataProcessor(self.config, self.logger)

            # Only run traditional if NOT using the formalized handlers
            formalized_providers = ['USGS_STREAMFLOW', 'WSC_STREAMFLOW', 'SMHI_STREAMFLOW', 'LAMAH_ICE_STREAMFLOW']
            is_formalized = any(obs in additional_obs for obs in formalized_providers)

            if not is_formalized:
                observed_data_processor.process_streamflow_data()

            observed_data_processor.process_fluxnet_data()

            # 4. Registry-based additional observations (GRACE, MODIS, USGS, etc.)

            for obs_type in additional_obs:
                try:
                    if ObservationRegistry.is_registered(obs_type):
                        self.logger.info(f"Processing registry-based observation: {obs_type}")
                        handler = ObservationRegistry.get_handler(obs_type, self.config, self.logger)
                        raw_path = handler.acquire()
                        processed_path = handler.process(raw_path)

                        # Visualize processed data
                        if self.reporting_manager and processed_path and processed_path.exists():
                            if processed_path.suffix == '.csv':
                                df = pd.read_csv(processed_path)
                                # Assuming first numeric column is the value
                                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                                if not numeric_cols.empty:
                                    self.reporting_manager.visualize_data_distribution(
                                        df[numeric_cols[0]],
                                        variable_name=f"{obs_type}_{numeric_cols[0]}",
                                        stage='preprocessing'
                                    )
                            elif processed_path.suffix in ['.tif', '.nc']:
                                self.reporting_manager.visualize_spatial_coverage(
                                    processed_path,
                                    variable_name=obs_type,
                                    stage='preprocessing'
                                )

                    else:
                        self.logger.warning(f"Observation type {obs_type} requested but no handler registered.")
                except Exception as e:
                    self.logger.warning(f"Failed to process additional observation {obs_type}: {e}")

            self.logger.info("Observed data processing completed successfully")

    def run_model_agnostic_preprocessing(self):
        """
        Run model-agnostic preprocessing including basin averaging and resampling.

        Raises:
            DataAcquisitionError: If preprocessing fails
        """
        # Create required directories
        basin_averaged_data = self.project_dir / 'forcing' / 'basin_averaged_data'
        catchment_intersection_dir = self.project_dir / 'shapefiles' / 'catchment_intersection'

        basin_averaged_data.mkdir(parents=True, exist_ok=True)
        catchment_intersection_dir.mkdir(parents=True, exist_ok=True)

        with symfluence_error_handler(
            "model-agnostic preprocessing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # Run geospatial statistics
            self.logger.debug("Running geospatial statistics")
            gs = GeospatialStatistics(self.config, self.logger)
            gs.run_statistics()

            # Run forcing resampling
            self.logger.debug("Running forcing resampling")
            fr = ForcingResampler(self.config, self.logger)
            fr.run_resampling()

            # Visualize preprocessed forcing if available
            if self.reporting_manager:
                try:
                    # Check for basin averaged files
                    basin_files = list(basin_averaged_data.glob("*.nc"))
                    if basin_files:
                        self.reporting_manager.visualize_spatial_coverage(basin_files[0], 'forcing_processed', 'preprocessing')
                except Exception as e:
                    self.logger.warning(f"Failed to visualize preprocessed forcing: {e}")

                # Visualize raw vs remapped forcing comparison
                try:
                    self._visualize_forcing_comparison(basin_averaged_data)
                except Exception as e:
                    self.logger.warning(f"Failed to visualize forcing comparison: {e}")

            # Integrate EM-Earth data if supplementation is enabled
            supplement_forcing = self._get_config_value(
                lambda: self.config.forcing.supplement,
                False
            )
            if supplement_forcing:
                self.logger.debug("Integrating EM-Earth data")
                self.em_earth_integrator.integrate_em_earth_data()

            self.logger.info("Model-agnostic preprocessing completed successfully")

    def validate_data_directories(self) -> bool:
        """Validate that required data directories exist."""
        return self.acquisition_service.validate_data_directories() if hasattr(self.acquisition_service, 'validate_data_directories') else self._validate_directories_fallback()

    def _validate_directories_fallback(self) -> bool:
        required_dirs = [
            self.project_dir / 'attributes',
            self.project_dir / 'forcing',
            self.project_dir / 'observations',
            self.project_dir / 'shapefiles'
        ]
        all_exist = True
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.logger.warning(f"Required directory does not exist: {dir_path}")
                all_exist = False
        return all_exist

    def _visualize_forcing_comparison(self, basin_averaged_data: Path) -> None:
        """
        Visualize raw vs. remapped forcing comparison.

        Args:
            basin_averaged_data: Path to basin averaged data directory
        """
        if not self.reporting_manager:
            return

        # Find remapped file (basin averaged)
        remapped_files = list(basin_averaged_data.glob("*.nc"))
        if not remapped_files:
            self.logger.debug("No remapped forcing files found for comparison visualization")
            return
        remapped_forcing_file = remapped_files[0]

        # Find raw forcing file (check merged_data first, then raw_data)
        raw_forcing_dir = self.project_dir / 'forcing' / 'merged_data'
        if not raw_forcing_dir.exists() or not list(raw_forcing_dir.glob("*.nc")):
            raw_forcing_dir = self.project_dir / 'forcing' / 'raw_data'

        raw_files = list(raw_forcing_dir.glob("*.nc")) if raw_forcing_dir.exists() else []
        if not raw_files:
            self.logger.debug("No raw forcing files found for comparison visualization")
            return
        raw_forcing_file = raw_files[0]

        # Find forcing grid shapefile
        forcing_grid_shp = self._find_forcing_shapefile()
        if forcing_grid_shp is None:
            self.logger.debug("Forcing grid shapefile not found for comparison visualization")
            return

        # Find HRU shapefile
        hru_shp = self._find_hru_shapefile()
        if hru_shp is None:
            self.logger.debug("HRU shapefile not found for comparison visualization")
            return

        # Call visualization
        self.reporting_manager.visualize_forcing_comparison(
            raw_forcing_file=raw_forcing_file,
            remapped_forcing_file=remapped_forcing_file,
            forcing_grid_shp=forcing_grid_shp,
            hru_shp=hru_shp
        )

    def _find_hru_shapefile(self) -> Optional[Path]:
        """
        Find the HRU/catchment shapefile.

        Returns:
            Path to HRU shapefile, or None if not found
        """
        catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        if not catchment_dir.exists():
            return None

        # Try to find HRU shapefile based on common naming patterns
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            'domain'
        )

        # Try explicit config value first
        catchment_name = self._get_config_value(
            lambda: self.config.paths.catchment_name,
            'default'
        )
        if catchment_name != 'default':
            explicit_path = catchment_dir / catchment_name
            if explicit_path.exists():
                return explicit_path

        # Search for HRU shapefiles with common patterns
        patterns = [
            f"{domain_name}_HRUs_*.shp",
            f"{domain_name}_catchment*.shp",
            "*HRU*.shp",
            "*catchment*.shp",
            "*.shp"  # Fallback to any shapefile
        ]

        for pattern in patterns:
            matches = list(catchment_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _find_forcing_shapefile(self) -> Optional[Path]:
        """
        Find the forcing grid shapefile.

        Returns:
            Path to forcing shapefile, or None if not found
        """
        forcing_shp_dir = self.project_dir / 'shapefiles' / 'forcing'
        if not forcing_shp_dir.exists():
            return None

        # Try explicit config value first
        forcing_dataset = self._get_config_value(
            lambda: self.config.forcing.dataset,
            'ERA5'
        )
        expected_path = forcing_shp_dir / f"forcing_{forcing_dataset}.shp"
        if expected_path.exists():
            return expected_path

        # Search for any forcing shapefile (handles cases like 'local' dataset)
        patterns = [
            "forcing_*.shp",
            "*.shp"  # Fallback to any shapefile
        ]

        for pattern in patterns:
            matches = list(forcing_shp_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_data_status(self) -> Dict[str, Any]:
        """Get status of data acquisition and preprocessing."""
        status = {
            'project_dir': str(self.project_dir),
            'attributes_acquired': (self.project_dir / 'attributes' / 'elevation' / 'dem').exists(),
            'forcings_acquired': (self.project_dir / 'forcing' / 'raw_data').exists(),
            'forcings_preprocessed': (self.project_dir / 'forcing' / 'basin_averaged_data').exists(),
            'observed_data_processed': (self.project_dir / 'observations' / 'streamflow' / 'preprocessed').exists(),
        }

        status['dem_exists'] = (self.project_dir / 'attributes' / 'elevation' / 'dem').exists()
        status['soilclass_exists'] = (self.project_dir / 'attributes' / 'soilclass').exists()
        status['landclass_exists'] = (self.project_dir / 'attributes' / 'landclass').exists()

        supplement_forcing = self._get_config_value(
            lambda: self.config.forcing.supplement,
            False
        )
        if supplement_forcing:
            status['em_earth_acquired'] = (self.project_dir / 'forcing' / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_dir / 'forcing' / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False

        return status
