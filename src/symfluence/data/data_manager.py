
# In utils/dataHandling_utils/data_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union
from symfluence.data.preprocessing.forcing_resampler import ForcingResampler
from symfluence.data.preprocessing.geospatial_statistics import GeospatialStatistics
from symfluence.data.utilities.variable_utils import VariableHandler
from symfluence.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.data.observation.registry import ObservationRegistry
from symfluence.data.acquisition.acquisition_service import AcquisitionService
from symfluence.data.preprocessing.em_earth_integrator import EMEarthIntegrator
from symfluence.core.exceptions import (
    DataAcquisitionError,
    symfluence_error_handler
)

from symfluence.core.mixins import ConfigurableMixin

# Import for type checking only (avoid circular imports)
try:
    from symfluence.core.config.models import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None

class DataManager(ConfigurableMixin):
    """
    Manages all data acquisition and preprocessing operations for SYMFLUENCE.
    
    Acts as a facade, delegating specialized tasks to:
    - AcquisitionService: For data download and initial acquisition
    - EMEarthIntegrator: For EM-Earth data processing
    - ObservedDataProcessor: For streamflow/observation processing
    - GeospatialStatistics/ForcingResampler: For preprocessing
    """
    
    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        # Phase 2: Support both typed config and dict config for backward compatibility
        if SymfluenceConfig and isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.typed_config = None
            self.config = config

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Always use dict config for delegates (they expect Dict, not typed config objects)
        component_config = self.config

        # Initialize delegates
        self.acquisition_service = AcquisitionService(component_config, logger, reporting_manager)
        self.em_earth_integrator = EMEarthIntegrator(component_config, logger)
        self.variable_handler = VariableHandler(component_config, self.logger, 'ERA5', 'SUMMA')
        
    def acquire_attributes(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_attributes()
        
    def acquire_forcings(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_forcings()

    def acquire_observations(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_observations()

    def acquire_em_earth_forcings(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_em_earth_forcings()
            
    def process_observed_data(self):
        """
        Process observed data including streamflow and additional variables.

        Raises:
            DataAcquisitionError: If data processing fails
        """
        self.logger.info("Processing observed data")
        
        # Use typed config if available
        component_config = self.typed_config if self.typed_config else self.config

        with symfluence_error_handler(
            "observed data processing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # 1. Parse observations to process
            additional_obs = self._resolve_config_value(
                lambda: self.typed_config.data.additional_observations,
                'ADDITIONAL_OBSERVATIONS',
                []
            )
            if isinstance(additional_obs, str):
                additional_obs = [o.strip() for o in additional_obs.split(',')]
            
            # 2. Check for primary streamflow provider and handle USGS/WSC migration
            streamflow_provider = str(self._resolve_config_value(
                lambda: self.typed_config.data.streamflow_data_provider,
                'STREAMFLOW_DATA_PROVIDER',
                ''
            )).upper()
            if streamflow_provider == 'USGS' and 'USGS_STREAMFLOW' not in additional_obs:
                # Automatically add USGS_STREAMFLOW if it's the primary provider but not in additional_obs
                additional_obs.append('USGS_STREAMFLOW')
            elif streamflow_provider == 'WSC' and 'WSC_STREAMFLOW' not in additional_obs:
                additional_obs.append('WSC_STREAMFLOW')
            
            # Check for USGS Groundwater download and ensure it's in additional_obs
            download_usgs_gw = self._resolve_config_value(
                lambda: self.typed_config.data.download_usgs_gw,
                'DOWNLOAD_USGS_GW',
                False
            )
            if isinstance(download_usgs_gw, str):
                download_usgs_gw = download_usgs_gw.lower() == 'true'
            
            if download_usgs_gw and 'USGS_GW' not in additional_obs:
                additional_obs.append('USGS_GW')
            
            # Check for MODIS Snow and ensure it's in additional_obs
            download_modis_snow = self._resolve_config_value(
                lambda: self.typed_config.data.download_modis_snow,
                'DOWNLOAD_MODIS_SNOW',
                False
            )
            if download_modis_snow and 'MODIS_SNOW' not in additional_obs:
                additional_obs.append('MODIS_SNOW')
            
            # Check for SNOTEL download and ensure it's in additional_obs
            download_snotel = self._resolve_config_value(
                lambda: self.typed_config.data.download_snotel,
                'DOWNLOAD_SNOTEL',
                False
            )
            if isinstance(download_snotel, str):
                download_snotel = download_snotel.lower() == 'true'
            
            if download_snotel and 'SNOTEL' not in additional_obs:
                additional_obs.append('SNOTEL')

            # 3. Traditional streamflow processing (for providers not yet migrated)
            observed_data_processor = ObservedDataProcessor(component_config, self.logger)
            
            # Only run traditional if NOT using the formalized handlers
            if streamflow_provider not in ['USGS', 'WSC'] or (
                'USGS_STREAMFLOW' not in additional_obs and 'WSC_STREAMFLOW' not in additional_obs
            ):
                observed_data_processor.process_streamflow_data()
            
            observed_data_processor.process_fluxnet_data()

            # 4. Registry-based additional observations (GRACE, MODIS, USGS, etc.)

            for obs_type in additional_obs:
                try:
                    if ObservationRegistry.is_registered(obs_type):
                        self.logger.info(f"Processing registry-based observation: {obs_type}")
                        handler = ObservationRegistry.get_handler(obs_type, component_config, self.logger)
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
        # Use typed config if available
        component_config = self.typed_config if self.typed_config else self.config

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
            gs = GeospatialStatistics(component_config, self.logger)
            gs.run_statistics()

            # Run forcing resampling
            self.logger.debug("Running forcing resampling")
            fr = ForcingResampler(component_config, self.logger)
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

            # Integrate EM-Earth data if supplementation is enabled
            supplement_forcing = self._resolve_config_value(
                lambda: self.typed_config.data.supplement_forcing,
                'SUPPLEMENT_FORCING',
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
        
        supplement_forcing = self._resolve_config_value(
            lambda: self.typed_config.data.supplement_forcing,
            'SUPPLEMENT_FORCING',
            False
        )
        if supplement_forcing:
            status['em_earth_acquired'] = (self.project_dir / 'forcing' / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_dir / 'forcing' / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False
        
        return status
