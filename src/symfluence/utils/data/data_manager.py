
# In utils/dataHandling_utils/data_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union
from symfluence.utils.data.preprocessing.agnosticPreProcessor import ForcingResampler, GeospatialStatistics
from symfluence.utils.data.utilities.variable_utils import VariableHandler
from symfluence.utils.data.acquisition.observed_processor import ObservedDataProcessor
from symfluence.utils.data.observation.registry import ObservationRegistry
from symfluence.utils.data.acquisition.acquisition_service import AcquisitionService
from symfluence.utils.data.preprocessing.em_earth_integrator import EMEarthIntegrator
from symfluence.utils.exceptions import (
    DataAcquisitionError,
    symfluence_error_handler
)

# Import for type checking only (avoid circular imports)
try:
    from symfluence.utils.config.models import SymfluenceConfig
except ImportError:
    SymfluenceConfig = None

class DataManager:
    """
    Manages all data acquisition and preprocessing operations for SYMFLUENCE.
    
    Acts as a facade, delegating specialized tasks to:
    - AcquisitionService: For data download and initial acquisition
    - EMEarthIntegrator: For EM-Earth data processing
    - ObservedDataProcessor: For streamflow/observation processing
    - GeospatialStatistics/ForcingResampler: For preprocessing
    """
    
    def __init__(self, config: Union[Dict[str, Any], 'SymfluenceConfig'], logger: logging.Logger):
        # Phase 2: Support both typed config and dict config for backward compatibility
        if SymfluenceConfig and isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.typed_config = None
            self.config = config

        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Initialize delegates
        self.acquisition_service = AcquisitionService(config, logger)
        self.em_earth_integrator = EMEarthIntegrator(config, logger)
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        
    def acquire_attributes(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_attributes()
        
    def acquire_forcings(self):
        """Delegate to AcquisitionService."""
        self.acquisition_service.acquire_forcings()

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

        with symfluence_error_handler(
            "observed data processing",
            self.logger,
            error_type=DataAcquisitionError
        ):
            # 1. Traditional streamflow processing
            observed_data_processor = ObservedDataProcessor(self.config, self.logger)
            observed_data_processor.process_streamflow_data()
            observed_data_processor.process_snotel_data()
            observed_data_processor.process_fluxnet_data()
            observed_data_processor.process_usgs_groundwater_data()

            # 2. Registry-based additional observations (GRACE, MODIS, etc.)
            additional_obs = self.config.get('ADDITIONAL_OBSERVATIONS', [])
            if isinstance(additional_obs, str):
                additional_obs = [o.strip() for o in additional_obs.split(',')]

            for obs_type in additional_obs:
                try:
                    handler = ObservationRegistry.get_handler(obs_type, self.config, self.logger)
                    raw_path = handler.acquire()
                    handler.process(raw_path)
                except Exception as e:
                    self.logger.warning(f"Failed to process additional observation {obs_type}: {e}")

            self.logger.info("Observed data processing completed successfully")

    def run_model_agnostic_preprocessing(self):
        """
        Run model-agnostic preprocessing including basin averaging and resampling.

        Raises:
            DataAcquisitionError: If preprocessing fails
        """
        self.logger.info("Starting model-agnostic preprocessing")

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
            self.logger.info("Running geospatial statistics")
            gs = GeospatialStatistics(self.config, self.logger)
            gs.run_statistics()

            # Run forcing resampling
            self.logger.info("Running forcing resampling")
            fr = ForcingResampler(self.config, self.logger)
            fr.run_resampling()

            # Integrate EM-Earth data if supplementation is enabled
            if self.config.get('SUPPLEMENT_FORCING', False):
                self.logger.info("SUPPLEMENT_FORCING enabled - integrating EM-Earth data")
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
        
        if self.config.get('SUPPLEMENT_FORCING', False):
            status['em_earth_acquired'] = (self.project_dir / 'forcing' / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_dir / 'forcing' / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False
        
        return status
