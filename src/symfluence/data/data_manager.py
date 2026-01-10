"""
Data management facade for SYMFLUENCE hydrological modeling.

Coordinates all data acquisition and preprocessing operations through
specialized services for forcing data, observations, and attribute processing.
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union

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
from symfluence.data.utilities.variable_utils import VariableHandler

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class DataManager(BaseManager):
    """
    Manages all data acquisition and preprocessing operations for SYMFLUENCE.

    Acts as a facade, delegating specialized tasks to:
    - AcquisitionService: For data download and initial acquisition
    - EMEarthIntegrator: For EM-Earth data processing
    - ObservedDataProcessor: For streamflow/observation processing
    - GeospatialStatistics/ForcingResampler: For preprocessing

    Inherits from BaseManager for standardized initialization and common patterns.
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
                lambda: self.config.data.download_usgs_gw,
                False
            )
            if isinstance(download_usgs_gw, str):
                download_usgs_gw = download_usgs_gw.lower() == 'true'
            
            if download_usgs_gw and 'USGS_GW' not in additional_obs:
                additional_obs.append('USGS_GW')
            
            # Check for MODIS Snow and ensure it's in additional_obs
            download_modis_snow = self._get_config_value(
                lambda: self.config.data.download_modis_snow,
                False
            )
            if download_modis_snow and 'MODIS_SNOW' not in additional_obs:
                additional_obs.append('MODIS_SNOW')
            
            # Check for SNOTEL download and ensure it's in additional_obs
            download_snotel = self._get_config_value(
                lambda: self.config.data.download_snotel,
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

            # Integrate EM-Earth data if supplementation is enabled
            supplement_forcing = self._get_config_value(
                lambda: self.config.data.supplement_forcing,
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
        
        supplement_forcing = self._get_config_value(
            lambda: self.config.data.supplement_forcing,
            False
        )
        if supplement_forcing:
            status['em_earth_acquired'] = (self.project_dir / 'forcing' / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_dir / 'forcing' / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False
        
        return status
