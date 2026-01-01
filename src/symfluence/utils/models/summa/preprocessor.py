"""
SUMMA model preprocessor.

This module contains the main SummaPreProcessor class that orchestrates
the preprocessing workflow for SUMMA model runs.
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Local imports
from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPreProcessor
from .forcing_processor import SummaForcingProcessor
from .config_manager import SummaConfigManager
from .attributes_manager import SummaAttributesManager
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_preprocessor('SUMMA')
class SummaPreProcessor(BaseModelPreProcessor):
    """
    Preprocessor for the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    Handles data preparation, configuration, and file setup for SUMMA model runs.
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "SUMMA"

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the SummaPreProcessor.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing setup parameters.
            logger (Any): Logger object for recording processing information.
        """
        # Initialize base class (handles standard paths and common setup)
        super().__init__(config, logger)

        # SUMMA-specific paths (base class now handles shapefile_path, merged_forcing_path, intersect_path)
        self.dem_path = self.get_dem_path()
        self.forcing_summa_path = self.project_dir / 'forcing' / 'SUMMA_input'

        # Catchment and river network (use base class methods)
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"

        self.river_network_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        self.river_network_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        if self.river_network_name == 'default':
            self.river_network_name = f"{self.domain_name}_riverNetwork_delineate.shp"

        # SUMMA-specific configuration (base class now handles forcing_dataset)
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')
        self.gruId = self.config.get('CATCHMENT_SHP_GRUID')
        self.data_step = self.forcing_time_step_size  # Use base class attribute
        self.coldstate_name = self.config.get('SETTINGS_SUMMA_COLDSTATE')
        self.parameter_name = self.config.get('SETTINGS_SUMMA_TRIALPARAMS')
        self.attribute_name = self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        self.forcing_measurement_height = float(self.config.get('FORCING_MEASUREMENT_HEIGHT'))

        # Initialize forcing processor
        self.forcing_processor = SummaForcingProcessor(
            config=self.config,
            logger=self.logger,
            forcing_basin_path=self.forcing_basin_path,
            forcing_summa_path=self.forcing_summa_path,
            intersect_path=self.intersect_path,
            catchment_path=self.catchment_path,
            project_dir=self.project_dir,
            setup_dir=self.setup_dir,
            domain_name=self.domain_name,
            forcing_dataset=self.forcing_dataset,
            data_step=self.data_step,
            gruId=self.gruId,
            hruId=self.hruId,
            catchment_name=self.catchment_name
        )

        # Initialize configuration manager
        self.config_manager = SummaConfigManager(
            config=self.config,
            logger=self.logger,
            project_dir=self.project_dir,
            setup_dir=self.setup_dir,
            forcing_summa_path=self.forcing_summa_path,
            catchment_path=self.catchment_path,
            catchment_name=self.catchment_name,
            dem_path=self.dem_path,
            hruId=self.hruId,
            gruId=self.gruId,
            data_step=self.data_step,
            coldstate_name=self.coldstate_name,
            parameter_name=self.parameter_name,
            attribute_name=self.attribute_name,
            forcing_measurement_height=self.forcing_measurement_height,
            filter_forcing_hru_ids_callback=self._filter_forcing_hru_ids,
            get_base_settings_source_dir_callback=self.get_base_settings_source_dir,
            get_default_path_callback=self._get_default_path,
            get_simulation_times_callback=self._get_simulation_times
        )

        # Initialize attributes manager
        self.attributes_manager = SummaAttributesManager(
            config=self.config,
            logger=self.logger,
            catchment_path=self.catchment_path,
            catchment_name=self.catchment_name,
            dem_path=self.dem_path,
            forcing_summa_path=self.forcing_summa_path,
            setup_dir=self.setup_dir,
            project_dir=self.project_dir,
            hruId=self.hruId,
            gruId=self.gruId,
            attribute_name=self.attribute_name,
            forcing_measurement_height=self.forcing_measurement_height,
            get_default_path_callback=self._get_default_path
        )

    def run_preprocessing(self):
        """
        Run the complete SUMMA spatial preprocessing workflow.

        This method orchestrates the preprocessing pipeline.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting SUMMA spatial preprocessing")

        with symfluence_error_handler(
            "SUMMA preprocessing",
            self.logger,
            error_type=ModelExecutionError
        ):
            self.apply_datastep_and_lapse_rate()
            self.copy_base_settings()
            self.create_file_manager()
            self.create_forcing_file_list()
            self.create_initial_conditions()
            self.create_trial_parameters()
            self.create_attributes_file()

            self.logger.info("SUMMA spatial preprocessing completed successfully")

    def copy_base_settings(self):
        """
        Copy SUMMA base settings from the source directory to the project's settings directory.

        Delegates to the configuration manager.
        """
        self.config_manager.copy_base_settings()


    def create_file_manager(self):
        """
        Create the SUMMA file manager configuration file.

        Delegates to the configuration manager.
        """
        self.config_manager.create_file_manager()

    def apply_datastep_and_lapse_rate(self):
        """
        Apply temperature lapse rate corrections to forcing data.

        Delegates to the forcing processor for actual implementation.
        """
        self.forcing_processor.apply_datastep_and_lapse_rate()

    def create_forcing_file_list(self):
        """
        Create a list of forcing files for SUMMA.

        Delegates to the forcing processor for actual implementation.
        """
        self.forcing_processor.create_forcing_file_list()

    def _filter_forcing_hru_ids(self, forcing_hru_ids):
        """
        Filter forcing HRU IDs against catchment shapefile.

        Delegates to the forcing processor for actual implementation.

        Args:
            forcing_hru_ids: List or array of HRU IDs from forcing data

        Returns:
            Filtered list of HRU IDs
        """
        return self.forcing_processor._filter_forcing_hru_ids(forcing_hru_ids)



    def create_initial_conditions(self):
        """
        Create the initial conditions (cold state) file for SUMMA.

        Delegates to the configuration manager.
        """
        self.config_manager.create_initial_conditions()


    def create_trial_parameters(self):
        """
        Create the trial parameters file for SUMMA.

        Delegates to the configuration manager.
        """
        self.config_manager.create_trial_parameters()



    def create_attributes_file(self):
        """
        Create the attributes file for SUMMA.

        Delegates to the attributes manager for actual implementation.
        """
        self.attributes_manager.create_attributes_file()

    def _get_simulation_times(self) -> tuple[str, str]:
        """
        Get the simulation start and end times from config or calculate defaults.

        Returns:
            tuple[str, str]: A tuple containing the simulation start and end times.

        Raises:
            ValueError: If the time format in the configuration is invalid.
        """
        sim_start = self.config.get('EXPERIMENT_TIME_START')
        sim_end = self.config.get('EXPERIMENT_TIME_END')

        if sim_start == 'default' or sim_end == 'default':
            start_year = self.config.get('EXPERIMENT_TIME_START').split('-')[0]
            end_year = self.config.get('EXPERIMENT_TIME_END').split('-')[0]
            if not start_year or not end_year:
                raise ValueError("EXPERIMENT_TIME_START or EXPERIMENT_TIME_END is missing from configuration")
            sim_start = f"{start_year}-01-01 01:00" if sim_start == 'default' else sim_start
            sim_end = f"{end_year}-12-31 22:00" if sim_end == 'default' else sim_end

        # Validate time format
        try:
            datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            datetime.strptime(sim_end, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError("Invalid time format in configuration. Expected 'YYYY-MM-DD HH:MM'")

        return sim_start, sim_end
