"""
SUMMA model preprocessor.

This module contains the main SummaPreProcessor class that orchestrates
the preprocessing workflow for SUMMA model runs.
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import pandas as pd
import xarray as xr

# Local imports
from symfluence.utils.models.registry import ModelRegistry
from symfluence.utils.models.base import BaseModelPreProcessor
from symfluence.utils.models.mixins import ObservationLoaderMixin
from .forcing_processor import SummaForcingProcessor
from .config_manager import SummaConfigManager
from .attributes_manager import SummaAttributesManager
from symfluence.utils.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_preprocessor('SUMMA')
class SummaPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Preprocessor for the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    Handles data preparation, configuration, and file setup for SUMMA model runs.
    Inherits observation loading from ObservationLoaderMixin.
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

        # SUMMA-specific configuration (Phase 3: Use typed config when available)
        if self.typed_config:
            # Typed access (clearer, type-safe)
            self.hruId = self.typed_config.paths.catchment_hruid
            self.gruId = self.typed_config.paths.catchment_gruid
            self.data_step = self.typed_config.forcing.time_step_size
            self.coldstate_name = self.typed_config.model.summa.coldstate if self.typed_config.model.summa else None
            self.parameter_name = self.typed_config.model.summa.trialparams if self.typed_config.model.summa else None
            self.attribute_name = self.typed_config.model.summa.attributes if self.typed_config.model.summa else None
            self.forcing_measurement_height = float(self.typed_config.forcing.measurement_height)
        else:
            # Fallback to dict access for backward compatibility
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

        Uses the template method pattern from BaseModelPreProcessor.

        Raises:
            ModelExecutionError: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting SUMMA spatial preprocessing")
        return self.run_preprocessing_template()

    def _pre_setup(self) -> None:
        """SUMMA-specific pre-setup: apply lapse rate corrections (template hook)."""
        self.apply_datastep_and_lapse_rate()

    def _prepare_forcing(self) -> None:
        """SUMMA-specific forcing preparation (template hook)."""
        self.create_forcing_file_list()

    def _create_model_configs(self) -> None:
        """SUMMA-specific configuration file creation (template hook)."""
        self.create_file_manager()
        self.create_initial_conditions()
        self.create_trial_parameters()
        self.create_attributes_file()

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
        if self.data_step != self.forcing_processor.data_step:
            self.data_step = self.forcing_processor.data_step
            self.config_manager.data_step = self.forcing_processor.data_step
            self.logger.info(f"Updated SUMMA data step to {self.data_step}s based on forcing data")

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
        # Phase 3: Use typed config when available
        if self.typed_config:
            sim_start = self.typed_config.domain.time_start
            sim_end = self.typed_config.domain.time_end
        else:
            sim_start = self.config.get('EXPERIMENT_TIME_START')
            sim_end = self.config.get('EXPERIMENT_TIME_END')

        if sim_start == 'default' or sim_end == 'default':
            start_year = sim_start.split('-')[0] if sim_start != 'default' else None
            end_year = sim_end.split('-')[0] if sim_end != 'default' else None
            if not start_year or not end_year:
                raise ValueError("EXPERIMENT_TIME_START or EXPERIMENT_TIME_END is missing from configuration")
            sim_start = f"{start_year}-01-01 01:00" if sim_start == 'default' else sim_start
            sim_end = f"{end_year}-12-31 22:00" if sim_end == 'default' else sim_end

        forcing_times = self._get_forcing_times()
        if forcing_times:
            sim_start_dt = datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            sim_end_dt = datetime.strptime(sim_end, "%Y-%m-%d %H:%M")

            start_floor = max((t for t in forcing_times if t <= sim_start_dt), default=forcing_times[0])
            end_ceil = min((t for t in forcing_times if t >= sim_end_dt), default=forcing_times[-1])

            if start_floor != sim_start_dt:
                self.logger.info(
                    f"Adjusting SUMMA start time to forcing timestep: {start_floor}"
                )
            if end_ceil != sim_end_dt:
                self.logger.info(
                    f"Adjusting SUMMA end time to forcing timestep: {end_ceil}"
                )

            if start_floor > end_ceil:
                self.logger.warning(
                    "Forcing timesteps do not cover requested range; using full forcing span."
                )
                start_floor = forcing_times[0]
                end_ceil = forcing_times[-1]

            sim_start = start_floor.strftime("%Y-%m-%d %H:%M")
            sim_end = end_ceil.strftime("%Y-%m-%d %H:%M")

        # Validate time format
        try:
            datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            datetime.strptime(sim_end, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError("Invalid time format in configuration. Expected 'YYYY-MM-DD HH:MM'")

        return sim_start, sim_end

    def _get_forcing_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        forcing_times = self._get_forcing_times()
        if not forcing_times:
            return None

        return (forcing_times[0], forcing_times[-1])

    def _get_forcing_times(self) -> list[datetime]:
        forcing_dir = self.forcing_summa_path
        if not forcing_dir.exists():
            return []

        forcing_files = sorted(forcing_dir.glob("*.nc"))
        if not forcing_files:
            return []

        unique_times = set()
        for forcing_file in forcing_files:
            try:
                with xr.open_dataset(forcing_file) as ds:
                    if "time" not in ds:
                        continue
                    times = pd.to_datetime(ds["time"].values)
            except Exception as exc:
                self.logger.warning(f"Failed to read forcing times from {forcing_file}: {exc}")
                continue

            if len(times) == 0:
                continue
            unique_times.update(pd.to_datetime(times).to_pydatetime())

        if not unique_times:
            return []

        return sorted(unique_times)
