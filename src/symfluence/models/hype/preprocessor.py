"""
HYPE model preprocessor.

Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
Uses the generalized pipeline pattern with manager classes for:
- Forcing data processing (HYPEForcingProcessor)
- Configuration file generation (HYPEConfigManager)
- Geographic data file generation (HYPEGeoDataManager)
"""

import logging
from typing import Any, Dict, Optional, cast

import pandas as pd

from symfluence.models.hype.forcing_processor import HYPEForcingProcessor
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor

from symfluence.data.utils.variable_utils import VariableHandler


@ModelRegistry.register_preprocessor('HYPE')
class HYPEPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    HYPE (HYdrological Predictions for the Environment) preprocessor for SYMFLUENCE.

    Handles complete preprocessing workflow for HYPE including forcing data processing,
    geographic data file generation, and configuration setup. HYPE is a semi-distributed
    hydrological model developed by SMHI (Sweden) that routes water through sub-basins
    with land use and soil type classifications.

    Key Operations:
        - Process forcing data into HYPE daily format (Pobs.txt, Tobs.txt)
        - Generate GeoData.txt with sub-basin characteristics
        - Create GeoClass.txt with land use and soil class definitions
        - Generate ForcKey.txt linking sub-basins to forcing stations
        - Set up info.txt with simulation settings and time periods
        - Create par.txt with model parameters
        - Configure internal routing network
        - Handle spinup period and time zone adjustments

    Workflow Steps:

        1. Initialize paths and load configuration
        2. Process forcing data and convert to daily timestep
        3. Generate geographic data files from catchment attributes
        4. Create land use and soil classification tables
        5. Set up routing network between sub-basins
        6. Configure simulation parameters and time settings
        7. Generate parameter file (par.txt) with calibration values
        8. Set up output specifications

    HYPE-Specific Features:
        - Internal routing: No need for external routing model
        - Sub-basin based: Discretization by drainage areas
        - Class-based: Multiple land use and soil classes per sub-basin
        - Daily timestep: Forcing data automatically aggregated to daily
        - Spinup handling: Automatic warm-up period configuration

    Required Input Files (Generated):
        - GeoData.txt: Sub-basin properties (area, elevation, routing)
        - GeoClass.txt: Land use and soil class fractions per sub-basin
        - ForcKey.txt: Mapping of forcing stations to sub-basins
        - Pobs.txt: Daily precipitation forcing
        - Tobs.txt: Daily temperature forcing
        - info.txt: Simulation settings and time configuration
        - par.txt: Model parameters
        - filedir.txt: File path specifications

    Inherits from:
        BaseModelPreProcessor: Common preprocessing patterns and utilities
        ObservationLoaderMixin: Observation data loading capabilities

    Uses Manager Pattern:
        - HYPEForcingProcessor: Forcing data merging and daily conversion
        - HYPEConfigManager: Configuration file generation (info.txt, par.txt)
        - HYPEGeoDataManager: Geographic data file generation (GeoData.txt, etc.)

    Attributes:
        config (SymfluenceConfig): Typed configuration object
        logger: Logger object for recording processing information
        project_dir (Path): Directory for the current project
        hype_setup_dir (Path): Directory for HYPE setup files
        hype_results_dir (Path): Directory for HYPE simulation results
        forcing_input_dir (Path): Directory with remapped forcing data
        calibration_params (Optional[Dict]): Parameter values for calibration runs
        spinup_days (int): Number of spinup days before evaluation period
        timeshift (int): Time zone adjustment in hours
        frac_threshold (float): Minimum class fraction to include (default: 0.1)

    Example:
        >>> from symfluence.models.hype.preprocessor import HYPEPreProcessor
        >>> preprocessor = HYPEPreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
        # Creates HYPE input files in: project_dir/settings/HYPE/
        # Generates: GeoData.txt, GeoClass.txt, ForcKey.txt, Pobs.txt, Tobs.txt,
        #            info.txt, par.txt, filedir.txt

    Note:
        HYPE requires sub-basin delineation and land use/soil class data.
        The model handles internal routing, so mizuRoute is not needed.
    """

    MODEL_NAME = "HYPE"

    def __init__(self, config, logger: logging.Logger, params: Optional[Dict[str, Any]] = None):
        """
        Initialize HYPE preprocessor with SYMFLUENCE configuration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                HYPE settings, forcing dataset, and domain configuration.
            logger: Logger instance for status messages and debugging.
            params: Optional calibration parameter dictionary to override defaults.

        Note:
            Sets up HYPE-specific directories and initializes manager classes
            for forcing processing, geographic data, and configuration.
        """
        # Initialize base class
        super().__init__(config, logger)
        self.calibration_params = params
        self.gistool_output = f"{str(self.project_dir / 'attributes' / 'gistool-outputs')}/"
        # HYPE needs the remapped forcing data and geospatial statistics
        self.forcing_input_dir = self.forcing_basin_path
        self.hype_setup_dir = self.project_dir / 'settings' / 'HYPE'

        # Use typed config
        forcing_dataset = self._get_config_value(
            lambda: self.config.forcing.dataset
        )

        experiment_id = self._get_config_value(
            lambda: self.config.domain.experiment_id
        )
        self.hype_results_dir = self.project_dir / "simulations" / experiment_id / "HYPE"
        self.hype_results_dir.mkdir(parents=True, exist_ok=True)
        # HYPE results dir MUST have a trailing slash for the info.txt file
        self.hype_results_dir_str = str(self.hype_results_dir).rstrip('/') + '/'
        self.cache_path = self.project_dir / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize time parameters from typed config
        self.timeshift = self._get_config_value(
            lambda: self.config.model.hype.timeshift if self.config.model and self.config.model.hype else None,
            0
        )
        self.spinup_days = self._get_config_value(
            lambda: self.config.model.hype.spinup_days if self.config.model and self.config.model.hype else None
        )
        self.frac_threshold = self._get_config_value(
            lambda: self.config.model.hype.frac_threshold if self.config.model and self.config.model.hype else None,
            0.1
        )

        # If spinup_days not provided, calculate from SPINUP_PERIOD
        if self.spinup_days is None:
            spinup_period = self.config_dict.get('SPINUP_PERIOD')
            if spinup_period:
                try:
                    start_date, end_date = [pd.to_datetime(s.strip()) for s in spinup_period.split(',')]
                    self.spinup_days = (end_date - start_date).days
                    self.logger.debug(f"Calculated HYPE spinup days from SPINUP_PERIOD: {self.spinup_days}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate HYPE spinup from {spinup_period}: {e}")
                    self.spinup_days = 0
            else:
                self.spinup_days = 0

        self.spinup_days = int(self.spinup_days)

        # inputs
        self.output_path = self.hype_setup_dir
        # Store original forcing data directory for calibration workers
        # During calibration, output_path may change but forcing data stays in original location
        self.forcing_data_dir = self.hype_setup_dir

        # Basin-averaged forcing data is already in CFIF format (from model-agnostic preprocessing)
        # Initialize variable handler to get correct input names
        var_handler = VariableHandler(self.config_dict, self.logger, 'CFIF', 'HYPE')
        dataset_map = var_handler.DATASET_MAPPINGS['CFIF']

        # Get input names for temperature and precipitation
        temp_in = var_handler._find_matching_variable('air_temperature', dataset_map)
        precip_in = var_handler._find_matching_variable('precipitation_flux', dataset_map)

        self.forcing_units = {
            'temperature': {
                'in_varname': temp_in,
                'in_units': dataset_map[cast(str, temp_in)]['units'],
                'out_units': 'degC'
            },
            'precipitation': {
                'in_varname': precip_in,
                'in_units': dataset_map[cast(str, precip_in)]['units'],
                'out_units': 'mm/day'
            },
        }

        # mapping geofabric fields to model names
        self.geofabric_mapping = {
            'basinID': {'in_varname': self.config_dict.get('RIVER_BASIN_SHP_RM_GRUID')},
            'nextDownID': {'in_varname': self.config_dict.get('RIVER_NETWORK_SHP_DOWNSEGID')},
            'area': {'in_varname': self.config_dict.get('RIVER_BASIN_SHP_AREA'), 'in_units': 'm^2', 'out_units': 'm^2'},
            'rivlen': {'in_varname': self.config_dict.get('RIVER_NETWORK_SHP_LENGTH'), 'in_units': 'm', 'out_units': 'm'}
        }

        # domain subbasins and rivers - handle different delineation methods
        method_suffix = self._get_method_suffix()
        self.subbasins_shapefile = str(self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_{method_suffix}.shp')

        # River network file might not always exist for lumped domains, fallback to river_basins if needed
        network_file = self.project_dir / 'shapefiles' / 'river_network' / f'{self.domain_name}_riverNetwork_{method_suffix}.shp'
        if not network_file.exists():
            # If no network file, try generic or fallback
            network_file = self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_{method_suffix}.shp'

        self.rivers_shapefile = str(network_file)

        # Initialize manager classes
        self._init_managers(forcing_dataset)

    def _init_managers(self, forcing_dataset: str) -> None:
        """Initialize the manager classes for the generalized pipeline."""
        # Forcing processor
        self.forcing_processor = HYPEForcingProcessor(
            config=self.config_dict,
            logger=self.logger,
            forcing_input_dir=self.forcing_input_dir,
            output_path=self.output_path,
            cache_path=self.cache_path,
            timeshift=self.timeshift,
            forcing_units=self.forcing_units
        )

        # Configuration manager
        self.config_manager = HYPEConfigManager(
            config=self.config_dict,
            logger=self.logger,
            output_path=self.output_path
        )

        # GeoData manager
        self.geodata_manager = HYPEGeoDataManager(
            config=self.config_dict,
            logger=self.logger,
            output_path=self.output_path,
            geofabric_mapping=self.geofabric_mapping
        )

    def copy_base_settings(self, source_dir=None, file_patterns=None):
        """
        Override base class method - HYPE generates all configs dynamically.

        HYPE does not require base settings files to be copied from a template.
        All configuration files (par.txt, info.txt, filedir.txt, GeoData.txt,
        GeoClass.txt, ForcKey.txt) are generated programmatically by the
        HYPEConfigManager and HYPEGeoDataManager classes.
        """
        self.logger.debug("HYPE does not require base settings - all configs generated dynamically")

    def run_preprocessing(self):
        """
        Execute complete HYPE preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info("Starting HYPE preprocessing")
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """HYPE-specific forcing data preparation (template hook)."""
        self.forcing_processor.process_forcing()

    def _create_model_configs(self) -> None:
        """HYPE-specific configuration file creation (template hook)."""
        # Get basin shapefile path
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        method_suffix = self._get_method_suffix()
        basin_name = f"{self.domain_name}_riverBasins_{method_suffix}.shp"
        basin_path = basin_dir / basin_name

        # Fallback for legacy naming
        if not basin_path.exists() and self.domain_name == 'bow_banff_minimal':
            legacy_basin = basin_dir / "Bow_at_Banff_lumped_riverBasins_lumped.shp"
            if legacy_basin.exists():
                basin_path = legacy_basin
                self.logger.info(f"Using legacy basins path: {basin_path.name}")

        # Get river network path
        river_dir = self._get_default_path('RIVER_NETWORK_PATH', 'shapefiles/river_network')
        river_name = f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        river_path = river_dir / river_name

        # Fallback for legacy naming
        if not river_path.exists() and self.domain_name == 'bow_banff_minimal':
            legacy_river = river_dir / "Bow_at_Banff_lumped_riverNetwork_lumped.shp"
            if legacy_river.exists():
                river_path = legacy_river
                self.logger.info(f"Using legacy river network path: {river_path.name}")

        # Write geographic data files using manager and get land use information
        land_uses = self.geodata_manager.create_geofiles(
            gistool_output=self.gistool_output,
            subbasins_shapefile=basin_path,
            rivers_shapefile=river_path,
            frac_threshold=self.config_dict.get('HYPE_FRAC_THRESHOLD', 0.05),
            intersect_base_path=self.intersect_path
        )

        # Write parameter file using manager
        self.config_manager.write_par_file(
            params=self.calibration_params,
            land_uses=land_uses
        )

        # Get experiment dates from config
        experiment_start = self.config_dict.get('EXPERIMENT_TIME_START')
        experiment_end = self.config_dict.get('EXPERIMENT_TIME_END')

        # Write info and file directory files using manager
        # Pass forcing_data_dir so filedir.txt points to correct location
        # (important for calibration workers using isolated directories)
        self.config_manager.write_info_filedir(
            spinup_days=self.spinup_days,
            results_dir=self.hype_results_dir_str,
            experiment_start=experiment_start,
            experiment_end=experiment_end,
            forcing_data_dir=self.forcing_data_dir
        )
