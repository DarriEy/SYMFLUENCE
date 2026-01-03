"""
HYPE model preprocessor.

Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd  # type: ignore
import numpy as np
import geopandas as gpd  # type: ignore
import xarray as xr  # type: ignore
import shutil

from symfluence.utils.models.hype.hypeFlow import (
    write_hype_forcing,
    write_hype_geo_files,
    write_hype_par_file,
    write_hype_info_filedir_files
)  # type: ignore
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import ObservationLoaderMixin
from symfluence.utils.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.utils.data.utilities.variable_utils import VariableHandler

@ModelRegistry.register_preprocessor('HYPE')
class HYPEPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    HYPE (HYdrological Predictions for the Environment) preprocessor for SYMFLUENCE.

    Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
    Inherits common functionality from BaseModelPreProcessor and observation loading from ObservationLoaderMixin.

    Attributes:
        config: SYMFLUENCE configuration settings (inherited)
        logger: Logger for the preprocessing workflow (inherited)
        project_dir: Project directory path (inherited)
        domain_name: Name of the modeling domain (inherited)
        setup_dir: HYPE setup directory (inherited as model-specific)
    """

    def _get_model_name(self) -> str:
        """Return model name for HYPE."""
        return "HYPE"

    def __init__(self, config: Dict[str, Any], logger: Any, params: Optional[Dict[str, Any]] = None):
        """Initialize HYPE preprocessor with SYMFLUENCE config."""
        # Initialize base class
        super().__init__(config, logger)
        self.calibration_params = params
        self.gistool_output = f"{str(self.project_dir / 'attributes' / 'gistool-outputs')}/"
        # HYPE needs the remapped forcing data and geospatial statistics
        self.forcing_input_dir = self.forcing_basin_path
        self.hype_setup_dir = f"{str(self.project_dir / 'settings' / 'HYPE')}/"
        # Phase 3: Use typed config when available
        if self.config:
            experiment_id = self.config.domain.experiment_id
            forcing_dataset = self.config.domain.forcing_dataset
        else:
            experiment_id = self.config_dict.get('EXPERIMENT_ID')
            forcing_dataset = self.config_dict.get('FORCING_DATASET')

        self.hype_results_dir = self.project_dir / "simulations" / experiment_id / "HYPE"
        self.hype_results_dir.mkdir(parents=True, exist_ok=True)
        self.hype_results_dir = f"{str(self.hype_results_dir)}/"
        self.cache_path = self.project_dir / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize time parameters (Phase 3: typed config)
        if self.typed_config and self.config.model.hype:
            self.timeshift = self.config.model.hype.timeshift
            self.spinup_days = self.config.model.hype.spinup_days
            self.frac_threshold = self.config.model.hype.frac_threshold
        else:
            self.timeshift = self.config_dict.get('HYPE_TIMESHIFT', 0)
            # Use 0 as default spinup (was 365, but 0 is more sensible for most cases)
            spinup_val = self.config_dict.get('HYPE_SPINUP_DAYS')
            self.spinup_days = spinup_val if spinup_val is not None else 0
            self.frac_threshold = self.config_dict.get('HYPE_FRAC_THRESHOLD', 0.1)

        # inputs
        self.output_path = self.hype_setup_dir

        # Initialize variable handler to get correct input names
        var_handler = VariableHandler(self.config_dict, self.logger, forcing_dataset, 'HYPE')
        dataset_map = var_handler.DATASET_MAPPINGS[forcing_dataset]
        
        # Get input names for temperature and precipitation
        temp_in = var_handler._find_matching_variable('air_temperature', dataset_map)
        precip_in = var_handler._find_matching_variable('precipitation_flux', dataset_map)

        self.forcing_units = {
            'temperature': {
                'in_varname': temp_in, 
                'in_units': dataset_map[temp_in]['units'], 
                'out_units': 'degC'
            },
            'precipitation': {
                'in_varname': precip_in, 
                'in_units': dataset_map[precip_in]['units'], 
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
        delim_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'delineate')
        self.subbasins_shapefile = str(self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_{delim_method}.shp')
        
        # River network file might not always exist for lumped domains, fallback to river_basins if needed
        network_file = self.project_dir / 'shapefiles' / 'river_network' / f'{self.domain_name}_riverNetwork_{delim_method}.shp'
        if not network_file.exists():
            # If no network file, try generic or fallback
            network_file = self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_{delim_method}.shp'
            
        self.rivers_shapefile = str(network_file)

    def run_preprocessing(self):
        """
        Execute complete HYPE preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info("Starting HYPE preprocessing")
        return self.run_preprocessing_template()

    def _prepare_forcing(self) -> None:
        """HYPE-specific forcing data preparation (template hook)."""
        write_hype_forcing(
            self.forcing_input_dir,
            self.timeshift,
            self.forcing_units,
            self.geofabric_mapping,
            self.output_path,
            f"{self.cache_path}/"
        )

    def _create_model_configs(self) -> None:
        """HYPE-specific configuration file creation (template hook)."""
        # Write geographic data files and get land use information
        land_uses = write_hype_geo_files(
            self.gistool_output,
            self.subbasins_shapefile,
            self.rivers_shapefile,
            self.frac_threshold,
            self.geofabric_mapping,
            self.output_path,
            self.intersect_path
        )

        # Write parameter file with dynamic land use parameters
        write_hype_par_file(self.output_path, params=self.calibration_params, land_uses=land_uses)

        # Get experiment dates from config
        experiment_start = self.config_dict.get('EXPERIMENT_TIME_START')
        experiment_end = self.config_dict.get('EXPERIMENT_TIME_END')

        # Write info and file directory files
        write_hype_info_filedir_files(
            self.output_path,
            self.spinup_days,
            self.hype_results_dir,
            experiment_start=experiment_start,
            experiment_end=experiment_end
        )
