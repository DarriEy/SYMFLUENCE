"""
HYPE model preprocessor.

Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
Uses the generalized pipeline pattern with manager classes for:
- Forcing data processing (HYPEForcingProcessor)
- Configuration file generation (HYPEConfigManager)
- Geographic data file generation (HYPEGeoDataManager)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import shutil

from symfluence.models.hype.forcing_processor import HYPEForcingProcessor
from symfluence.models.hype.config_manager import HYPEConfigManager
from symfluence.models.hype.geodata_manager import HYPEGeoDataManager
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import ObservationLoaderMixin
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.data.utilities.variable_utils import VariableHandler


@ModelRegistry.register_preprocessor('HYPE')
class HYPEPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    HYPE (HYdrological Predictions for the Environment) preprocessor for SYMFLUENCE.

    Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
    Inherits common functionality from BaseModelPreProcessor and observation loading
    from ObservationLoaderMixin.

    Uses the generalized pipeline pattern with manager classes:
    - HYPEForcingProcessor: Handles forcing data merging and daily conversion
    - HYPEConfigManager: Handles info.txt, filedir.txt, and par.txt generation
    - HYPEGeoDataManager: Handles GeoData.txt, GeoClass.txt, and ForcKey.txt generation

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
        self.hype_setup_dir = self.project_dir / 'settings' / 'HYPE'

        # Phase 3: Use typed config when available
        forcing_dataset = self._resolve_config_value(
            lambda: self.config.forcing.dataset,
            'FORCING_DATASET',
        )

        experiment_id = self._resolve_config_value(
            lambda: self.config.domain.experiment_id,
            'EXPERIMENT_ID'
        )
        self.hype_results_dir = self.project_dir / "simulations" / experiment_id / "HYPE"
        self.hype_results_dir.mkdir(parents=True, exist_ok=True)
        # HYPE results dir MUST have a trailing slash for the info.txt file
        self.hype_results_dir_str = str(self.hype_results_dir).rstrip('/') + '/'
        self.cache_path = self.project_dir / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize time parameters (Phase 3: typed config)
        if self.typed_config and self.typed_config.model.hype:
            self.timeshift = self._resolve_config_value(
                lambda: self.typed_config.model.hype.timeshift,
                'HYPE_TIMESHIFT',
                0
            )
            self.spinup_days = self._resolve_config_value(
                lambda: self.typed_config.model.hype.spinup_days,
                'HYPE_SPINUP_DAYS'
            )
            self.frac_threshold = self._resolve_config_value(
                lambda: self.typed_config.model.hype.frac_threshold,
                'HYPE_FRAC_THRESHOLD',
                0.1
            )
        else:
            self.timeshift = self.config_dict.get('HYPE_TIMESHIFT', 0)
            self.spinup_days = self.config_dict.get('HYPE_SPINUP_DAYS')
            self.frac_threshold = self.config_dict.get('HYPE_FRAC_THRESHOLD', 0.1)

        # If spinup_days not provided, calculate from SPINUP_PERIOD
        if self.spinup_days is None:
            spinup_period = self.config_dict.get('SPINUP_PERIOD')
            if spinup_period:
                try:
                    start_date, end_date = [pd.to_datetime(s.strip()) for s in spinup_period.split(',')]
                    self.spinup_days = (end_date - start_date).days
                    self.logger.info(f"Calculated HYPE spinup days from SPINUP_PERIOD: {self.spinup_days}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate HYPE spinup from {spinup_period}: {e}")
                    self.spinup_days = 0
            else:
                self.spinup_days = 0

        self.spinup_days = int(self.spinup_days)

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
        self.config_manager.write_info_filedir(
            spinup_days=self.spinup_days,
            results_dir=self.hype_results_dir_str,
            experiment_start=experiment_start,
            experiment_end=experiment_end
        )
