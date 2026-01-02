"""
HYPE model preprocessor.

Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd  # type: ignore
import numpy as np
import geopandas as gpd  # type: ignore
import xarray as xr  # type: ignore
import shutil

from symfluence.utils.models.hypeFlow import (
    write_hype_forcing,
    write_hype_geo_files,
    write_hype_par_file,
    write_hype_info_filedir_files
)  # type: ignore
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import ObservationLoaderMixin
from symfluence.utils.exceptions import ModelExecutionError, symfluence_error_handler


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

    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE preprocessor with SYMFLUENCE config."""
        # Initialize base class
        super().__init__(config, logger)
        self.gistool_output = f"{str(self.project_dir / 'attributes' / 'gistool-outputs')}/"
        self.easymore_output = f"{str(self.project_dir / 'forcing' / 'easymore-outputs')}/"
        self.hype_setup_dir = f"{str(self.project_dir / 'settings' / 'HYPE')}/"
        # Phase 3: Use typed config when available
        if self.typed_config:
            experiment_id = self.typed_config.domain.experiment_id
        else:
            experiment_id = self.config.get('EXPERIMENT_ID')

        self.hype_results_dir = self.project_dir / "simulations" / experiment_id / "HYPE"
        self.hype_results_dir.mkdir(parents=True, exist_ok=True)
        self.hype_results_dir = f"{str(self.hype_results_dir)}/"
        self.cache_path = self.project_dir / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize time parameters (Phase 3: typed config)
        if self.typed_config and self.typed_config.model.hype:
            self.timeshift = self.typed_config.model.hype.timeshift
            self.spinup_days = self.typed_config.model.hype.spinup_days
            self.frac_threshold = self.typed_config.model.hype.frac_threshold
        else:
            self.timeshift = self.config.get('HYPE_TIMESHIFT')
            self.spinup_days = self.config.get('HYPE_SPINUP_DAYS')
            self.frac_threshold = self.config.get('HYPE_FRAC_THRESHOLD')

        # inputs
        self.output_path = self.hype_setup_dir

        self.forcing_units = {
            # required variable # name of var in input data, units in input data, required units for HYPE
            'temperature': {'in_varname': 'RDRS_v2.1_P_TT_09944', 'in_units': 'celsius', 'out_units': 'celsius'},
            'precipitation': {'in_varname': 'RDRS_v2.1_A_PR0_SFC', 'in_units': 'm/hr', 'out_units': 'mm/day'},
        }

        # mapping geofabric fields to model names
        self.geofabric_mapping = {
            'basinID': {'in_varname': self.config.get('RIVER_BASIN_SHP_RM_GRUID')},
            'nextDownID': {'in_varname': self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')},
            'area': {'in_varname': self.config.get('RIVER_BASIN_SHP_AREA'), 'in_units': 'm^2', 'out_units': 'm^2'},
            'rivlen': {'in_varname': self.config.get('RIVER_NETWORK_SHP_LENGTH'), 'in_units': 'm', 'out_units': 'm'}
        }

        # domain subbasins and rivers
        self.subbasins_shapefile = str(self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_delineate.shp')
        self.rivers_shapefile = str(self.project_dir / 'shapefiles' / 'river_network' / f'{self.domain_name}_riverNetwork_delineate.shp')

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
            self.easymore_output,
            self.timeshift,
            self.forcing_units,
            self.geofabric_mapping,
            self.output_path,
            f"{self.cache_path}/"
        )

    def _create_model_configs(self) -> None:
        """HYPE-specific configuration file creation (template hook)."""
        # Write geographic data files
        write_hype_geo_files(
            self.gistool_output,
            self.subbasins_shapefile,
            self.rivers_shapefile,
            self.frac_threshold,
            self.geofabric_mapping,
            self.output_path
        )

        # Write parameter file
        write_hype_par_file(self.output_path)

        # Write info and file directory files
        write_hype_info_filedir_files(self.output_path, self.spinup_days, self.hype_results_dir)
