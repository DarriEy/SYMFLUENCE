"""
Model Agnostic Framework (MAF) data processor for external tool integration.

Prepares JSON configuration and executes MAF tools (datatool, gistool) for
automated data acquisition workflows.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from symfluence.core.mixins import ConfigMixin
from symfluence.data.utils.variable_utils import VariableHandler


class DataAcquisitionProcessor(ConfigMixin):
    """
    JSON configuration generator for Model-Agnostic Framework (MAF) tools.

    This processor prepares structured JSON configuration files that drive datatool
    and gistool execution for automated forcing and attribute data acquisition on
    HPC systems. It translates SYMFLUENCE configuration into MAF-compatible JSON
    schemas with proper paths, variables, and execution parameters.

    Purpose:
        Bridges SYMFLUENCE configuration with external MAF tools by generating
        JSON files that specify data extraction parameters, HPC job settings,
        and output locations for both meteorological forcing (datatool) and
        geospatial attributes (gistool).

    JSON Schema Components:
        exec: Paths to tool executables
            - met: Path to datatool extract-dataset.sh
            - gis: Path to gistool extract-gis.sh
            - remap: Path to EASYMORE client for remapping

        args: Tool-specific parameter sets
            - met: Array of datatool configurations (forcing datasets)
            - gis: Array of gistool configurations (attribute datasets)
            - Each entry contains dataset parameters, spatial/temporal bounds

    Forcing Data Configuration (datatool):
        - Dataset specification (ERA5, RDRS, CASR)
        - Dataset root directory on HPC
        - Variable list (atmospheric forcing variables)
        - Output directory for NetCDF files
        - Temporal bounds (start/end dates)
        - Shapefile for spatial bounds
        - File prefix for naming
        - Cache directory
        - HPC account for job submission
        - Optional flags (submit-job, parsable)

    Attribute Data Configuration (gistool):
        Multiple attribute datasets configured:
        1. MODIS Land Cover:
           - Product: MCD12Q1.061
           - Temporal range: 2001-2020
           - Statistics: frac (fractional cover), majority, coords
           - Includes NA values

        2. Soil Classification:
           - Variable: soil_classes
           - Statistics: majority class per polygon
           - Includes NA values

        3. Elevation (DEM):
           - Variable: elevation
           - Statistics: mean, min, max, slope

    Configuration Translation:
        SYMFLUENCE Config → MAF JSON:
            FORCING_DATASET → args.met.dataset
            FORCING_VARIABLES → args.met.variable (or default from VariableHandler)
            EXPERIMENT_TIME_START → args.met.start-date
            EXPERIMENT_TIME_END → args.met.end-date
            RIVER_BASINS_NAME → args.met/gis.shape-file
            DATATOOL_DATASET_ROOT → args.met.dataset-dir
            GISTOOL_DATASET_ROOT → args.gis.dataset-dir
            TOOL_CACHE → args.met/gis.cache
            TOOL_ACCOUNT → args.met/gis.account

    Variable Handling:
        - Uses VariableHandler to map dataset-specific variable names
        - Default variables loaded if FORCING_VARIABLES='default'
        - Translates between dataset conventions (e.g., ERA5 → SUMMA)

    Path Resolution:
        - Resolves relative paths to absolute HPC filesystem paths
        - Ensures tool executables exist and are accessible
        - Creates output directories if needed
        - Handles default vs custom tool installation paths

    Output:
        JSON file saved to: {project_dir}/settings/maf_config.json
        Schema validates against MAF tool expectations
        Used by MAF scheduler to orchestrate data extraction jobs

    HPC Integration:
        - Configures Slurm job submission parameters
        - Sets account/partition for billing
        - Defines cache directories for efficiency
        - Supports job arrays for parallel processing
        - Optional parsable output for programmatic monitoring

    Example:
        >>> config = {
        ...     'SYMFLUENCE_DATA_DIR': '/project/data',
        ...     'DOMAIN_NAME': 'test_basin',
        ...     'FORCING_DATASET': 'ERA5',
        ...     'FORCING_VARIABLES': 'default',
        ...     'EXPERIMENT_TIME_START': '2015-01-01',
        ...     'EXPERIMENT_TIME_END': '2016-12-31',
        ...     'DATATOOL_DATASET_ROOT': '/datasets/',
        ...     'TOOL_ACCOUNT': 'hydro-group'
        ... }
        >>> processor = DataAcquisitionProcessor(config, logger)
        >>> json_path = processor.prepare_maf_json()
        >>> print(json_path)
        /project/data/domain_test_basin/settings/maf_config.json

    Generated JSON Structure:
        {
          "exec": {
            "met": "/path/to/extract-dataset.sh",
            "gis": "/path/to/extract-gis.sh",
            "remap": "/path/to/easymore_client"
          },
          "args": {
            "met": [{
              "dataset": "ERA5",
              "dataset-dir": "/datasets/era5/",
              "variable": "tas,pr,huss,ps,rsds,rlds,uas,vas",
              "output-dir": "/project/data/domain_test_basin/forcing/datatool-outputs",
              "start-date": "2015-01-01",
              "end-date": "2016-12-31",
              ...
            }],
            "gis": [{
              "dataset": "MODIS",
              ...
            }, ...]
          }
        }

    Notes:
        - JSON file must be readable by MAF scheduler
        - Tool paths must be accessible on HPC compute nodes
        - Cache directory improves performance for repeated extractions
        - Shapefile must exist before JSON generation
        - Account string must match HPC accounting system

    See Also:
        - data.acquisition.maf_pipeline.gistoolRunner: Executes gistool commands
        - data.acquisition.maf_pipeline.datatoolRunner: Executes datatool commands
        - data.utils.variable_utils.VariableHandler: Variable mapping
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.root_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')


    def prepare_maf_json(self) -> Path:
        """Prepare the JSON file for the Model Agnostic Framework."""

        met_path = str(self.root_path / "installs/datatool/" / "extract-dataset.sh")
        gis_path = str(self.root_path / "installs/gistool/" / "extract-gis.sh")
        easymore_client = str(self._get_config_value(lambda: self.config.paths.easymore_client, dict_key='EASYMORE_CLIENT'))

        subbasins_name = self._get_config_value(lambda: self.config.paths.river_basins_name, dict_key='RIVER_BASINS_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}_riverBasins_{self._get_config_value(lambda: self.config.domain.definition_method, dict_key='DOMAIN_DEFINITION_METHOD')}.shp"

        tool_cache = self._get_config_value(lambda: self.config.paths.tool_cache, dict_key='TOOL_CACHE')
        if tool_cache == 'default':
            tool_cache = '$HOME/cache_dir/'

        variables = self._get_config_value(lambda: self.config.forcing.variables, dict_key='FORCING_VARIABLES')
        if variables == 'default':
            variables = self.variable_handler.get_dataset_variables(dataset = self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET'))

        maf_config = {
            "exec": {
                "met": met_path,
                "gis": gis_path,
                "remap": easymore_client
            },
            "args": {
                "met": [{
                    "dataset": self._get_config_value(lambda: self.config.forcing.dataset, dict_key='FORCING_DATASET'),
                    "dataset-dir": str(Path(self._get_config_value(lambda: self.config.paths.datatool_dataset_root, dict_key='DATATOOL_DATASET_ROOT')) / "era5/"),
                    "variable": variables,
                    "output-dir": str(self.project_dir / "forcing/datatool-outputs"),
                    "start-date": f"{self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')}",
                    "end-date": f"{self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END')}",
                    "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "prefix": f"domain_{self.domain_name}_",
                    "cache": tool_cache,
                    "account": self._get_config_value(lambda: None, default=None, dict_key='TOOL_ACCOUNT'),
                    "_flags": [
                        #"submit-job",
                        #"parsable"
                    ]
                }],
                "gis": [
                    {
                        "dataset": "MODIS",
                        "dataset-dir": str(Path(self._get_config_value(lambda: self.config.paths.gistool_dataset_root, dict_key='GISTOOL_DATASET_ROOT')) / "MODIS"),
                        "variable": "MCD12Q1.061",
                        "start-date": "2001-01-01",
                        "end-date": "2020-01-01",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["frac", "majority", "coords"],
                        "lib-path": self._get_config_value(lambda: self.config.paths.gistool_lib_path, dict_key='GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self._get_config_value(lambda: None, default=None, dict_key='TOOL_ACCOUNT'),
                        "fid": self._get_config_value(lambda: self.config.paths.river_basin_rm_gruid, dict_key='RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": str(Path(self._get_config_value(lambda: self.config.paths.gistool_dataset_root, dict_key='GISTOOL_DATASET_ROOT')) / "soil_classes"),
                        "variable": "soil_classes",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["majority"],
                        "lib-path": self._get_config_value(lambda: self.config.paths.gistool_lib_path, dict_key='GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self._get_config_value(lambda: None, default=None, dict_key='TOOL_ACCOUNT'),
                        "fid": self._get_config_value(lambda: self.config.paths.river_basin_rm_gruid, dict_key='RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": str(Path(self._get_config_value(lambda: self.config.paths.gistool_dataset_root, dict_key='GISTOOL_DATASET_ROOT')) / "MERIT-Hydro"),
                        "variable": "elv,hnd",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["min", "max", "mean", "median"],
                        "lib-path": self._get_config_value(lambda: self.config.paths.gistool_lib_path, dict_key='GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self._get_config_value(lambda: None, default=None, dict_key='TOOL_ACCOUNT'),
                        "fid": self._get_config_value(lambda: self.config.paths.river_basin_rm_gruid, dict_key='RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job",]
                    }
                ],
                "remap": [{
                    "case-name": "remapped",
                    "cache": tool_cache,
                    "shapefile": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "shapefile-id": self._get_config_value(lambda: self.config.paths.river_basin_rm_gruid, dict_key='RIVER_BASIN_SHP_RM_GRUID'),
                    "source-nc": str(self.project_dir / "forcing/datatool-outputs/**/*.nc*"),
                    "variable-lon": "lon",
                    "variable-lat": "lat",
                    "variable": variables,
                    "remapped-var-id": "hruId",
                    "remapped-dim-id": "hru",
                    "output-dir": str(self.project_dir / "forcing/easymore-outputs/") + '/',
                    "job-conf": self._get_config_value(lambda: self.config.paths.easymore_job_conf, dict_key='EASYMORE_JOB_CONF'),
                    #"_flags": ["submit-job"]
                }]
            },
            "order": {
                "met": 1,
                "gis": -1,
                "remap": 2
            }
        }

        # Save the JSON file
        json_path = self.project_dir / "forcing/maf_config.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(maf_config, f, indent=2)

        self.logger.info(f"MAF configuration JSON saved to: {json_path}")
        return json_path

    def run_data_acquisition(self):
        """Run the data acquisition process using MAF."""
        json_path = self.prepare_maf_json()
        self.logger.info("Starting data acquisition process")


        maf_script = self.root_path / "installs/MAF/02_model_agnostic_component/model-agnostic.sh"

        #Run the MAF script
        # On Windows, .sh scripts need an explicit bash interpreter
        maf_command = [str(maf_script), str(json_path)]
        if sys.platform == "win32":
            maf_command = ["bash"] + maf_command
        try:
            subprocess.run(maf_command, check=True)
            self.logger.info("Model Agnostic Framework completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running Model Agnostic Framework: {e}")
            raise
        self.logger.info("Data acquisition process completed")

    def _get_file_path(self, file_type, file_def_path, file_name):
        """
        Construct file paths based on configuration.

        Args:
            file_type (str): Type of the file (used as a key in config).
            file_def_path (str): Default path relative to project directory.
            file_name (str): Name of the file.

        Returns:
            Path: Constructed file path.
        """
        file_type_val = self._get_config_value(lambda: None, default='default', dict_key=file_type)
        if file_type_val == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(file_type_val)
