import json
import subprocess
from pathlib import Path
from typing import Dict, Any

from symfluence.utils.data.utilities.variable_utils import VariableHandler

class DataAcquisitionProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        

    def prepare_maf_json(self) -> Path:
        """Prepare the JSON file for the Model Agnostic Framework."""

        met_path = str(self.root_path / "installs/datatool/" / "extract-dataset.sh")
        gis_path = str(self.root_path / "installs/gistool/" / "extract-gis.sh")
        easymore_client = str(self.config.get('EASYMORE_CLIENT'))

        subbasins_name = self.config.get('RIVER_BASINS_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"

        tool_cache = self.config.get('TOOL_CACHE')
        if tool_cache == 'default':
            tool_cache = '$HOME/cache_dir/'

        variables = self.config.get('FORCING_VARIABLES')
        if variables == 'default':
            variables = self.variable_handler.get_dataset_variables(dataset = self.config.get('FORCING_DATASET'))

        maf_config = {
            "exec": {
                "met": met_path,
                "gis": gis_path,
                "remap": easymore_client
            },
            "args": {
                "met": [{
                    "dataset": self.config.get('FORCING_DATASET'),
                    "dataset-dir": str(Path(self.config.get('DATATOOL_DATASET_ROOT')) / "era5/"),
                    "variable": variables,
                    "output-dir": str(self.project_dir / "forcing/datatool-outputs"),
                    "start-date": f"{self.config.get('EXPERIMENT_TIME_START')}",
                    "end-date": f"{self.config.get('EXPERIMENT_TIME_END')}",
                    "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "prefix": f"domain_{self.domain_name}_",
                    "cache": tool_cache,
                    "account": self.config.get('TOOL_ACCOUNT'),
                    "_flags": [
                        #"submit-job",
                        #"parsable"
                    ]
                }],
                "gis": [
                    {
                        "dataset": "MODIS",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MODIS"),
                        "variable": "MCD12Q1.061",
                        "start-date": "2001-01-01",
                        "end-date": "2020-01-01",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["frac", "majority", "coords"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "soil_classes"),
                        "variable": "soil_classes",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["majority"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MERIT-Hydro"),
                        "variable": "elv,hnd",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["min", "max", "mean", "median"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job",]
                    }
                ],
                "remap": [{
                    "case-name": "remapped",
                    "cache": tool_cache,
                    "shapefile": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "shapefile-id": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                    "source-nc": str(self.project_dir / "forcing/datatool-outputs/**/*.nc*"),
                    "variable-lon": "lon",
                    "variable-lat": "lat",
                    "variable": variables,
                    "remapped-var-id": "hruId",
                    "remapped-dim-id": "hru",
                    "output-dir": str(self.project_dir / "forcing/easymore-outputs/") + '/',
                    "job-conf": self.config.get('EASYMORE_JOB_CONF'),
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
        with open(json_path, 'w') as f:
            json.dump(maf_config, f, indent=2)

        self.logger.info(f"MAF configuration JSON saved to: {json_path}")
        return json_path

    def run_data_acquisition(self):
        """Run the data acquisition process using MAF."""
        json_path = self.prepare_maf_json()
        self.logger.info("Starting data acquisition process")


        maf_script = self.root_path / "installs/MAF/02_model_agnostic_component/model-agnostic.sh"
        
        #Run the MAF script
        try:
            subprocess.run([str(maf_script), str(json_path)], check=True)
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
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
