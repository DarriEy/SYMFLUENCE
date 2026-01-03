
import logging
import sys
from pathlib import Path
from symfluence.utils.models.mesh.preprocessor import MESHPreProcessor

# Mock config
config = {
    'DOMAIN_NAME': 'test_domain',
    'SYMFLUENCE_DATA_DIR': './data',
    'RIVER_BASINS_PATH': 'default',
    'DOMAIN_DEFINITION_METHOD': 'delineate',
    'MESH_FORCING_VARS': 'default',
    'MESH_FORCING_UNITS': {},
    'MESH_FORCING_TO_UNITS': {},
    'MESH_LANDCOVER_STATS_PATH': None,
    'MESH_LANDCOVER_STATS_FILE': 'stats.csv',
    'MESH_LANDCOVER_STATS_DIR': './attributes',
    'MESH_FORCING_PATH': './forcing',
    'MESH_MAIN_ID': 'GRU_ID',
    'MESH_DS_MAIN_ID': 'DSLINKNO',
    'MESH_LANDCOVER_CLASSES': 'default',
    'MESH_DDB_VARS': 'default',
    'MESH_DDB_UNITS': 'default',
    'MESH_DDB_TO_UNITS': 'default',
    'MESH_DDB_MIN_VALUES': 'default',
    'MESH_GRU_DIM': 'NGRU',
    'MESH_HRU_DIM': 'subbasin',
    'MESH_OUTLET_VALUE': 0,
}

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MESH_TEST")

try:
    preprocessor = MESHPreProcessor(config, logger)
    print("MESHPreProcessor initialized successfully.")
    preprocessor.run_preprocessing()
except Exception as e:
    print(f"Error: {e}")
