"""
FUSE initialization presets.

This module registers FUSE-specific presets with the PresetRegistry,
keeping model-specific configuration within the model directory.
"""

from symfluence.cli.preset_registry import PresetRegistry


@PresetRegistry.register_preset('fuse-provo')
def fuse_provo_preset():
    """FUSE model for Provo River, Utah with ERA5 forcing."""
    return {
        'description': 'FUSE model for Provo River, Utah with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'DOMAIN_NAME': 'provo_river',
            'EXPERIMENT_ID': 'run_1',
            'EXPERIMENT_TIME_START': '1999-10-01 00:00',
            'EXPERIMENT_TIME_END': '2002-09-30 23:00',
            'CALIBRATION_PERIOD': '2000-10-01, 2001-09-30',
            'EVALUATION_PERIOD': '2001-10-01, 2002-09-30',
            'SPINUP_PERIOD': '1999-10-01, 2000-09-30',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'POUR_POINT_COORDS': '40.5577278126503/-111.168783841314',
            'BOUNDING_BOX_COORDS': '41/-111.74291666666666/40.0/-110.6208333333333',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'ROUTING_DELINEATION': 'lumped',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 2500,
            'LUMPED_WATERSHED_METHOD': 'TauDEM',

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FUSE_SPATIAL_MODE': 'lumped',
            'FUSE_EXE': 'fuse.exe',
            'FUSE_INSTALL_PATH': 'default',
            'SETTINGS_FUSE_PATH': 'default',
            'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
            'EXPERIMENT_OUTPUT_FUSE': 'default',
            'ROUTING_MODEL': 'none',

            # FUSE calibration parameters
            'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',

            # Evaluation settings
            'STREAMFLOW_DATA_PROVIDER': 'USGS',
            'DOWNLOAD_USGS_DATA': True,
            'STATION_ID': '10154200',
            'SIM_REACH_ID': 1,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'fuse_decisions': {
            'RFERR': ['multiplc_e'],
            'ARCH1': ['tension1_1'],
            'ARCH2': ['fixedsiz_2'],
            'QSURF': ['arno_x_vic'],
            'QPERC': ['perc_w2sat'],
            'ESOIL': ['rootweight'],
            'QINTF': ['intflwnone'],
            'Q_TDH': ['rout_gamma'],
            'SNOWM': ['temp_index']
        }
    }


@PresetRegistry.register_preset('fuse-basic')
def fuse_basic_preset():
    """Generic FUSE lumped setup with ERA5 forcing."""
    return {
        'description': 'Generic FUSE lumped setup with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'EXPERIMENT_ID': 'run_1',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'ROUTING_DELINEATION': 'lumped',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 1000,
            'LUMPED_WATERSHED_METHOD': 'TauDEM',

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'FUSE',
            'FUSE_SPATIAL_MODE': 'lumped',
            'FUSE_EXE': 'fuse.exe',
            'FUSE_INSTALL_PATH': 'default',
            'SETTINGS_FUSE_PATH': 'default',
            'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
            'EXPERIMENT_OUTPUT_FUSE': 'default',
            'ROUTING_MODEL': 'none',

            # FUSE calibration parameters
            'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1',

            # Evaluation settings
            'DOWNLOAD_USGS_DATA': True,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'fuse_decisions': {
            'RFERR': ['multiplc_e'],
            'ARCH1': ['tension1_1'],
            'ARCH2': ['fixedsiz_2'],
            'QSURF': ['arno_x_vic'],
            'QPERC': ['perc_w2sat'],
            'ESOIL': ['rootweight'],
            'QINTF': ['intflwnone'],
            'Q_TDH': ['rout_gamma'],
            'SNOWM': ['temp_index']
        }
    }
