"""
Default configuration values for FUSE model.

This module registers FUSE-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('FUSE')
class FUSEDefaults:
    """FUSE model default configuration values."""

    FUSE_SPATIAL_MODE = 'lumped'
    ROUTING_MODEL = 'none'
    FUSE_INSTALL_PATH = 'default'
    SETTINGS_FUSE_PATH = 'default'
    SETTINGS_FUSE_FILEMANAGER = 'fm_catch.txt'
    FUSE_EXE = 'fuse.exe'
    EXPERIMENT_OUTPUT_FUSE = 'default'
    SETTINGS_FUSE_PARAMS_TO_CALIBRATE = 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE'
