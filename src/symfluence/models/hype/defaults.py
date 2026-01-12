"""
Default configuration values for HYPE model.

This module registers HYPE-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('HYPE')
class HYPEDefaults:
    """HYPE model default configuration values."""

    SETTINGS_HYPE_PATH = 'default'
    HYPE_INSTALL_PATH = 'default'
    HYPE_EXE = 'hype'
    SETTINGS_HYPE_CONTROL_FILE = 'info.txt'
    HYPE_PARAMS_TO_CALIBRATE = 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp'
