"""
RHESSys build instructions for SYMFLUENCE.

This module defines how to build RHESSys from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria
- Optional WMFire library linking for fire spread simulation

RHESSys (Regional Hydro-Ecologic Simulation System) is an ecosystem-
hydrological model that simulates water, carbon, and nitrogen cycling.

WMFire Integration:
    If WMFire library (libwmfire.so/dylib) is found at installs/wmfire/lib,
    RHESSys will be built with fire spread support (wmfire=T). Build WMFire
    first to enable this capability. WMFire requires Boost headers.
"""

from symfluence.cli.services import (
    BuildInstructionsRegistry,
    get_bison_detection_and_build,
    get_common_build_environment,
    get_flex_detection_and_build,
    get_geos_proj_detection,
    get_netcdf_detection,
)
from symfluence.models.rhessys.build_script import RHESSYS_BUILD_COMMAND


def _build_rhessys_definition(
    common_env,
    geos_proj_detect,
    bison_detect,
    flex_detect,
    netcdf_detect,
):
    """Build the RHESSys tool definition payload."""
    return {
        'description': 'RHESSys - Regional Hydro-Ecologic Simulation System',
        'config_path_key': 'RHESSYS_INSTALL_PATH',
        'config_exe_key': 'RHESSys_EXE',
        'default_path_suffix': 'installs/rhessys/bin',
        'default_exe': 'rhessys',
        'repository': 'https://github.com/RHESSys/RHESSys.git',
        'branch': None,
        'install_dir': 'rhessys',
        'build_commands': [
            common_env,
            geos_proj_detect,
            bison_detect,
            flex_detect,
            netcdf_detect,
            RHESSYS_BUILD_COMMAND,
        ],
        'dependencies': ['gdal-config', 'proj', 'geos-config'],
        'test_command': None,  # RHESSys requires a world file; cannot test without one
        'verify_install': {
            'file_paths': ['bin/rhessys', 'bin/rhessys.exe'],
            'check_type': 'exists_any'
        },
        'order': 14
    }


@BuildInstructionsRegistry.register('rhessys')
def get_rhessys_build_instructions():
    """Get RHESSys build instructions."""
    return _build_rhessys_definition(
        common_env=get_common_build_environment(),
        geos_proj_detect=get_geos_proj_detection(),
        bison_detect=get_bison_detection_and_build(),
        flex_detect=get_flex_detection_and_build(),
        netcdf_detect=get_netcdf_detection(),
    )
