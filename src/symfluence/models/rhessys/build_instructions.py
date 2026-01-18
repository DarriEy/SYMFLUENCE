"""
RHESSys build instructions for SYMFLUENCE.

This module defines how to build RHESSys from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

RHESSys (Regional Hydro-Ecologic Simulation System) is an ecosystem-
hydrological model that simulates water, carbon, and nitrogen cycling.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_geos_proj_detection,
    get_bison_detection_and_build,
)


@BuildInstructionsRegistry.register('rhessys')
def get_rhessys_build_instructions():
    """
    Get RHESSys build instructions.

    RHESSys requires GEOS and PROJ libraries for geospatial operations.
    The build uses make and includes patches for modern compiler compatibility.

    Returns:
        Dictionary with complete build configuration for RHESSys.
    """
    common_env = get_common_build_environment()
    geos_proj_detect = get_geos_proj_detection()
    bison_detect = get_bison_detection_and_build()

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
            r'''
set -e
echo "Building RHESSys..."
cd rhessys

echo "GEOS_CFLAGS: $GEOS_CFLAGS, PROJ_CFLAGS: $PROJ_CFLAGS"

# Force gcc compiler (RHESSys Makefile defaults to clang which may not be available)
# Use system gcc if CC not already set
if [ -z "$CC" ]; then
    if [ -x /usr/bin/gcc ]; then
        export CC=/usr/bin/gcc
    else
        export CC=$(command -v gcc || echo gcc)
    fi
fi
echo "Using C compiler: $CC"

# Apply patches for compiler compatibility
perl -i.bak -pe 's/int\s+key_compare\s*\(\s*void\s*\*\s*e1\s*,\s*void\s*\*\s*e2\s*\)/int key_compare(const void *e1, const void *e2)/' util/key_compare.c
perl -i.bak -pe 's/int\s+key_compare\s*\(\s*void\s*\*\s*,\s*void\s*\*\s*\)\s*;/int key_compare(const void *, const void *);/' util/sort_patch_layers.c
perl -i.bak -pe 's/(\s*)sort_patch_layers\(patch, \*rec\);/sort_patch_layers(patch, rec);/' util/sort_patch_layers.c
perl -i.bak -pe 's/^\s*#define MAXSTR 200/\/\/#define MAXSTR 200/' include/rhessys_fire.h
perl -i.bak -pe 's/(^)/#include \"rhessys.h\"\n#include <math.h>\n$1/' init/assign_base_station_xy.c
perl -i.bak -pe 's/(#include <math.h>)/$1\n#define is_approximately(a, b, epsilon) (fabs((a) - (b)) < (epsilon))/' init/assign_base_station_xy.c

# Verify patches
grep "const void \*e1" util/key_compare.c || { echo "Patching key_compare.c failed"; exit 1; }
grep "const void \*" util/sort_patch_layers.c || { echo "Patching sort_patch_layers.c failed"; exit 1; }
echo "Patches verified."

# Build with detected flags - explicitly pass CC to override Makefile's clang default
make V=1 CC="$CC" netcdf=T CMD_OPTS="-DCLIM_GRID_XY $GEOS_CFLAGS $PROJ_CFLAGS $GEOS_LDFLAGS $PROJ_LDFLAGS"

mkdir -p ../bin
mv rhessys* ../bin/rhessys || true
chmod +x ../bin/rhessys
            '''.strip()
        ],
        'dependencies': ['gdal-config', 'proj', 'geos-config'],
        'test_command': '-h',
        'verify_install': {
            'file_paths': ['bin/rhessys'],
            'check_type': 'exists'
        },
        'order': 14
    }
