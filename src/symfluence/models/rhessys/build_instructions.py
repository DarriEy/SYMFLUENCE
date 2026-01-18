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
    get_flex_detection_and_build,
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
    flex_detect = get_flex_detection_and_build()

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

# Fix assign_base_station_xy.c - station->lon/lat don't exist in base_station_object struct.
# The is_approximately() function is defined in util/alloc.c under #ifdef LIU_NETCDF_READER,
# which is enabled via the makefile's DEFINES. We just need to comment out the broken lon/lat condition.
echo "Patching assign_base_station_xy.c for missing struct members..."
cat > /tmp/rhessys_xy_patch.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my @lines = <$fh>;
close($fh);

my $content = join('', @lines);

# Comment out the broken lon/lat condition line
# The pattern is: (is_approximately(x,station->lon, ... station->lat, ...)))
$content =~ s/\(is_approximately\(x,station->lon,[^)]+\)\s*&&\s*is_approximately\(y,station->lat,[^)]+\)\)/1 \/* lon\/lat check disabled - members dont exist *\//g;

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file\n";
PERLSCRIPT
perl /tmp/rhessys_xy_patch.pl init/assign_base_station_xy.c

# Fix construct_netcdf_grid.c - use perl for reliable pattern replacement
echo "Patching construct_netcdf_grid.c for missing struct members..."
cat > /tmp/rhessys_grid_patch.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Replace base_station[...].x with base_station[...].proj_x
$changes += ($content =~ s/(\bbase_station\s*\[[^\]]+\])\s*\.\s*x\b/$1.proj_x/g);
# Replace base_station[...].y with base_station[...].proj_y
$changes += ($content =~ s/(\bbase_station\s*\[[^\]]+\])\s*\.\s*y\b/$1.proj_y/g);
# Replace base_station[...].lat with base_station[...].proj_y
$changes += ($content =~ s/(\bbase_station\s*\[[^\]]+\])\s*\.\s*lat\b/$1.proj_y/g);
# Replace base_station[...].lon with base_station[...].proj_x
$changes += ($content =~ s/(\bbase_station\s*\[[^\]]+\])\s*\.\s*lon\b/$1.proj_x/g);
# Fix proj_yearly_clim which should be yearly_clim
$changes += ($content =~ s/\.proj_yearly_clim/.yearly_clim/g);

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes replacements\n";
PERLSCRIPT
perl /tmp/rhessys_grid_patch.pl init/construct_netcdf_grid.c
echo "Patched construct_netcdf_grid.c"

# Fix construct_netcdf_header.c - use perl for reliable pattern replacement
echo "Patching construct_netcdf_header.c for missing struct members..."
cat > /tmp/rhessys_header_patch.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Replace .lon with .proj_x (any context with basestations)
$changes += ($content =~ s/(\bbasestations\s*\[[^\]]+\](?:\s*\[[^\]]+\])?)\s*\.\s*lon/$1.proj_x/g);
$changes += ($content =~ s/(\bbasestations\s*\[[^\]]+\](?:\s*\[[^\]]+\])?)\s*->\s*lon/$1->proj_x/g);

# Replace .lat with .proj_y (any context with basestations)
$changes += ($content =~ s/(\bbasestations\s*\[[^\]]+\](?:\s*\[[^\]]+\])?)\s*\.\s*lat/$1.proj_y/g);
$changes += ($content =~ s/(\bbasestations\s*\[[^\]]+\](?:\s*\[[^\]]+\])?)\s*->\s*lat/$1->proj_y/g);

# Remove const from calc_resolution signature to fix pointer type mismatch
$changes += ($content =~ s/const\s+struct\s+base_station_object\s*\*\*\s*basestations/struct base_station_object **basestations/g);

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes replacements\n";

# Verify no .lon or .lat remain with basestations
if ($content =~ /basestations.*\.(lon|lat)/) {
    print "WARNING: Some basestations.lon/lat patterns may remain\n";
}
PERLSCRIPT
perl /tmp/rhessys_header_patch.pl init/construct_netcdf_header.c

# Verify the patch worked - should NOT find .lon or .lat with basestations
if grep -q 'basestations\[.*\]\.lon\|basestations\[.*\]\.lat' init/construct_netcdf_header.c 2>/dev/null; then
    echo "ERROR: construct_netcdf_header.c patch failed - .lon/.lat still present"
    grep -n '\.lon\|\.lat' init/construct_netcdf_header.c | head -5
    exit 1
fi
echo "Patched construct_netcdf_header.c successfully"

# Note: Forward declarations for get_netcdf_var/get_netcdf_var_timeserias/get_indays are in rhessys.h
# under #ifdef LIU_NETCDF_READER which is defined by the makefile with netcdf=T

# Verify patches
grep "const void \*e1" util/key_compare.c || { echo "Patching key_compare.c failed"; exit 1; }
grep "const void \*" util/sort_patch_layers.c || { echo "Patching sort_patch_layers.c failed"; exit 1; }
echo "Patches verified."

# Patch makefile to remove test dependency from rhessys target
# This allows building without glib-2.0 which is required only for tests
echo "Patching makefile to skip test dependency..."
sed -i.bak 's/^rhessys: \$(OBJECTS) test$/rhessys: $(OBJECTS)/' makefile
grep -q "^rhessys: \$(OBJECTS)$" makefile && echo "Makefile patched successfully" || echo "Warning: Makefile patch may not have applied"

# Build with detected flags - explicitly pass CC to override Makefile's clang default
# Add -Wno-error flags for GCC 14 compatibility (newer GCC treats some warnings as errors)
# Note: -DCLIM_GRID_XY is disabled because the RHESSys source has broken code paths
# that reference non-existent struct members (x, y, lat, lon in base_station_object)
# IMPORTANT: Use CMD_OPTS for extra flags, NOT CFLAGS override - the makefile's CFLAGS
# includes $(DEFINES) with -DLIU_NETCDF_READER which is required for is_approximately()
COMPAT_FLAGS="-Wno-error=incompatible-pointer-types -Wno-error=int-conversion -Wno-error=implicit-function-declaration"
make V=1 CC="$CC" netcdf=T CMD_OPTS="$COMPAT_FLAGS $GEOS_CFLAGS $PROJ_CFLAGS $GEOS_LDFLAGS $PROJ_LDFLAGS"

mkdir -p ../bin
# Try multiple possible locations for rhessys binary
if [ -f rhessys/rhessys ]; then
    mv rhessys/rhessys ../bin/rhessys
elif [ -f rhessys ]; then
    mv rhessys ../bin/rhessys
else
    # Pattern match for rhessys*
    mv rhessys* ../bin/rhessys 2>/dev/null || true
fi

# Verify binary exists
if [ ! -f ../bin/rhessys ]; then
    echo "ERROR: rhessys binary not found after build"
    exit 1
fi

chmod +x ../bin/rhessys
echo "RHESSys binary successfully built at ../bin/rhessys"
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
