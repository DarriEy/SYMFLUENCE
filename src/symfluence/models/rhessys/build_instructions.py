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

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_geos_proj_detection,
    get_bison_detection_and_build,
    get_flex_detection_and_build,
    get_netcdf_detection,
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
    netcdf_detect = get_netcdf_detection()

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
            r'''
set -e
echo "Building RHESSys..."
cd rhessys

# Detect netcdf library location for linking
NETCDF_LDFLAGS=""

# Check HPC environment variables first (EasyBuild module system)
if [ -n "${EBROOTNETCDF:-}" ]; then
    for libdir in "$EBROOTNETCDF/lib64" "$EBROOTNETCDF/lib"; do
        if [ -f "$libdir/libnetcdf.so" ] || [ -f "$libdir/libnetcdf.a" ]; then
            NETCDF_LDFLAGS="-L$libdir -lnetcdf"
            echo "Found HPC module netcdf library in: $libdir"
            break
        fi
    done
fi

# Try nc-config if available and NETCDF_LDFLAGS not yet set
if [ -z "$NETCDF_LDFLAGS" ] && command -v nc-config >/dev/null 2>&1; then
    NETCDF_LDFLAGS="$(nc-config --libs 2>/dev/null || echo "")"
    if [ -n "$NETCDF_LDFLAGS" ]; then
        echo "Using nc-config libs: $NETCDF_LDFLAGS"
    fi
fi

# Fallback to manual detection if nc-config didn't work
if [ -z "$NETCDF_LDFLAGS" ] && [ -n "$NETCDF_C" ]; then
    for libdir in "$NETCDF_C/lib64" "$NETCDF_C/lib" "$CONDA_PREFIX/lib"; do
        if [ -n "$libdir" ] && ([ -f "$libdir/libnetcdf.so" ] || [ -f "$libdir/libnetcdf.a" ] || [ -f "$libdir/libnetcdf.dylib" ]); then
            NETCDF_LDFLAGS="-L$libdir -lnetcdf"
            echo "Found netcdf library in: $libdir"
            break
        fi
    done
fi

# Final fallback - just try -lnetcdf
if [ -z "$NETCDF_LDFLAGS" ]; then
    NETCDF_LDFLAGS="-lnetcdf"
    echo "WARNING: NetCDF library path not found. Using default: $NETCDF_LDFLAGS"
    echo "On HPC systems, ensure you have loaded the netcdf module:"
    echo "  module load netcdf  # or netcdf-c"
fi
echo "NETCDF_LDFLAGS: $NETCDF_LDFLAGS"

# Set up flex library flags if flex was detected/built
FLEX_LDFLAGS=""

# Check HPC module environment variable first (EasyBuild)
if [ -n "${EBROOTFLEX:-}" ]; then
    for libdir in "$EBROOTFLEX/lib64" "$EBROOTFLEX/lib"; do
        if [ -f "$libdir/libfl.so" ] || [ -f "$libdir/libfl.a" ]; then
            FLEX_LDFLAGS="-L$libdir -lfl"
            echo "Found HPC module flex library in: $libdir"
            break
        fi
    done
fi

# Use FLEX_LIB_DIR if set by flex_detect snippet
if [ -z "$FLEX_LDFLAGS" ] && [ -n "${FLEX_LIB_DIR:-}" ] && [ -d "${FLEX_LIB_DIR}" ]; then
    if [ -f "${FLEX_LIB_DIR}/libfl.a" ] || [ -f "${FLEX_LIB_DIR}/libfl.so" ]; then
        FLEX_LDFLAGS="-L${FLEX_LIB_DIR} -lfl"
        echo "FLEX_LDFLAGS: $FLEX_LDFLAGS (from FLEX_LIB_DIR)"
    fi
fi

# Check if flex was built locally in the install directory (one level up from rhessys source)
if [ -z "$FLEX_LDFLAGS" ]; then
    for flexlib in "../flex/lib" "../../flex/lib"; do
        if [ -f "$flexlib/libfl.a" ] || [ -f "$flexlib/libfl.so" ]; then
            # Convert to absolute path
            FLEX_LIB_ABS=$(cd "$flexlib" && pwd)
            FLEX_LDFLAGS="-L${FLEX_LIB_ABS} -lfl"
            echo "Found locally-built flex library at: $FLEX_LIB_ABS"
            export LIBRARY_PATH="${FLEX_LIB_ABS}:${LIBRARY_PATH:-}"
            export LD_LIBRARY_PATH="${FLEX_LIB_ABS}:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

# Search common system paths for libfl
if [ -z "$FLEX_LDFLAGS" ]; then
    for libdir in /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /lib64 /lib; do
        if [ -f "$libdir/libfl.a" ] || [ -f "$libdir/libfl.so" ]; then
            FLEX_LDFLAGS="-L$libdir -lfl"
            echo "Found system flex library in: $libdir"
            break
        fi
    done
fi

# Final fallback - just try -lfl (might work if it's in standard paths)
if [ -z "$FLEX_LDFLAGS" ]; then
    FLEX_LDFLAGS="-lfl"
    echo "WARNING: libfl not found in expected locations. Trying default -lfl"
    echo "If linking fails, ensure flex-devel is installed or load the flex module"
fi

echo "FLEX_LDFLAGS: $FLEX_LDFLAGS"

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

# Check for WMFire library and enable fire spread support if available
WMFIRE_FLAG=""
WMFIRE_LDFLAGS=""
mkdir -p ../lib

# Look for WMFire library in common locations
OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    WMFIRE_LIB_NAME="libwmfire.dylib"
else
    WMFIRE_LIB_NAME="libwmfire.so"
fi

# Check if WMFire was already built in the wmfire install directory
WMFIRE_INSTALL_DIR="${INSTALLS_DIR:-../..}/wmfire/lib"
if [ -f "$WMFIRE_INSTALL_DIR/$WMFIRE_LIB_NAME" ]; then
    echo "Found WMFire library at $WMFIRE_INSTALL_DIR/$WMFIRE_LIB_NAME"
    cp "$WMFIRE_INSTALL_DIR/$WMFIRE_LIB_NAME" ../lib/
    WMFIRE_FLAG="wmfire=T"
    WMFIRE_LDFLAGS="-L../lib"
    echo "WMFire support ENABLED"
fi

# Also check if WMFire is in the FIRE directory (build from source in same repo)
if [ -z "$WMFIRE_FLAG" ] && [ -d "../FIRE" ]; then
    if [ -f "../FIRE/$WMFIRE_LIB_NAME" ]; then
        echo "Found WMFire library in FIRE directory"
        cp "../FIRE/$WMFIRE_LIB_NAME" ../lib/
        WMFIRE_FLAG="wmfire=T"
        WMFIRE_LDFLAGS="-L../lib"
        echo "WMFire support ENABLED (from FIRE directory)"
    elif [ -f "../FIRE/WMFire.cpp" ]; then
        echo "WMFire source found but not built. Building WMFire..."
        # Build WMFire from source
        pushd ../FIRE > /dev/null
        CXX=${CXX:-g++}

        # Find Boost headers
        BOOST_INCLUDE=""
        for boost_dir in "$CONDA_PREFIX/include" /opt/homebrew/include /usr/local/include /usr/include; do
            if [ -d "$boost_dir/boost" ]; then
                BOOST_INCLUDE="-I$boost_dir"
                break
            fi
        done

        if [ "$OS" = "Darwin" ]; then
            SHARED_FLAG="-dynamiclib"
        else
            SHARED_FLAG="-shared"
        fi

        $CXX -c -fPIC $BOOST_INCLUDE -O2 -o RanNums.o RanNums.cpp 2>/dev/null || echo "WMFire build requires Boost headers"
        $CXX -c -fPIC $BOOST_INCLUDE -O2 -o WMFire.o WMFire.cpp 2>/dev/null || true
        if [ -f "RanNums.o" ] && [ -f "WMFire.o" ]; then
            $CXX $SHARED_FLAG -fPIC -o $WMFIRE_LIB_NAME RanNums.o WMFire.o
            if [ -f "$WMFIRE_LIB_NAME" ]; then
                # From ../FIRE, ../lib is at the same level as rhessys source dir
                mv $WMFIRE_LIB_NAME ../lib/
                WMFIRE_FLAG="wmfire=T"
                WMFIRE_LDFLAGS="-L../lib"
                echo "WMFire built and enabled"
            fi
            rm -f RanNums.o WMFire.o
        fi
        popd > /dev/null
    fi
fi

if [ -z "$WMFIRE_FLAG" ]; then
    echo "WMFire library not found - building RHESSys without fire spread support"
    echo "To enable WMFire: install Boost headers and build WMFire first"
fi

# Build RHESSys with optional WMFire support
make V=1 CC="$CC" netcdf=T $WMFIRE_FLAG CMD_OPTS="$COMPAT_FLAGS $GEOS_CFLAGS $PROJ_CFLAGS $GEOS_LDFLAGS $PROJ_LDFLAGS $NETCDF_LDFLAGS $FLEX_LDFLAGS $WMFIRE_LDFLAGS"

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
