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
    clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    for libdir in "$NETCDF_C/lib64" "$NETCDF_C/lib" "$clp/lib"; do
        if [ -n "$libdir" ] && ([ -f "$libdir/libnetcdf.so" ] || [ -f "$libdir/libnetcdf.a" ] || [ -f "$libdir/libnetcdf.dylib" ] || [ -f "$libdir/libnetcdf.dll.a" ] || [ -f "$libdir/netcdf.lib" ]); then
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
        if [ -f "$libdir/libfl.so" ] || [ -f "$libdir/libfl.a" ] || [ -f "$libdir/libfl.dll.a" ]; then
            FLEX_LDFLAGS="-L$libdir -lfl"
            echo "Found HPC module flex library in: $libdir"
            break
        fi
    done
fi

# Use FLEX_LIB_DIR if set by flex_detect snippet
if [ -z "$FLEX_LDFLAGS" ] && [ -n "${FLEX_LIB_DIR:-}" ] && [ -d "${FLEX_LIB_DIR}" ]; then
    if [ -f "${FLEX_LIB_DIR}/libfl.a" ] || [ -f "${FLEX_LIB_DIR}/libfl.so" ] || [ -f "${FLEX_LIB_DIR}/libfl.dll.a" ]; then
        FLEX_LDFLAGS="-L${FLEX_LIB_DIR} -lfl"
        echo "FLEX_LDFLAGS: $FLEX_LDFLAGS (from FLEX_LIB_DIR)"
    fi
fi

# Check if flex was built locally in the install directory (one level up from rhessys source)
if [ -z "$FLEX_LDFLAGS" ]; then
    for flexlib in "../flex/lib" "../../flex/lib"; do
        if [ -f "$flexlib/libfl.a" ] || [ -f "$flexlib/libfl.so" ] || [ -f "$flexlib/libfl.dll.a" ]; then
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

# Check conda environment for flex library
if [ -z "$FLEX_LDFLAGS" ] && [ -n "$CONDA_PREFIX" ]; then
    clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    for libdir in "$clp/lib"; do
        if [ -f "$libdir/libfl.a" ] || [ -f "$libdir/libfl.so" ] || [ -f "$libdir/libfl.dll.a" ]; then
            FLEX_LDFLAGS="-L$libdir -lfl"
            echo "Found flex library in conda: $libdir"
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

# Fix types.h: GCC 15 (C23 default) makes bool/true/false keywords.
# Remove the conflicting typedef and constants - the compiler provides them.
echo "Patching include/types.h to remove bool/true/false redefinitions..."
perl -i.bak -0777 -pe '
    s{typedef\s+short\s+bool;\s*\n\s*static\s+const\s+short\s+true\s*=\s*1;\s*\n\s*static\s+const\s+short\s+false\s*=\s*0;}
    {/* bool/true/false provided by compiler (C23) or stdbool.h */\n#include <stdbool.h>}s
' include/types.h
grep -q "stdbool.h" include/types.h && echo "types.h patched" || echo "WARNING: types.h patch may not have applied"

perl -i.bak -pe 's/int\s+key_compare\s*\(\s*void\s*\*\s*e1\s*,\s*void\s*\*\s*e2\s*\)/int key_compare(const void *e1, const void *e2)/' util/key_compare.c
perl -i.bak -pe 's/int\s+key_compare\s*\(\s*void\s*\*\s*,\s*void\s*\*\s*\)\s*;/int key_compare(const void *, const void *);/' util/sort_patch_layers.c
perl -i.bak -pe 's/(\s*)sort_patch_layers\(patch, \*rec\);/sort_patch_layers(patch, rec);/' util/sort_patch_layers.c
perl -i.bak -pe 's/^\s*#define MAXSTR 200/\/\/#define MAXSTR 200/' include/rhessys_fire.h

# Fix assign_base_station_xy.c - station->lon/lat don't exist in base_station_object struct.
# The is_approximately() function is defined in util/alloc.c under #ifdef LIU_NETCDF_READER,
# which is enabled via the makefile's DEFINES. We just need to comment out the broken lon/lat condition.
echo "Patching assign_base_station_xy.c for missing struct members..."
cat > ${TMPDIR:-/tmp}/rhessys_xy_patch.pl << 'PERLSCRIPT'
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
perl ${TMPDIR:-/tmp}/rhessys_xy_patch.pl init/assign_base_station_xy.c

# Fix construct_netcdf_grid.c - use perl for reliable pattern replacement
echo "Patching construct_netcdf_grid.c for missing struct members..."
cat > ${TMPDIR:-/tmp}/rhessys_grid_patch.pl << 'PERLSCRIPT'
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
perl ${TMPDIR:-/tmp}/rhessys_grid_patch.pl init/construct_netcdf_grid.c
echo "Patched construct_netcdf_grid.c"

# Fix construct_netcdf_header.c - use perl for reliable pattern replacement
echo "Patching construct_netcdf_header.c for missing struct members..."
cat > ${TMPDIR:-/tmp}/rhessys_header_patch.pl << 'PERLSCRIPT'
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
perl ${TMPDIR:-/tmp}/rhessys_header_patch.pl init/construct_netcdf_header.c

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

# Fix WMFire fire grid bug: patch ID 0 should not be treated as valid patch
# The condition if(tmpPatchID>=0) incorrectly treats 0 as valid, causing NULL pointer dereference
# when the patches array is allocated but not populated (no patch 0 exists in the world file)
echo "Patching construct_fire_grid.c for WMFire patch ID bug..."
if [ -f "init/construct_fire_grid.c" ]; then
    sed -i.bak 's/if(tmpPatchID>=0)/if(tmpPatchID>0)/' init/construct_fire_grid.c
    if grep -q 'if(tmpPatchID>0)' init/construct_fire_grid.c; then
        echo "Patched construct_fire_grid.c: tmpPatchID>=0 -> tmpPatchID>0"
    else
        echo "WARNING: construct_fire_grid.c patch may not have applied"
    fi
fi

# Fix handle_event.c to include WMFire event handler (missing in source)
echo "Patching handle_event.c to add fire_grid_on support..."
cat > ${TMPDIR:-/tmp}/rhessys_event_patch.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Add function declaration
if ($content !~ /void\s+execute_firespread_event/) {
    $changes += ($content =~ s/(\s*)void\s+execute_state_output_event/$1void execute_firespread_event(\n$1\tstruct world_object *,\n$1\tstruct command_line_object *,\n$1\tstruct date);\n$1void execute_state_output_event/);
}

# Add event handler
if ($content !~ /fire_grid_on/) {
    # Match the last else block
    $changes += ($content =~ s/(\s*)else\{\s*fprintf\(stderr,"FATAL ERROR: in handle event - event %s not recognized/\n$1else if ( !strcmp(event[0].command,"fire_grid_on") ){\n$1\texecute_firespread_event(world, command_line, current_date);\n$1}\n$1else{\n$1\tfprintf(stderr,"FATAL ERROR: in handle event - event %s not recognized/);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/rhessys_event_patch.pl tec/handle_event.c

# =============================================================
# SYMFLUENCE Patches for enhanced hydrological functionality
# =============================================================
# These patches add:
# 1. Subsurface-to-GW recharge pathway (-subsurfacegw flag)
# 2. NaN/negative sat_deficit numerical stability guard
#
# Enabled via: symfluence binary install rhessys --patched
# =============================================================

if [ "${SYMFLUENCE_PATCHED:-}" = "1" ]; then
echo "Applying SYMFLUENCE patches (--patched flag enabled)..."

# -------------------------------------------------------------
# SYMFLUENCE Patch 1: Add subsurface_gw_flag to rhessys.h
# -------------------------------------------------------------
echo "Patching rhessys.h for subsurface_gw_flag..."
cat > ${TMPDIR:-/tmp}/symfluence_rhessys_h.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Add subsurface_gw_flag after gw_flag in command_line_object struct
if ($content !~ /subsurface_gw_flag/) {
    $changes += ($content =~ s/(int\s+gw_flag;)/$1\n        int             subsurface_gw_flag;  \/* SYMFLUENCE: route subsurface drainage to GW *\//);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/symfluence_rhessys_h.pl include/rhessys.h

# -------------------------------------------------------------
# SYMFLUENCE Patch 1 (cont): Initialize subsurface_gw_flag in construct_command_line.c
# -------------------------------------------------------------
echo "Patching construct_command_line.c for flag initialization and parsing..."
cat > ${TMPDIR:-/tmp}/symfluence_cmdline.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Initialize subsurface_gw_flag = 0 after gw_flag = 1
if ($content !~ /subsurface_gw_flag\s*=\s*0/) {
    $changes += ($content =~ s/(command_line\[0\]\.gw_flag\s*=\s*1;)/$1\n\tcommand_line[0].subsurface_gw_flag = 0;  \/* SYMFLUENCE *\//);
}

# Add -subsurfacegw flag parsing after -gw handling block
# Look for the closing of the -gw block and add after it
if ($content !~ /-subsurfacegw/) {
    my $subsurfacegw_block = q{

			/*-------------------------------------------------*/
			/* SYMFLUENCE: subsurface drainage to GW pathway   */
			/*-------------------------------------------------*/
			else if ( strcmp(main_argv[i],"-subsurfacegw") == 0 ){
				i++;
				command_line[0].subsurface_gw_flag = 1;
			}/* end if */
};
    # Insert after the gw_loss_coeff_mult line's closing block
    $changes += ($content =~ s/(command_line\[0\]\.gw_loss_coeff_mult\s*=.*?i\+\+;\s*\}\s*\/\*\s*end if\s*\*\/)/$1$subsurfacegw_block/s);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/symfluence_cmdline.pl init/construct_command_line.c

# -------------------------------------------------------------
# SYMFLUENCE Patch 1 (cont): Add valid options
# -------------------------------------------------------------
echo "Patching valid_option.c for new flags..."
cat > ${TMPDIR:-/tmp}/symfluence_valid.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Add -longwaveevap and -subsurfacegw to valid options if not present
if ($content !~ /-subsurfacegw/) {
    # Find the last strcmp before the closing parenthesis and i = 0
    # Add our new options before the final closing paren
    $changes += ($content =~ s/(\(strcmp\(command_line,"-msr"\)\s*==\s*0\))/$1 ||\n\t\t(strcmp(command_line,"-longwaveevap") == 0) ||\n\t\t(strcmp(command_line,"-subsurfacegw") == 0)/);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/symfluence_valid.pl tec/valid_option.c

# -------------------------------------------------------------
# SYMFLUENCE Patch 1 (cont): Add subsurface-to-GW recharge in patch_daily_F.c
# -------------------------------------------------------------
echo "Patching patch_daily_F.c for subsurface-to-GW recharge..."
cat > ${TMPDIR:-/tmp}/symfluence_patch_daily.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Add subsurface-to-GW recharge code after rz_drainage updates
if ($content !~ /subsurface_gw_flag/) {
    my $gw_recharge_block = q{

	/* ---------------------------------------------- */
	/* SYMFLUENCE: Subsurface-to-GW recharge.         */
	/* Routes sat_to_gw_coeff * subsurface drainage   */
	/* to the hillslope GW store for baseflow.        */
	/* Gated by both -gw and -subsurfacegw flags.     */
	/* ---------------------------------------------- */
	if ((command_line[0].gw_flag) && (command_line[0].subsurface_gw_flag)) {
		double total_sub_drain = unsat_drainage + rz_drainage;
		if (total_sub_drain > ZERO) {
			double sub_gw = patch[0].soil_defaults[0][0].sat_to_gw_coeff
						* total_sub_drain;
			patch[0].gw_drainage += sub_gw;
			patch[0].sat_deficit += sub_gw;
			hillslope[0].gw.storage += (sub_gw * patch[0].area / hillslope[0].area);
		}
	}
};
    # Insert after the hourly_rz_drainage line
    $changes += ($content =~ s/(patch\[0\]\.hourly_rz_drainage\s*\+=\s*rz_drainage;)/$1$gw_recharge_block/);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/symfluence_patch_daily.pl cycle/patch_daily_F.c

# -------------------------------------------------------------
# SYMFLUENCE Patch 2: NaN/negative sat_deficit guard
# -------------------------------------------------------------
echo "Patching compute_subsurface_routing.c for NaN guards..."
cat > ${TMPDIR:-/tmp}/symfluence_nan_guard.pl << 'PERLSCRIPT'
use strict;
use warnings;
my $file = $ARGV[0];
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

my $changes = 0;

# Ensure math.h is included (for isnan)
if ($content !~ /#include\s*<math\.h>/) {
    $changes += ($content =~ s/(#include\s*<stdio\.h>)/$1\n#include <math.h>/);
}

# Add first NaN guard after overland_flow = 0.0
if ($content !~ /SYMFLUENCE.*NaN/) {
    my $nan_guard1 = q{

		/* ---------------------------------------------- */
		/* SYMFLUENCE: NaN/negative sat_deficit guard.    */
		/* Prevents numerical instability from propagating*/
		/* through the simulation.                        */
		/* ---------------------------------------------- */
		if (isnan(patch[0].sat_deficit) || patch[0].sat_deficit < 0.0) {
			patch[0].sat_deficit = 0.05;
		}
};
    $changes += ($content =~ s/(patch\[0\]\.overland_flow\s*=\s*0\.0;)/$1$nan_guard1/);
}

# Add second NaN guard after sat_deficit += Qout - Qin
if ($content !~ /Post-routing NaN guard/) {
    my $nan_guard2 = q{

			/* SYMFLUENCE: Post-routing NaN guard */
			if (isnan(patch[0].sat_deficit) || patch[0].sat_deficit < 0.0) {
				patch[0].sat_deficit = 0.05;
			}
};
    $changes += ($content =~ s/(patch\[0\]\.sat_deficit\s*\+=\s*\(patch\[0\]\.Qout\s*-\s*patch\[0\]\.Qin\);)/$1$nan_guard2/);
}

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
print "Patched $file with $changes changes\n";
PERLSCRIPT
perl ${TMPDIR:-/tmp}/symfluence_nan_guard.pl hydro/compute_subsurface_routing.c

# Verify SYMFLUENCE patches applied
echo "Verifying SYMFLUENCE patches..."
if grep -q "subsurface_gw_flag" include/rhessys.h && \
   grep -q "subsurfacegw" init/construct_command_line.c && \
   grep -q "subsurfacegw" tec/valid_option.c && \
   grep -q "subsurface_gw_flag" cycle/patch_daily_F.c && \
   grep -q "SYMFLUENCE" hydro/compute_subsurface_routing.c; then
    echo "SYMFLUENCE patches applied successfully"
else
    echo "WARNING: Some SYMFLUENCE patches may not have applied correctly"
fi

else
    echo "Building stock RHESSys (use --patched flag to enable SYMFLUENCE patches)"
fi

# Patch makefile to remove test dependency from rhessys target
# This allows building without glib-2.0 which is required only for tests
echo "Patching makefile to skip test dependency..."
sed -i.bak 's/^rhessys: \$(OBJECTS) test$/rhessys: $(OBJECTS)/' makefile

# On Windows, python3 triggers the Microsoft Store stub; conda provides 'python'
# Also add strndup polyfill (not in MinGW C runtime)
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        echo "Patching makefile: python3 -> python (Windows conda)..."
        sed -i 's/python3 /python /' makefile

        echo "Adding strndup polyfill for MinGW..."
        cat > ${TMPDIR:-/tmp}/_strndup_patch.pl << 'PERLEOF'
use strict; use warnings;
my $f = $ARGV[0];
open my $fh,'<',$f or die $!;
my $c = do{local $/;<$fh>};
close $fh;
# Insert polyfill right after the last #include line
my $poly = qq{\n#ifdef __MINGW32__\n#include <stdlib.h>\nchar *strndup(const char *s, size_t n) {\n    size_t len = strnlen(s, n);\n    char *d = (char *)malloc(len + 1);\n    if (d) { memcpy(d, s, len); d[len] = 0; }\n    return d;\n}\n#endif\n};
$c =~ s/(#include\s*"strings\.h")/$1$poly/;
open $fh,'>',$f or die $!;
print $fh $c;
close $fh;
PERLEOF
        perl ${TMPDIR:-/tmp}/_strndup_patch.pl util/strings.c
        ;;
esac
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
case "$OS" in
    Darwin)
        WMFIRE_LIB_NAME="libwmfire.dylib"
        ;;
    MSYS*|MINGW*|CYGWIN*)
        WMFIRE_LIB_NAME="libwmfire.dll"
        ;;
    *)
        WMFIRE_LIB_NAME="libwmfire.so"
        ;;
esac

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
        clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
        for boost_dir in "$clp/include" "$CONDA_PREFIX/include" /opt/homebrew/include /usr/local/include /usr/include; do
            if [ -d "$boost_dir/boost" ]; then
                BOOST_INCLUDE="-I$boost_dir"
                break
            fi
        done

        case "$OS" in
            Darwin)  SHARED_FLAG="-dynamiclib" ;;
            *)       SHARED_FLAG="-shared" ;;
        esac

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
# Try multiple possible locations for rhessys binary (handles .exe and versioned names)
RHESSYS_BIN=""
for candidate in rhessys.exe rhessys rhessys7*.exe rhessys7* rhessys/rhessys.exe rhessys/rhessys; do
    for match in $candidate; do
        if [ -f "$match" ]; then
            RHESSYS_BIN="$match"
            break 2
        fi
    done
done

if [ -n "$RHESSYS_BIN" ]; then
    cp -f "$RHESSYS_BIN" ../bin/rhessys
    echo "Staged: $RHESSYS_BIN -> ../bin/rhessys"
else
    echo "ERROR: rhessys binary not found after build"
    exit 1
fi

# On Windows, also ensure bin/rhessys.exe exists for native tools
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if [ ! -f ../bin/rhessys.exe ] && [ -f ../bin/rhessys ]; then
            cp ../bin/rhessys ../bin/rhessys.exe
        fi
        ;;
esac

chmod +x ../bin/rhessys 2>/dev/null || true
echo "RHESSys binary successfully built at ../bin/rhessys"
            '''.strip()
        ],
        'dependencies': ['gdal-config', 'proj', 'geos-config'],
        'test_command': '-h',
        'verify_install': {
            'file_paths': ['bin/rhessys', 'bin/rhessys.exe'],
            'check_type': 'exists_any'
        },
        'order': 14
    }
