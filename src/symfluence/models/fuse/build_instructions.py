"""
FUSE build instructions for SYMFLUENCE.

This module defines how to build FUSE from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

FUSE (Framework for Understanding Structural Errors) is a modular
rainfall-runoff modeling framework.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
    get_hdf5_detection,
    get_netcdf_lib_detection,
)


@BuildInstructionsRegistry.register('fuse')
def get_fuse_build_instructions():
    """
    Get FUSE build instructions.

    FUSE requires NetCDF and HDF5 libraries. The build uses make
    and requires special handling for legacy Fortran code.

    Returns:
        Dictionary with complete build configuration for FUSE.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()
    hdf5_detect = get_hdf5_detection()
    netcdf_lib_detect = get_netcdf_lib_detection()

    return {
        'description': 'Framework for Understanding Structural Errors',
        'config_path_key': 'FUSE_INSTALL_PATH',
        'config_exe_key': 'FUSE_EXE',
        'default_path_suffix': 'installs/fuse/bin',
        'default_exe': 'fuse.exe',
        'repository': 'https://github.com/CH-Earth/fuse.git',
        'branch': None,
        'install_dir': 'fuse',
        'build_commands': [
            common_env,
            netcdf_detect,
            hdf5_detect,
            netcdf_lib_detect,
            r'''
# Map to FUSE Makefile variable names
export NCDF_PATH="$NETCDF_FORTRAN"
export HDF_PATH="$HDF5_ROOT"

# Platform-specific linker flags for Homebrew
if command -v brew >/dev/null 2>&1; then
  export CPPFLAGS="${CPPFLAGS:+$CPPFLAGS }-I$(brew --prefix netcdf)/include -I$(brew --prefix netcdf-fortran)/include"
  export LDFLAGS="${LDFLAGS:+$LDFLAGS }-L$(brew --prefix netcdf)/lib -L$(brew --prefix netcdf-fortran)/lib"
  [ -d "$HDF_PATH/include" ] && export CPPFLAGS="${CPPFLAGS} -I${HDF_PATH}/include"
  [ -d "$HDF_PATH/lib" ] && export LDFLAGS="${LDFLAGS} -L${HDF_PATH}/lib"
fi

echo "=== FUSE Build Starting ==="
echo "FC=${FC}, NCDF_PATH=${NCDF_PATH}, HDF_PATH=${HDF_PATH}"

cd build
export F_MASTER="$(cd .. && pwd)/"

# Construct library and include paths
LIBS="-L${HDF5_LIB_DIR} -lhdf5 -lhdf5_hl -L${NETCDF_LIB_DIR} -lnetcdff -L${NETCDF_C_LIB_DIR} -lnetcdf"
INCLUDES="-I${HDF5_INC_DIR} -I${NCDF_PATH}/include -I${NETCDF_C}/include"

# =====================================================
# STEP 1: Create a gfortran wrapper to force compiler flags
# =====================================================
echo ""
echo "=== Step 1: Creating gfortran wrapper ==="

# The FUSE Makefile doesn't pass FFLAGS to the compile step.
# Solution: Create a wrapper script that intercepts gfortran calls
# and adds our required flags.

WRAPPER_DIR="$(pwd)/wrapper"
mkdir -p "$WRAPPER_DIR"

# Create the wrapper script
cat > "$WRAPPER_DIR/gfortran" << 'WRAPEOF'
#!/bin/bash
# Wrapper for gfortran that adds required flags for FUSE compilation
# These flags allow long lines and disable -Werror

EXTRA_FLAGS="-ffree-line-length-none -fallow-argument-mismatch -std=legacy -Wno-error -Wno-line-truncation"

# Find the real gfortran
REAL_GFORTRAN="/usr/bin/gfortran"
if [ ! -x "$REAL_GFORTRAN" ]; then
    REAL_GFORTRAN=$(which gfortran 2>/dev/null | grep -v wrapper | head -1)
fi

# Call the real gfortran with our extra flags
exec "$REAL_GFORTRAN" $EXTRA_FLAGS "$@"
WRAPEOF

chmod +x "$WRAPPER_DIR/gfortran"
echo "Created gfortran wrapper at: $WRAPPER_DIR/gfortran"

# Put our wrapper first in PATH so make uses it
export PATH="$WRAPPER_DIR:$PATH"
echo "PATH updated: $WRAPPER_DIR is first"

# Verify the wrapper works
echo "Testing wrapper..."
"$WRAPPER_DIR/gfortran" --version | head -1

# Also set FC to use our wrapper explicitly
export FC="$WRAPPER_DIR/gfortran"
echo "FC set to: $FC"

# =====================================================
# STEP 2: Disable MPI code blocks entirely
# =====================================================
echo ""
echo "=== Step 2: Disabling MPI code ==="

# The gfortran wrapper handles long lines with -ffree-line-length-none
# Here we need to completely disable MPI code since MPI is not installed.
# Simply commenting out #ifdef doesn't work - we need to comment out
# ALL lines between #ifdef __MPI__ and #endif/#else

# Create a perl script to comment out MPI blocks
cat > disable_mpi.pl << 'PERLEOF'
#!/usr/bin/perl -i
use strict;
use warnings;

my $in_mpi_block = 0;
my $in_else_block = 0;

while (<>) {
    # Start of MPI block
    if (/^#ifdef __MPI__/) {
        print "! MPI DISABLED: $_";
        $in_mpi_block = 1;
        $in_else_block = 0;
        next;
    }

    # #else - switch to non-MPI code (stop commenting)
    if (/^#else/ && $in_mpi_block) {
        print "! MPI DISABLED: $_";
        $in_mpi_block = 0;
        $in_else_block = 1;
        next;
    }

    # #endif - end of block
    if (/^#endif/ && ($in_mpi_block || $in_else_block)) {
        print "! MPI DISABLED: $_";
        $in_mpi_block = 0;
        $in_else_block = 0;
        next;
    }

    # Inside MPI block - comment out the line
    if ($in_mpi_block) {
        # Don't double-comment already commented lines
        if (/^\s*!/) {
            print;
        } else {
            print "! MPI DISABLED: $_";
        }
        next;
    }

    # Inside else block (non-MPI code) - keep as is
    # Or outside any block - keep as is
    print;
}
PERLEOF
chmod +x disable_mpi.pl

# Apply to files with MPI code
GET_GFORCE_FILE="FUSE_SRC/FUSE_NETCDF/get_gforce.f90"
if [ -f "$GET_GFORCE_FILE" ]; then
    perl disable_mpi.pl "$GET_GFORCE_FILE"
    # Also comment out standalone 'use mpi' if any
    sed -i 's/^\([[:space:]]*\)use mpi[[:space:]]*$/\1! use mpi  ! MPI not available/' "$GET_GFORCE_FILE"
    echo "  Disabled MPI in get_gforce.f90"
fi

FUSE_DRIVER_FILE="FUSE_SRC/FUSE_DMSL/fuse_driver.f90"
if [ -f "$FUSE_DRIVER_FILE" ]; then
    perl disable_mpi.pl "$FUSE_DRIVER_FILE"
    echo "  Disabled MPI in fuse_driver.f90"
fi

# Verify MPI is disabled
echo "  Checking for remaining MPI references..."
MPI_REFS=$(grep -r "MPI_COMM_WORLD\|call MPI_" FUSE_SRC --include="*.f90" | grep -v "^[[:space:]]*!" | grep -v "MPI DISABLED" | head -5)
if [ -n "$MPI_REFS" ]; then
    echo "  WARNING: Some MPI references remain:"
    echo "$MPI_REFS"
else
    echo "  All MPI code disabled"
fi

rm -f disable_mpi.pl
echo "MPI disabling complete"

# =====================================================
# STEP 3: Pre-compile fixed-form Fortran
# =====================================================
echo ""
echo "=== Step 3: Pre-compile fixed-form Fortran ==="
FFLAGS_FIXED="-O2 -c -ffixed-form -fallow-argument-mismatch -std=legacy -Wno-error"
${FC} ${FFLAGS_FIXED} -o sce_16plus.o "FUSE_SRC/FUSE_SCE/sce_16plus.f" || echo "Warning: sce_16plus.f compilation issue"

# =====================================================
# STEP 4: Run make (with our patched Makefile)
# =====================================================
echo ""
echo "=== Step 4: Running make ==="

# Clean first
make clean 2>/dev/null || true

# Run make with explicit variables
make -j1 FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}"

# Check result
if [ -f "fuse.exe" ] || [ -f "../bin/fuse.exe" ]; then
    echo "Build successful!"
    if [ -f "fuse.exe" ] && [ ! -f "../bin/fuse.exe" ]; then
        mkdir -p ../bin && cp fuse.exe ../bin/
    fi
else
    echo "Build failed - fuse.exe not found"
    echo "Checking for partial build products..."
    ls -la *.o 2>/dev/null | wc -l
    ls -la *.mod 2>/dev/null | wc -l
    exit 1
fi

echo "=== FUSE Build Complete ==="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,  # FUSE exits with error when run without args
        'verify_install': {
            'file_paths': ['bin/fuse.exe'],
            'check_type': 'exists'
        },
        'order': 5
    }
