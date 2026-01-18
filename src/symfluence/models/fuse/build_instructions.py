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
# STEP 1: Patch the Makefile to use our compiler flags
# =====================================================
echo ""
echo "=== Step 1: Patching Makefile ==="
if [ -f "Makefile" ]; then
    cp Makefile Makefile.original

    # The FUSE Makefile has a 'compile' target that runs gfortran directly
    # WITHOUT using FFLAGS_NORMA. We need to patch this rule.

    # Define our flags
    OUR_FFLAGS="-O3 -ffree-line-length-none -fmax-errors=0 -cpp -fallow-argument-mismatch -std=legacy -Wno-error -Wno-line-truncation"

    # Method 1: Set FCFLAGS environment variable (gfortran picks this up)
    export FCFLAGS="$OUR_FFLAGS"
    export FFLAGS="$OUR_FFLAGS"

    # Method 2: Patch the Makefile to add $(FFLAGS_NORMA) to the compile rule
    # The compile rule looks like: $(FC) $(SOURCES) $(LIBS) $(INCLUDES) -o fuse.exe
    # We need to add $(FFLAGS_NORMA) after $(FC)

    # First, add our FFLAGS_NORMA definition at the top
    cat > Makefile.patched << 'MKEOF'
# Override FFLAGS for FUSE build - disable -Werror and enable long lines
FFLAGS_NORMA = -O3 -ffree-line-length-none -fmax-errors=0 -cpp -fallow-argument-mismatch -std=legacy -Wno-error -Wno-line-truncation
FFLAGS_FIXED = -O2 -ffixed-form -fallow-argument-mismatch -std=legacy -Wno-error -Wno-line-truncation
MKEOF

    # Now patch the compile rule in the original Makefile
    # The compile target has a line like: $(FC) $(F90FILES)... -o fuse.exe
    # We need to add $(FFLAGS_NORMA) after $(FC)
    sed 's/\$(FC) \$(F90FILES)/$(FC) $(FFLAGS_NORMA) $(F90FILES)/g' Makefile.original >> Makefile.patched
    sed -i 's/\$(FC)  \$(F90FILES)/$(FC) $(FFLAGS_NORMA) $(F90FILES)/g' Makefile.patched

    # Also handle case where FC is followed directly by source file paths
    sed -i 's/\$(FC) \//$(FC) $(FFLAGS_NORMA) \//g' Makefile.patched

    mv Makefile.patched Makefile

    echo "Makefile patched - checking compile rule:"
    grep -A2 "^compile:" Makefile || grep "fuse.exe" Makefile | head -3
else
    echo "ERROR: Makefile not found"
    ls -la
    exit 1
fi

# =====================================================
# STEP 2: Comprehensive source file patching
# =====================================================
echo ""
echo "=== Step 2: Patching ALL source files with long lines ==="

# The FUSE codebase has many lines > 132 chars which gfortran truncates
# This causes syntax errors. We need to break ALL long lines.

# Create a perl script to fix long lines in Fortran files
cat > fix_long_lines.pl << 'PERLEOF'
#!/usr/bin/perl -i.bak
use strict;
use warnings;

while (<>) {
    # Skip comment lines
    if (/^\s*!/) {
        print;
        next;
    }

    # If line is > 125 chars (leaving margin), try to break it
    while (length($_) > 125 && !/^\s*!/) {
        # Try to break at "; " (statement separator)
        if (s/^(.{60,120});\s*(.+)$/$1\n   $2/) {
            print "$1;\n";
            $_ = "   $2";
            next;
        }
        # Try to break at ")then; " pattern
        if (s/^(.{60,120}\)then);\s*(.+)$/$1\n   $2/) {
            print "$1;\n";
            $_ = "   $2";
            next;
        }
        # If we can't break it safely, just print and move on
        last;
    }
    print;
}
PERLEOF
chmod +x fix_long_lines.pl

# Apply to all .f90 files
echo "  Fixing long lines in all Fortran files..."
find FUSE_SRC -name "*.f90" -exec perl fix_long_lines.pl {} \;

# Additional specific patches for known problematic files

# Patch fuse_fileManager.f90 line 141 - very long string
FILEMANAGER_FILE="FUSE_SRC/FUSE_HOOK/fuse_fileManager.f90"
if [ -f "$FILEMANAGER_FILE" ]; then
    perl -i -pe "s|(message='This version of FUSE requires the file manager to follow the following format:  ')//trim\(fuseFileManagerHeader\)//(' not '//trim\(temp\))|\1\&\n               //trim(fuseFileManagerHeader)//\2|" "$FILEMANAGER_FILE"
    echo "  Patched fuse_fileManager.f90"
fi

# Patch get_gforce.f90 - has many long lines AND MPI issues
GET_GFORCE_FILE="FUSE_SRC/FUSE_NETCDF/get_gforce.f90"
if [ -f "$GET_GFORCE_FILE" ]; then
    # Break specific long lines that the generic script might miss
    # Line 375: nf90_get_var call
    perl -i -pe 's/(ierr = nf90_get_var\(ncid_forc, ncid_var\(ivar\), gTemp, start=\(\/1,startSpat2,iTim\/\), count=\(\/nSpat1,nSpat2,1\/\)\));/$1\n   /g' "$GET_GFORCE_FILE"

    # Lines 379-381: if statements with assignments
    perl -i -pe 's/(if\(trim\(cVec\(iVar\)%vname\) == trim\(vname_\w+\) \)then); (gForce.*)/$1\n    $2/g' "$GET_GFORCE_FILE"

    # Line 483: another nf90_get_var call
    perl -i -pe 's/(ierr = nf90_get_var\(ncid_forc, ncid_var\(ivar\), gTemp, start=\(\/1,startSpat2,itim_start\/\), count=\(\/nSpat1,nSpat2,numtim\/\)\));/$1\n   /g' "$GET_GFORCE_FILE"

    # Comment out 'use mpi' - MPI not available
    perl -i -pe 's/^(\s*)(use mpi\s*)$/$1!$2  ! Disabled - MPI not available\n/' "$GET_GFORCE_FILE"

    # Replace preprocessor directives with Fortran comments
    perl -i -pe 's/^#ifdef __MPI__/! MPI DISABLED: ifdef __MPI__/' "$GET_GFORCE_FILE"
    perl -i -pe 's/^#else/! MPI DISABLED: else/' "$GET_GFORCE_FILE"
    perl -i -pe 's/^#endif/! MPI DISABLED: endif/' "$GET_GFORCE_FILE"

    echo "  Patched get_gforce.f90 (long lines + MPI)"
fi

# Patch fuse_driver.f90 for MPI preprocessor directives
FUSE_DRIVER_FILE="FUSE_SRC/FUSE_DMSL/fuse_driver.f90"
if [ -f "$FUSE_DRIVER_FILE" ]; then
    perl -i -pe 's/^#ifdef __MPI__/! MPI DISABLED: ifdef __MPI__/' "$FUSE_DRIVER_FILE"
    perl -i -pe 's/^#else/! MPI DISABLED: else/' "$FUSE_DRIVER_FILE"
    perl -i -pe 's/^#endif/! MPI DISABLED: endif/' "$FUSE_DRIVER_FILE"
    echo "  Patched fuse_driver.f90 (MPI directives)"
fi

# Verify patches - check for remaining long lines
echo ""
echo "  Checking for remaining long lines (>130 chars)..."
LONG_LINES=$(find FUSE_SRC -name "*.f90" -exec awk 'length > 130 {print FILENAME": line "NR": "length" chars"}' {} \; | head -10)
if [ -n "$LONG_LINES" ]; then
    echo "  WARNING: Some long lines remain:"
    echo "$LONG_LINES"
else
    echo "  All source files have lines <= 130 chars"
fi

echo "Source patching complete"

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
