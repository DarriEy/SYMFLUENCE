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

echo "Build environment: FC=${FC}, NCDF_PATH=${NCDF_PATH}, HDF_PATH=${HDF_PATH}"

# Build FUSE
cd build
make clean 2>/dev/null || true
export F_MASTER="$(cd .. && pwd)/"

# Construct library and include paths
LIBS="-L${HDF5_LIB_DIR} -lhdf5 -lhdf5_hl -L${NETCDF_LIB_DIR} -lnetcdff -L${NETCDF_C_LIB_DIR} -lnetcdf"
INCLUDES="-I${HDF5_INC_DIR} -I${NCDF_PATH}/include -I${NETCDF_C}/include"

# Legacy compiler flags for old Fortran code
# -Wno-line-truncation: FUSE has some long lines that get truncated
# -Wno-error=line-truncation: Don't treat line truncation as fatal error
# -Wno-error: Don't treat any warnings as errors (Makefile may add -Werror)
# -fallow-argument-mismatch: Legacy code has type mismatches
# -std=legacy: Allow legacy Fortran constructs
# -ffree-line-length-512: Allow lines up to 512 chars (more reliable than -none)
EXTRA_FLAGS="-fallow-argument-mismatch -std=legacy -Wno-line-truncation -Wno-error=line-truncation -Wno-error"
FFLAGS_NORMA="-O3 -ffree-line-length-512 -fmax-errors=0 -cpp ${EXTRA_FLAGS}"
FFLAGS_FIXED="-O2 -c -ffixed-form ${EXTRA_FLAGS}"

# Pre-compile sce_16plus.f to avoid broken Makefile rule
echo "Pre-compiling sce_16plus.f..."
${FC} ${FFLAGS_FIXED} -o sce_16plus.o "FUSE_SRC/FUSE_SCE/sce_16plus.f" || { echo "Failed to compile sce_16plus.f"; exit 1; }

# FUSE has complex Fortran module dependencies that the Makefile doesn't handle well.
# Solution: Multi-pass compilation - keep compiling all module files until no new .mod files appear.

echo "Building FUSE (with multi-pass module pre-compilation)..."

# Create objects directory
mkdir -p objects

# Find all Fortran source files that define modules
echo "Finding all module-defining source files..."
MODULE_SOURCES=$(find ../build/FUSE_SRC -name "*.f90" -exec grep -l "^[[:space:]]*module[[:space:]]" {} \; 2>/dev/null | sort -u)
echo "Found $(echo "$MODULE_SOURCES" | wc -l | tr -d ' ') module files"

# Multi-pass compilation: keep compiling until no new .mod files are created
MAX_PASSES=10
PASS=1
while [ $PASS -le $MAX_PASSES ]; do
    echo ""
    echo "=== Compilation pass $PASS ==="
    MOD_COUNT_BEFORE=$(ls -1 *.mod 2>/dev/null | wc -l | tr -d ' ')

    COMPILED=0
    FAILED=0
    for src in $MODULE_SOURCES; do
        # Extract module name from file
        basename_src=$(basename "$src")

        # Try to compile, suppress errors (some will fail due to missing deps)
        if ${FC} ${FFLAGS_NORMA} ${INCLUDES} -c "$src" 2>/dev/null; then
            COMPILED=$((COMPILED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done

    MOD_COUNT_AFTER=$(ls -1 *.mod 2>/dev/null | wc -l | tr -d ' ')
    NEW_MODS=$((MOD_COUNT_AFTER - MOD_COUNT_BEFORE))

    echo "Pass $PASS: Compiled=$COMPILED, Failed=$FAILED, New .mod files=$NEW_MODS, Total .mod files=$MOD_COUNT_AFTER"

    # If no new .mod files were created, we're done
    if [ $NEW_MODS -eq 0 ] && [ $PASS -gt 1 ]; then
        echo "No new modules created - compilation stabilized"
        break
    fi

    PASS=$((PASS + 1))
done

# Now specifically check and compile critical modules with verbose errors
echo ""
echo "=== Verifying critical modules ==="

# kinds_dmsl_kit_FUSE is the base
if [ ! -f "kinds_dmsl_kit_fuse.mod" ]; then
    KINDS_SRC=$(find .. -name "kinds_dmsl_kit_FUSE.f90" 2>/dev/null | head -1)
    if [ -n "$KINDS_SRC" ]; then
        echo "Compiling kinds_dmsl_kit_FUSE (verbose)..."
        ${FC} ${FFLAGS_NORMA} ${INCLUDES} -c "$KINDS_SRC" || { echo "FATAL: kinds_dmsl_kit_FUSE failed"; exit 1; }
    fi
fi

# fuse_fileManager is required by fuse_driver
if [ ! -f "fuse_filemanager.mod" ]; then
    FILEMANAGER_SRC=$(find .. -name "fuse_fileManager.f90" 2>/dev/null | head -1)
    if [ -n "$FILEMANAGER_SRC" ]; then
        echo "Compiling fuse_fileManager (verbose, showing errors)..."
        ${FC} ${FFLAGS_NORMA} ${INCLUDES} -c "$FILEMANAGER_SRC" 2>&1 || echo "fuse_fileManager compilation failed - see errors above"
    fi
fi

# List all .mod files for debugging
echo ""
echo "Available .mod files:"
ls -1 *.mod 2>/dev/null | head -30 || echo "  (none)"

# Check critical modules
if [ ! -f "fuse_filemanager.mod" ]; then
    echo ""
    echo "WARNING: fuse_filemanager.mod still not created"
    echo "Checking what modules fuse_fileManager.f90 needs..."
    FILEMANAGER_SRC=$(find .. -name "fuse_fileManager.f90" 2>/dev/null | head -1)
    if [ -n "$FILEMANAGER_SRC" ]; then
        echo "USE statements in fuse_fileManager.f90:"
        grep -i "^[[:space:]]*use[[:space:]]" "$FILEMANAGER_SRC" | head -20
    fi
fi

# Now run make - the module should exist
echo "Running make..."
if make -j1 FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" \
       FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}"; then
  echo "Build completed"
else
  echo "Build failed - checking for .mod files..."
  find . -name "*.mod" 2>/dev/null
  echo "NetCDF lib: ${NETCDF_LIB_DIR}, HDF5 lib: ${HDF5_LIB_DIR}"
  exit 1
fi

# Stage binary
if [ -f "../bin/fuse.exe" ]; then
  echo "Binary at ../bin/fuse.exe"
elif [ -f "fuse.exe" ]; then
  mkdir -p ../bin && cp fuse.exe ../bin/
  echo "Binary staged to ../bin/fuse.exe"
else
  echo "fuse.exe not found after build"
  exit 1
fi
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
