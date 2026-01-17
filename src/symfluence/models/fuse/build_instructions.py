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
EXTRA_FLAGS="-fallow-argument-mismatch -std=legacy"
FFLAGS_NORMA="-O3 -ffree-line-length-none -fmax-errors=0 -cpp ${EXTRA_FLAGS}"
FFLAGS_FIXED="-O2 -c -ffixed-form ${EXTRA_FLAGS}"

# Pre-compile sce_16plus.f to avoid broken Makefile rule
echo "Pre-compiling sce_16plus.f..."
${FC} ${FFLAGS_FIXED} -o sce_16plus.o "FUSE_SRC/FUSE_SCE/sce_16plus.f" || { echo "Failed to compile sce_16plus.f"; exit 1; }

# FUSE has complex Fortran module dependencies that the Makefile doesn't handle well.
# Strategy: Run make multiple times - first pass builds modules, second pass links.
# This is a common workaround for Makefiles with incomplete Fortran dependencies.

echo "Building FUSE (multi-pass for Fortran module dependencies)..."

# First pass: Let make build what it can, ignoring errors
# This typically builds the base modules that other files need
echo "Pass 1: Building base modules..."
make -j1 FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" \
     FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}" 2>&1 || true

# Check what .mod files were created
echo "Modules created after pass 1:"
ls -la *.mod 2>/dev/null || echo "  (none in current dir)"
ls -la objects/*.mod 2>/dev/null || echo "  (none in objects/)"

# Second pass: Now that modules exist, dependent files can compile
echo "Pass 2: Building dependent modules..."
make -j1 FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" \
     FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}" 2>&1 || true

# Third pass: Final link
echo "Pass 3: Final build..."
if make -j1 FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" \
       FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}"; then
  echo "Build completed"
else
  echo "Build failed - NetCDF lib: ${NETCDF_LIB_DIR}, HDF5 lib: ${HDF5_LIB_DIR}"
  echo "Checking for partial build..."
  ls -la *.exe fuse* 2>/dev/null || true
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
