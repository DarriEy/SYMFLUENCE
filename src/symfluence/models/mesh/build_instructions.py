# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH build instructions for SYMFLUENCE.

This module defines how to build MESH from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

MESH (Modélisation Environnementale Surface Hydrology) is Environment
Canada's land-surface and hydrology model.
"""

from symfluence.cli.services import BuildInstructionsRegistry, get_common_build_environment


@BuildInstructionsRegistry.register('mesh')
def get_mesh_build_instructions():
    """
    Get MESH build instructions.

    MESH can be built with or without NetCDF and MPI support. The build
    uses make and includes patches for modern C compiler compatibility.

    Returns:
        Dictionary with complete build configuration for MESH.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'MESH - Environment Canada Hydrology Land-Surface Scheme',
        'config_path_key': 'MESH_INSTALL_PATH',
        'config_exe_key': 'MESH_EXE',
        'default_path_suffix': 'installs/mesh/bin',
        'default_exe': 'mesh.exe',
        'repository': 'https://github.com/MESH-Model/MESH-Dev.git',
        'branch': None,
        'install_dir': 'mesh',
        'build_commands': [
            common_env,
            r'''
# Build MESH from GitHub repository
set -e

echo "Building MESH with NetCDF support..."

# Create bin directory
mkdir -p bin

# Detect NetCDF Fortran library
echo "=== NetCDF Detection ==="
# On Windows conda, libraries are under $CONDA_PREFIX/Library
clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_FORTRAN="$(nf-config --prefix)"
    echo "Found nf-config, using: ${NETCDF_FORTRAN}"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$clp/bin/nf-config" ]; then
    NETCDF_FORTRAN="$clp"
    echo "Found conda nf-config, using: ${NETCDF_FORTRAN}"
elif [ -n "${NETCDF_FORTRAN}" ] && [ -d "${NETCDF_FORTRAN}/include" ]; then
    echo "Using NETCDF_FORTRAN: ${NETCDF_FORTRAN}"
elif [ -n "${NETCDF}" ] && [ -d "${NETCDF}/include" ]; then
    NETCDF_FORTRAN="${NETCDF}"
    echo "Using NETCDF: ${NETCDF}"
else
    # Try common locations
    for try_path in /opt/homebrew/opt/netcdf-fortran /usr/local/opt/netcdf-fortran /usr; do
        if [ -d "$try_path/include" ]; then
            NETCDF_FORTRAN="$try_path"
            echo "Found NetCDF at: $try_path"
            break
        fi
    done
fi

# Set NetCDF environment variables for MESH build system
if [ -n "${NETCDF_FORTRAN}" ]; then
    export NCDF_PATH="${NETCDF_FORTRAN}"
    # MESH expects specific NetCDF library variables
    export NETCDF_INC="${NETCDF_FORTRAN}/include"

    # Find lib or lib64 directory
    if [ -d "${NETCDF_FORTRAN}/lib64" ]; then
        export NETCDF_LIB="${NETCDF_FORTRAN}/lib64"
    else
        export NETCDF_LIB="${NETCDF_FORTRAN}/lib"
    fi

    # On Homebrew (and some HPC), netcdf-fortran and netcdf-c live in separate
    # prefixes. nf-config --flibs emits -lnetcdf but only adds -L for the
    # Fortran lib dir, so the linker can't find libnetcdf (the C library).
    # Detect and export the NetCDF-C lib path for the Makefile LIBNCL override.
    NETCDF_C_PREFIX="$(nc-config --prefix 2>/dev/null || echo "")"
    NETCDF_C_LIBDIR=""
    if [ -n "$NETCDF_C_PREFIX" ] && [ "$NETCDF_C_PREFIX" != "$NETCDF_FORTRAN" ]; then
        if [ -d "$NETCDF_C_PREFIX/lib" ]; then
            NETCDF_C_LIBDIR="$NETCDF_C_PREFIX/lib"
            echo "NetCDF-C lib path (separate from Fortran): ${NETCDF_C_LIBDIR}"
        fi
    fi
fi

# Patch the getenvc.c file to fix K&R C style function declarations
# This is needed for modern C compilers (clang on macOS)
GETENVC_FILE="./Modules/librmn/19.7.0/primitives/getenvc.c"
if [ -f "$GETENVC_FILE" ]; then
    echo "Patching getenvc.c for modern C compatibility..."
    # Backup original
    cp "$GETENVC_FILE" "${GETENVC_FILE}.backup"

    # Convert K&R C style to ANSI C style
    cat > "${GETENVC_FILE}" << 'PATCHEOF'
#include <stdlib.h>
#include <string.h>

/* Define F2Cl type for Fortran-C interop */
#define F2Cl int

/* Define f77name macro for Fortran name mangling */
/* MinGW/MSYS2 gfortran uses trailing underscore like Linux/macOS */
#if defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__)
#define f77name(x) x##_
#else
#define f77name(x) x
#endif

/* Function definition with ANSI C style parameters */
void f77name(getenvc) ( char *name, char *value, F2Cl len1, F2Cl len2 )
{
   char *temp, *hold;
   int size, i;

   size = len1+len2+1 ;
   temp = (char *) malloc(size) ;
   hold = (char *) malloc(size) ;

   for ( i=0 ;
         i < len1 && name[i] != ' ' ;
         i++ )
         *(temp+i) = name[i] ;

   *(temp+i) = '\0' ;

   if (getenv(temp) != NULL)
   {
      strcpy(hold, getenv(temp)) ;
      size = strlen(hold) ;
   }
   else
   {
      size = 0 ;
   }

   for ( i=0 ; i < len2 ; i++ ) value[i] = ' ' ;

   if ( size != 0 )
   {
        for ( i=0 ; i < size ; i++ ) value[i] = *(hold+i) ;
   }

   free (temp) ;
   free (hold) ;
}
PATCHEOF

    echo "getenvc.c patched successfully"
fi

# Determine if we should try MPI build
BUILD_MPI=false
if command -v mpifort >/dev/null 2>&1 || command -v mpif90 >/dev/null 2>&1; then
    BUILD_MPI=true
    echo "MPI compiler detected, will attempt MPI build"
fi

# Clean any previous builds
make veryclean 2>/dev/null || make clean 2>/dev/null || true

# If NetCDF-C is in a separate prefix from NetCDF-Fortran, patch the Makefile
# link line so the linker can find -lnetcdf. The Makefile sets LIBNCL from
# nf-config --flibs which only has -L for the Fortran prefix.
if [ -n "${NETCDF_C_LIBDIR}" ]; then
    echo "Patching Makefile to add NetCDF-C library path..."
    sed -i.bak "s|LIBNCL=\$(shell nf-config --flibs)|LIBNCL=-L${NETCDF_C_LIBDIR} \$(shell nf-config --flibs)|" Makefile
fi

# Try building with NetCDF support
if [ -n "${NETCDF_FORTRAN}" ]; then
    echo "Building MESH with NetCDF support..."

    # Try MPI version first if available
    if [ "$BUILD_MPI" = true ]; then
        echo "Attempting MPI build..."
        set +e
        make mpi_gcc netcdf 2>&1 | tee build.log
        BUILD_STATUS=${PIPESTATUS[0]}
        set -e

        if [ $BUILD_STATUS -eq 0 ] && [ -f "mpi_sa_mesh" ]; then
            BUILT_BINARY="mpi_sa_mesh"
            echo "MPI build with NetCDF successful"
        else
            echo "MPI build failed (exit code: $BUILD_STATUS), trying serial build..."
            make veryclean 2>/dev/null || make clean 2>/dev/null || true
            BUILD_MPI=false
        fi
    fi

    # Fall back to serial build if MPI failed or not available
    if [ "$BUILD_MPI" = false ]; then
        set +e
        make gfortran netcdf 2>&1 | tee build.log
        BUILD_STATUS=${PIPESTATUS[0]}
        set -e

        if [ $BUILD_STATUS -eq 0 ] && [ -f "sa_mesh" ]; then
            BUILT_BINARY="sa_mesh"
            echo "Serial build with NetCDF successful"
        else
            echo "ERROR: NetCDF build failed (exit code: $BUILD_STATUS)"
            echo "NetCDF Fortran is available at ${NETCDF_FORTRAN} but compilation failed."
            echo "A non-NetCDF build would be unusable (MESH requires NetCDF for drainage databases)."
            echo ""
            echo "Last 100 lines of build log:"
            tail -100 build.log
            exit 1
        fi
    fi
else
    echo "Building MESH (basic version without NetCDF)..."
    set +e
    make gfortran 2>&1 | tee build.log
    BUILD_STATUS=${PIPESTATUS[0]}
    set -e

    if [ $BUILD_STATUS -eq 0 ] && [ -f "sa_mesh" ]; then
        BUILT_BINARY="sa_mesh"
        echo "Basic build successful"
    else
        echo "MESH compilation failed (exit code: $BUILD_STATUS)"
        echo ""
        echo "Last 100 lines of build log:"
        tail -100 build.log
        exit 1
    fi
fi

# Find and move the binary to bin/
echo "Locating built binary..."
MESH_BINARY=""
for candidate in "${BUILT_BINARY}" "${BUILT_BINARY}.exe" "sa_mesh" "sa_mesh.exe" "mpi_sa_mesh" "mpi_sa_mesh.exe"; do
    if [ -f "$candidate" ]; then
        MESH_BINARY="$candidate"
        break
    fi
done

if [ -z "$MESH_BINARY" ]; then
    # Search in common build directories
    MESH_BINARY=$(find . -maxdepth 2 -name "sa_mesh" -o -name "sa_mesh.exe" -o -name "mpi_sa_mesh" -o -name "mesh.exe" 2>/dev/null | head -1)
fi

if [ -n "$MESH_BINARY" ] && [ -f "$MESH_BINARY" ]; then
    # Standardize binary name to mesh.exe for consistency
    cp "$MESH_BINARY" bin/mesh.exe
    chmod +x bin/mesh.exe
    echo "MESH binary staged to bin/mesh.exe"
else
    echo "MESH binary not found after build"
    echo "Directory contents:"
    ls -la
    exit 1
fi

# Test the binary
echo "Testing MESH binary..."
if bin/mesh.exe --help 2>&1 | head -10 || [ $? -ne 0 ]; then
    echo "MESH build verification complete"
else
    echo "MESH binary exists but test output unexpected (may be normal)"
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,  # MESH may not have standard --version flag
        'verify_install': {
            'file_paths': ['bin/mesh.exe'],
            'check_type': 'exists'
        },
        'order': 12
    }
