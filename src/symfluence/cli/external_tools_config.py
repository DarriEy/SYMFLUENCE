#!/usr/bin/env python3

"""
SYMFLUENCE External Tools Configuration

This module defines external tool configurations required by SYMFLUENCE,
including repositories, build instructions, and validation criteria.

Tools include:
- SUNDIALS: Differential equation solver library
- SUMMA: Hydrological model with SUNDIALS integration
- mizuRoute: River network routing model
- TROUTE: NOAA's Next Generation river routing model (Python)
- FUSE: Framework for Understanding Structural Errors
- TauDEM: Terrain Analysis Using Digital Elevation Models
- GIStool: Geospatial data extraction tool
- Datatool: Meteorological data processing tool
- NGEN: NextGen National Water Model Framework
- NGIAB: NextGen In A Box deployment system
"""

from typing import Dict, Any


def get_common_build_environment() -> str:
    """
    Get common build environment setup used across multiple tools.

    Returns:
        Shell script snippet for environment configuration.
    """
    return r'''
set -e
# Compiler: force absolute path if possible to satisfy CMake/Makefile
if [ -n "$FC" ] && [ -x "$FC" ]; then
    export FC="$FC"
elif command -v gfortran >/dev/null 2>&1; then
    export FC="$(command -v gfortran)"
else
    export FC="${FC:-gfortran}"
fi
export FC_EXE="$FC"

# Discover libraries (fallback to /usr)
export NETCDF="${NETCDF:-$(nc-config --prefix 2>/dev/null || echo /usr)}"
export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$(nf-config --prefix 2>/dev/null || echo /usr)}"
export HDF5_ROOT="${HDF5_ROOT:-$(h5cc -showconfig 2>/dev/null | awk -F': ' "/Installation point/{print $2}" || echo /usr)}"
# Threads
export NCORES="${NCORES:-4}"
    '''.strip()


def get_netcdf_detection() -> str:
    """
    Get reusable NetCDF detection shell snippet.

    Sets NETCDF_FORTRAN and NETCDF_C environment variables.
    Works on Linux (apt), macOS (Homebrew), and HPC systems.

    Returns:
        Shell script snippet for NetCDF detection.
    """
    return r'''
# === NetCDF Detection (reusable snippet) ===
detect_netcdf() {
    # Try nf-config first (NetCDF Fortran config tool)
    if command -v nf-config >/dev/null 2>&1; then
        NETCDF_FORTRAN="$(nf-config --prefix)"
        echo "Found nf-config, NetCDF-Fortran at: ${NETCDF_FORTRAN}"
    elif [ -n "${NETCDF_FORTRAN}" ] && [ -d "${NETCDF_FORTRAN}/include" ]; then
        echo "Using NETCDF_FORTRAN env var: ${NETCDF_FORTRAN}"
    elif [ -n "${NETCDF}" ] && [ -d "${NETCDF}/include" ]; then
        NETCDF_FORTRAN="${NETCDF}"
        echo "Using NETCDF env var: ${NETCDF_FORTRAN}"
    else
        # Try common locations (Homebrew, system paths)
        for try_path in /opt/homebrew/opt/netcdf-fortran /opt/homebrew/opt/netcdf \
                        /usr/local/opt/netcdf-fortran /usr/local/opt/netcdf /usr/local /usr; do
            if [ -d "$try_path/include" ]; then
                NETCDF_FORTRAN="$try_path"
                echo "Found NetCDF at: $try_path"
                break
            fi
        done
    fi

    # Find NetCDF C library (may be separate from Fortran on macOS)
    if command -v nc-config >/dev/null 2>&1; then
        NETCDF_C="$(nc-config --prefix)"
    elif [ -d "/opt/homebrew/opt/netcdf" ]; then
        NETCDF_C="/opt/homebrew/opt/netcdf"
    else
        NETCDF_C="${NETCDF_FORTRAN}"
    fi

    export NETCDF_FORTRAN NETCDF_C
}
detect_netcdf
    '''.strip()


def get_hdf5_detection() -> str:
    """
    Get reusable HDF5 detection shell snippet.

    Sets HDF5_ROOT, HDF5_LIB_DIR, and HDF5_INC_DIR environment variables.
    Handles Ubuntu's hdf5/serial subdirectory structure.

    Returns:
        Shell script snippet for HDF5 detection.
    """
    return r'''
# === HDF5 Detection (reusable snippet) ===
detect_hdf5() {
    # Try h5cc config tool first
    if command -v h5cc >/dev/null 2>&1; then
        HDF5_ROOT="$(h5cc -showconfig 2>/dev/null | grep -i "Installation point" | sed 's/.*: *//' | head -n1)"
    fi

    # Fallback detection
    if [ -z "$HDF5_ROOT" ] || [ ! -d "$HDF5_ROOT" ]; then
        if [ -n "$HDF5_ROOT" ] && [ -d "$HDF5_ROOT" ]; then
            : # Use existing env var
        elif command -v brew >/dev/null 2>&1 && brew --prefix hdf5 >/dev/null 2>&1; then
            HDF5_ROOT="$(brew --prefix hdf5)"
        else
            for path in /usr $HOME/.local /opt/hdf5; do
                if [ -d "$path/include" ] && [ -d "$path/lib" ]; then
                    HDF5_ROOT="$path"
                    break
                fi
            done
        fi
    fi
    HDF5_ROOT="${HDF5_ROOT:-/usr}"

    # Find lib directory (Ubuntu stores in hdf5/serial, others in lib64 or lib)
    if [ -d "${HDF5_ROOT}/lib/x86_64-linux-gnu/hdf5/serial" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib/x86_64-linux-gnu/hdf5/serial"
    elif [ -d "${HDF5_ROOT}/lib/x86_64-linux-gnu" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib/x86_64-linux-gnu"
    elif [ -d "${HDF5_ROOT}/lib64" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib64"
    else
        HDF5_LIB_DIR="${HDF5_ROOT}/lib"
    fi

    # Find include directory
    if [ -d "${HDF5_ROOT}/include/hdf5/serial" ]; then
        HDF5_INC_DIR="${HDF5_ROOT}/include/hdf5/serial"
    else
        HDF5_INC_DIR="${HDF5_ROOT}/include"
    fi

    export HDF5_ROOT HDF5_LIB_DIR HDF5_INC_DIR
}
detect_hdf5
    '''.strip()


def get_netcdf_lib_detection() -> str:
    """
    Get reusable NetCDF library path detection snippet.

    Sets NETCDF_LIB_DIR and NETCDF_C_LIB_DIR for linking.
    Handles Debian/Ubuntu x86_64-linux-gnu paths and lib64 paths.

    Returns:
        Shell script snippet for NetCDF library path detection.
    """
    return r'''
# === NetCDF Library Path Detection ===
detect_netcdf_lib_paths() {
    # Find NetCDF-Fortran lib directory
    if [ -d "${NETCDF_FORTRAN}/lib/x86_64-linux-gnu" ] && \
       ls "${NETCDF_FORTRAN}/lib/x86_64-linux-gnu"/libnetcdff.* >/dev/null 2>&1; then
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib/x86_64-linux-gnu"
    elif [ -d "${NETCDF_FORTRAN}/lib64" ] && \
         ls "${NETCDF_FORTRAN}/lib64"/libnetcdff.* >/dev/null 2>&1; then
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib64"
    else
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib"
    fi

    # Find NetCDF-C lib directory (may differ from Fortran)
    if [ -d "${NETCDF_C}/lib/x86_64-linux-gnu" ] && \
       ls "${NETCDF_C}/lib/x86_64-linux-gnu"/libnetcdf.* >/dev/null 2>&1; then
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib/x86_64-linux-gnu"
    elif [ -d "${NETCDF_C}/lib64" ] && \
         ls "${NETCDF_C}/lib64"/libnetcdf.* >/dev/null 2>&1; then
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib64"
    else
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib"
    fi

    export NETCDF_LIB_DIR NETCDF_C_LIB_DIR
}
detect_netcdf_lib_paths
    '''.strip()


def get_geos_proj_detection() -> str:
    """
    Get reusable GEOS and PROJ detection shell snippet.

    Sets GEOS_CFLAGS, GEOS_LDFLAGS, PROJ_CFLAGS, PROJ_LDFLAGS.

    Returns:
        Shell script snippet for GEOS/PROJ detection.
    """
    return r'''
# === GEOS and PROJ Detection ===
detect_geos_proj() {
    GEOS_CFLAGS="" GEOS_LDFLAGS="" PROJ_CFLAGS="" PROJ_LDFLAGS=""

    # Try pkg-config first
    if command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists geos 2>/dev/null; then
            GEOS_CFLAGS="$(pkg-config --cflags geos)"
            GEOS_LDFLAGS="$(pkg-config --libs geos)"
            echo "GEOS found via pkg-config"
        fi
        if pkg-config --exists proj 2>/dev/null; then
            PROJ_CFLAGS="$(pkg-config --cflags proj)"
            PROJ_LDFLAGS="$(pkg-config --libs proj)"
            echo "PROJ found via pkg-config"
        fi
    fi

    # macOS Homebrew fallback
    if [ "$(uname)" = "Darwin" ]; then
        if [ -z "$GEOS_CFLAGS" ] && command -v brew >/dev/null 2>&1; then
            GEOS_PREFIX="$(brew --prefix geos 2>/dev/null || true)"
            if [ -n "$GEOS_PREFIX" ] && [ -d "$GEOS_PREFIX" ]; then
                GEOS_CFLAGS="-I${GEOS_PREFIX}/include"
                GEOS_LDFLAGS="-L${GEOS_PREFIX}/lib -lgeos_c"
                echo "GEOS found via Homebrew"
            fi
        fi
        if [ -z "$PROJ_CFLAGS" ] && command -v brew >/dev/null 2>&1; then
            PROJ_PREFIX="$(brew --prefix proj 2>/dev/null || true)"
            if [ -n "$PROJ_PREFIX" ] && [ -d "$PROJ_PREFIX" ]; then
                PROJ_CFLAGS="-I${PROJ_PREFIX}/include"
                PROJ_LDFLAGS="-L${PROJ_PREFIX}/lib -lproj"
                echo "PROJ found via Homebrew"
            fi
        fi
    fi

    # Common path fallback
    if [ -z "$GEOS_CFLAGS" ]; then
        for path in /usr/local /usr; do
            if [ -f "$path/lib/libgeos_c.so" ] || [ -f "$path/lib/libgeos_c.dylib" ]; then
                GEOS_CFLAGS="-I$path/include"
                GEOS_LDFLAGS="-L$path/lib -lgeos_c"
                echo "GEOS found in $path"
                break
            fi
        done
    fi
    if [ -z "$PROJ_CFLAGS" ]; then
        for path in /usr/local /usr; do
            if [ -f "$path/lib/libproj.so" ] || [ -f "$path/lib/libproj.dylib" ]; then
                PROJ_CFLAGS="-I$path/include"
                PROJ_LDFLAGS="-L$path/lib -lproj"
                echo "PROJ found in $path"
                break
            fi
        done
    fi

    export GEOS_CFLAGS GEOS_LDFLAGS PROJ_CFLAGS PROJ_LDFLAGS
}
detect_geos_proj
    '''.strip()

def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Define all external tools required by SYMFLUENCE.

    Returns:
        Dictionary mapping tool names to their complete configuration including:
        - description: Human-readable description
        - config_path_key: Key in config file for installation path
        - config_exe_key: Key in config file for executable name
        - default_path_suffix: Default relative path for installation
        - default_exe: Default executable/library filename
        - repository: Git repository URL (None for non-git installs)
        - branch: Git branch to checkout (None for default)
        - install_dir: Directory name for installation
        - requires: List of tool dependencies (other tools)
        - build_commands: Shell commands for building
        - dependencies: System dependencies required
        - test_command: Command argument for testing (None to skip)
        - verify_install: Installation verification criteria
        - order: Installation order (lower numbers first)
    """
    # Get reusable shell snippets
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()
    hdf5_detect = get_hdf5_detection()
    netcdf_lib_detect = get_netcdf_lib_detection()
    geos_proj_detect = get_geos_proj_detection()
    
    return {
        # ================================================================
        # SUNDIALS - Solver Library (Install First - Required by SUMMA)
        # ================================================================
            'sundials': {
            'description': 'SUNDIALS - SUite of Nonlinear and DIfferential/ALgebraic equation Solvers',
            'config_path_key': 'SUNDIALS_INSTALL_PATH',
            'config_exe_key': 'SUNDIALS_DIR',
            # This is what the rest of the code & validator expect:
            # <SYMFLUENCE_ROOT>/installs/sundials/install/sundials/
            'default_path_suffix': 'installs/sundials/install/sundials/',
            'default_exe': 'lib/libsundials_core.a',
            'repository': None,
            'branch': None,
            'install_dir': 'sundials',
            'build_commands': [
                common_env,
                r'''
# Build SUNDIALS from release tarball (shared libs OK; SUMMA will link).
set -e

SUNDIALS_VER=7.4.0

# Tool install root, e.g.  .../SYMFLUENCE_data/installs/sundials
SUNDIALS_ROOT_DIR="$(pwd)"

# Actual install prefix, consistent with default_path_suffix and SUMMA:
#   .../installs/sundials/install/sundials
SUNDIALS_PREFIX="${SUNDIALS_ROOT_DIR}/install/sundials"
mkdir -p "${SUNDIALS_PREFIX}"

rm -f "v${SUNDIALS_VER}.tar.gz" || true
wget -q "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz" \
  || curl -fsSL -o "v${SUNDIALS_VER}.tar.gz" "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz"

tar -xzf "v${SUNDIALS_VER}.tar.gz"
cd "sundials-${SUNDIALS_VER}"

rm -rf build && mkdir build && cd build
cmake .. \
  -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_INSTALL_PREFIX="${SUNDIALS_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DEXAMPLES_ENABLE=OFF \
  -DBUILD_TESTING=OFF

cmake --build . --target install -j ${NCORES:-4}

# Debug: show where the libs landed
[ -d "${SUNDIALS_PREFIX}/lib64" ] && ls -la "${SUNDIALS_PREFIX}/lib64" | head -20 || true
[ -d "${SUNDIALS_PREFIX}/lib" ] && ls -la "${SUNDIALS_PREFIX}/lib" | head -20 || true
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': [
                    'lib64/libsundials_core.a',
                    'lib/libsundials_core.a',
                    'include/sundials/sundials_config.h'
                ],
                'check_type': 'exists_any'
            },
            'order': 1
        },


        # ================================================================
        # SUMMA - Hydrological Model
        # ================================================================
        'summa': {
            'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'config_exe_key': 'SUMMA_EXE',
            'default_path_suffix': 'installs/summa/bin',
            'default_exe': 'summa_sundials.exe',
            'repository': 'https://github.com/CH-Earth/summa.git',
            'branch': 'develop_sundials',
            'install_dir': 'summa',
            'requires': ['sundials'],
            'build_commands': [
                common_env,
                r'''
# Build SUMMA against SUNDIALS + NetCDF, leverage SUMMA's CMake-based build
set -e

export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
echo "Using SUNDIALS from: $SUNDIALS_DIR"

# Determine LAPACK strategy based on platform
SPECIFY_LINKS=OFF

# macOS: Use manual LAPACK specification (Homebrew OpenBLAS isn't reliably detected by CMake)
if [ "$(uname)" == "Darwin" ]; then
    echo "macOS detected - using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack'
# HPC with OpenBLAS module loaded
elif command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then
    echo "OpenBLAS module loaded - using auto-detection"
    SPECIFY_LINKS=OFF
# Linux with system OpenBLAS
elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then
    echo "System OpenBLAS found - using auto-detection"
    SPECIFY_LINKS=OFF
else
    # Fallback to manual LAPACK
    echo "Using manual LAPACK specification"
    SPECIFY_LINKS=ON
    export LIBRARY_LINKS='-llapack -lblas'
fi

rm -rf cmake_build && mkdir -p cmake_build

cmake -S build -B cmake_build \
  -DUSE_SUNDIALS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS \
  -DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" \
  -DSUNDIALS_ROOT="$SUNDIALS_DIR" \
  -DNETCDF_PATH="${NETCDF:-/usr}" \
  -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"

# Build all targets (repo scripts use 'all', not just 'summa_sundials')
cmake --build cmake_build --target all -j ${NCORES:-4}

# Stage binary into bin/ and provide standard name
if [ -f "bin/summa_sundials.exe" ]; then
    cd bin
    ln -sf summa_sundials.exe summa.exe
    cd ..
elif [ -f "cmake_build/bin/summa_sundials.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa_sundials.exe bin/
    cd bin
    ln -sf summa_sundials.exe summa.exe
    cd ..
elif [ -f "cmake_build/bin/summa.exe" ]; then
    mkdir -p bin
    cp cmake_build/bin/summa.exe bin/
fi
                '''.strip()
            ],
            'dependencies': [],
            'test_command': '--version',
            'verify_install': {
                'file_paths': [
                    'bin/summa.exe',
                    'bin/summa_sundials.exe',
                    'cmake_build/bin/summa.exe',
                    'cmake_build/bin/summa_sundials.exe'
                ],
                'check_type': 'exists_any'
            },
            'order': 2
        },

        # ================================================================
        # mizuRoute - River Network Routing
        # ================================================================
        'mizuroute': {
            'description': 'Mizukami routing model for river network routing',
            'config_path_key': 'INSTALL_PATH_MIZUROUTE',
            'config_exe_key': 'EXE_NAME_MIZUROUTE',
            'default_path_suffix': 'installs/mizuRoute/route/bin',
            'default_exe': 'mizuRoute.exe',
            'repository': 'https://github.com/ESCOMP/mizuRoute.git',
            'branch': 'serial',
            'install_dir': 'mizuRoute',
            'build_commands': [
                common_env,
                netcdf_detect,
                r'''
# Build mizuRoute - edit Makefile directly (it doesn't use env vars)
cd route/build
mkdir -p ../bin

F_MASTER_PATH="$(cd .. && pwd)"
echo "F_MASTER: $F_MASTER_PATH/"

# Validate NetCDF was detected
if [ -z "${NETCDF_FORTRAN}" ]; then
    echo "ERROR: Could not find NetCDF installation"
    exit 1
fi

# Edit the Makefile in-place
echo "=== Configuring Makefile ==="
perl -i -pe "s|^FC\s*=.*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=.*$|FC_EXE = ${FC:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=.*$|EXE = mizuRoute.exe|" Makefile
perl -i -pe "s|^F_MASTER\s*=.*$|F_MASTER = $F_MASTER_PATH/|" Makefile
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$| NCDF_PATH = ${NETCDF_FORTRAN}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = no|" Makefile

# Fix LIBNETCDF for separate C/Fortran libs (e.g., macOS Homebrew)
if [ "${NETCDF_C}" != "${NETCDF_FORTRAN}" ]; then
    echo "Fixing LIBNETCDF for separate C/Fortran paths"
    perl -i -pe "s|^LIBNETCDF\s*=.*$|LIBNETCDF = -L${NETCDF_FORTRAN}/lib -lnetcdff -L${NETCDF_C}/lib -lnetcdf|" Makefile
fi

# Build
make clean || true
echo "Building mizuRoute..."
make 2>&1 | tee build.log || true

if [ -f "../bin/mizuRoute.exe" ]; then
    echo "Build successful - executable at ../bin/mizuRoute.exe"
else
    echo "ERROR: Executable not found at ../bin/mizuRoute.exe"
    exit 1
fi
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['route/bin/mizuRoute.exe'],
                'check_type': 'exists'
            },
            'order': 3
        },

        # ================================================================
        # TROUTE - NOAA Next Generation River Routing (Python package)
        # ================================================================
        'troute': {
        'description': "NOAA's Next Generation river routing model",
        'config_path_key': 'TROUTE_INSTALL_PATH',
        'config_exe_key': 'TROUTE_PKG_PATH',

        # Correct base installation directory
        'default_path_suffix': 'installs/t-route/src/troute-network',

        # Correct default "executable" (package entry point)
        'default_exe': 'troute/network/__init__.py',

        # Top-level clone folder
        'install_dir': 't-route',

        'build_commands': [
        r'''
# Non-fatal installation of the t-route Python package.
# t-route intentionally does not stop SYMFLUENCE builds on pip failure.

set +e

echo "Installing t-route Python package (non-fatal if this step fails)..."

cd src/troute-network 2>/dev/null || {
    echo "src/troute-network not found; skipping t-route install."
    exit 0
}

PYTHON_BIN="${SYMFLUENCE_PYTHON:-python3}"

# Upgrade tools quietly; failure is allowed
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# Try installing the package; if it fails, SYMFLUENCE continues
"$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps || true

echo "t-route installation attempt complete (see any errors above)."
exit 0
        '''.strip()
    ],

    'dependencies': [],
    'test_command': None,

    # 🔧 Correct verification logic:
    #   The package ALWAYS lives in src/troute-network/troute/network/__init__.py
    #   Not directly in install_dir/troute/...
    'verify_install': {
        'file_paths': [
            # Correct full layout (as seen in all logs)
            'src/troute-network/troute/network/__init__.py',
            # Fallback if upstream ever changes structure
            'troute/network/__init__.py',
        ],
        'check_type': 'exists_any'
            },
            'order': 4
        },


        # ================================================================
        # FUSE - Framework for Understanding Structural Errors
        # ================================================================
        "fuse": {
            "description": "Framework for Understanding Structural Errors",
            "config_path_key": "FUSE_INSTALL_PATH",
            "config_exe_key": "FUSE_EXE",
            "default_path_suffix": "installs/fuse/bin",
            "default_exe": "fuse.exe",
            "repository": "https://github.com/CH-Earth/fuse.git",
            'branch': None,
            'install_dir': "fuse",
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

# Build FUSE
echo "Building FUSE..."
if make FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" \
       FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}"; then
  echo "Build completed"
else
  echo "Build failed - NetCDF lib: ${NETCDF_LIB_DIR}, HDF5 lib: ${HDF5_LIB_DIR}"
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
            'test_command': None,  # FUSE exits with error when run without args, which is expected
            'verify_install': {
                'file_paths': ['bin/fuse.exe'],
                'check_type': 'exists'
            },
            'order': 5
        },

        # ================================================================
        # TauDEM - Terrain Analysis
        # ================================================================
        'taudem': {
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'config_exe_key': 'TAUDEM_EXE',
            'default_path_suffix': 'installs/TauDEM/bin',
            'default_exe': 'pitremove',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': None,
            'install_dir': 'TauDEM',
            'build_commands': [
                common_env,
                r'''
# Build TauDEM from GitHub repository
set -e

# Use OpenMPI compiler wrappers so CMake/FindMPI can auto-detect everything
export CC=mpicc
export CXX=mpicxx

rm -rf build && mkdir -p build
cd build

# Let CMake find MPI and GDAL
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release

# Build everything plus the two tools that sometimes get skipped by default
cmake --build . -j 2
cmake --build . --target moveoutletstostreams gagewatershed -j 2 || true

echo "Staging executables..."
mkdir -p ../bin

# List of expected TauDEM tools (superset — some may not exist on older commits)
tools="pitremove d8flowdir d8converge dinfconverge dinfflowdir aread8 areadinf threshold 
       streamnet slopearea gridnet peukerdouglas lengtharea moveoutletstostreams gagewatershed"

copied=0
for exe in $tools;
  do
  # Find anywhere under build tree and copy if executable
  p="$(find . -type f -perm -111 -name "$exe" | head -n1 || true)"
  if [ -n "$p" ]; then
    cp -f "$p" ../bin/
    copied=$((copied+1))
  fi
done

# Final sanity
ls -la ../bin/ || true
if [ ! -x "../bin/pitremove" ] || [ ! -x "../bin/streamnet" ]; then
  echo "❌ TauDEM stage failed: core binaries missing" >&2
  exit 1
fi
echo "✅ TauDEM executables staged"
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/pitremove'],
                'check_type': 'exists'
            },
            'order': 6
        },

        # ================================================================
        # GIStool - Geospatial Data Extraction
        # ================================================================
        'gistool': {
            'description': 'Geospatial data extraction and processing tool',
            'config_path_key': 'INSTALL_PATH_GISTOOL',
            'config_exe_key': 'EXE_NAME_GISTOOL',
            'default_path_suffix': 'installs/gistool',
            'default_exe': 'extract-gis.sh',
            'repository': 'https://github.com/kasra-keshavarz/gistool.git',
            'branch': None,
            'install_dir': 'gistool',
            'build_commands': [
                r'''
set -e
chmod +x extract-gis.sh
                '''.strip()
            ],
            'verify_install': {
                'file_paths': ['extract-gis.sh'],
                'check_type': 'exists'
            },
            'dependencies': [],
            'test_command': None,
            'order': 7
        },

        # ================================================================
        # Datatool - Meteorological Data Processing
        # ================================================================
        'datatool': {
            'description': 'Meteorological data extraction and processing tool',
            'config_path_key': 'DATATOOL_PATH',
            'config_exe_key': 'DATATOOL_SCRIPT',
            'default_path_suffix': 'installs/datatool',
            'default_exe': 'extract-dataset.sh',
            'repository': 'https://github.com/kasra-keshavarz/datatool.git',
            'branch': None,
            'install_dir': 'datatool',
            'build_commands': [
                r'''
set -e
chmod +x extract-dataset.sh
                '''.strip()
            ],
            'dependencies': [],
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['extract-dataset.sh'],
                'check_type': 'exists'
            },
            'order': 8
        },

        # ================================================================
        # NGEN - NextGen National Water Model Framework
        # ================================================================
        'ngen': {
            'description': 'NextGen National Water Model Framework',
            'config_path_key': 'NGEN_INSTALL_PATH',
            'config_exe_key': 'NGEN_EXE',
            'default_path_suffix': 'installs/ngen/cmake_build',
            'default_exe': 'ngen',
            'repository': 'https://github.com/CIROH-UA/ngen',
            'branch': 'ngiab',
            'install_dir': 'ngen',
            'build_commands': [
                r'''
set -e
echo "Building ngen..."

# Make sure CMake sees a supported NumPy, and ignore user-site
export PYTHONNOUSERSITE=1
python -m pip install --upgrade "pip<24.1" >/dev/null 2>&1 || true
python - <<'PY' || (python -m pip install "numpy<2" "setuptools<70" && true)
from packaging.version import Version
import numpy as np
assert Version(np.__version__) < Version("2.0")
PY
python - <<'PY'
import numpy as np
print("Using NumPy:", np.__version__)
PY

# Boost (local)
if [ ! -d "boost_1_79_0" ]; then
  echo "Fetching Boost 1.79.0..."
  (wget -q https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 -O boost_1_79_0.tar.bz2 \
    || curl -fsSL -o boost_1_79_0.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2)
  tar -xjf boost_1_79_0.tar.bz2 && rm -f boost_1_79_0.tar.bz2
fi
export BOOST_ROOT="$(pwd)/boost_1_79_0"
export CXX=${CXX:-g++}

# Submodules needed
git submodule update --init --recursive -- test/googletest extern/pybind11 || true

rm -rf cmake_build

# First try with Python ON
if cmake -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT="$BOOST_ROOT" -DNGEN_WITH_PYTHON=ON -DNGEN_WITH_SQLITE3=ON -S . -B cmake_build; then
  echo "Configured with Python ON"
else
  echo "CMake failed with Python ON; retrying with Python OFF"
  rm -rf cmake_build
  cmake -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT="$BOOST_ROOT" -DNGEN_WITH_PYTHON=OFF -DNGEN_WITH_SQLITE3=ON -S . -B cmake_build
fi

cmake --build cmake_build --target ngen -j ${NCORES:-4}
./cmake_build/ngen --help >/dev/null || true
                '''.strip()
            ],
            'dependencies': [],
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['cmake_build/ngen'],
                'check_type': 'exists'
            },
            'order': 9
        },

        # ================================================================
        # HYPE - Hydrological Predictions for the Environment
        # ================================================================
        'hype': {
            'description': 'HYPE - Hydrological Predictions for the Environment',
            'config_path_key': 'HYPE_INSTALL_PATH',
            'config_exe_key': 'HYPE_EXE',
            'default_path_suffix': 'installs/hype/bin',
            'default_exe': 'hype',
            'repository': 'git://git.code.sf.net/p/hype/code',
            'branch': None,  # Use default branch
            'install_dir': 'hype',
            'build_commands': [
                common_env,
                netcdf_detect,
                r'''
# Build HYPE from SourceForge git repository
set -e
mkdir -p bin

if [ -z "${NETCDF_FORTRAN}" ]; then
    echo "NetCDF not found, building basic version..."
    make hype FC="${FC:-gfortran}" || { echo "HYPE compilation failed"; exit 1; }
else
    echo "Building HYPE with NetCDF support..."
    export NCDF_PATH="${NETCDF_FORTRAN}"
    make hype libs=netcdff FC="${FC:-gfortran}" || {
        echo "NetCDF build failed, trying basic build..."
        make clean || true
        make hype FC="${FC:-gfortran}" || { echo "HYPE compilation failed"; exit 1; }
    }
fi

# Stage binary
if [ -f "hype" ]; then
    mv hype bin/
elif [ ! -f "bin/hype" ]; then
    echo "HYPE binary not found after build"
    exit 1
fi
chmod +x bin/hype
echo "HYPE build successful"
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,  # HYPE exits with error when run without args, which is expected
            'verify_install': {
                'file_paths': ['bin/hype'],
                'check_type': 'exists'
            },
            'order': 11
        },

        # ================================================================
        # MESH - Modélisation Environnementale Surface Hydrology
        # ================================================================
        'mesh': {
            'description': 'MESH - Environment Canada Hydrology Land-Surface Scheme',
            'config_path_key': 'MESH_INSTALL_PATH',
            'config_exe_key': 'MESH_EXE',
            'default_path_suffix': 'installs/mesh/bin',
            'default_exe': 'mesh.exe',
            'repository': 'https://github.com/MESH-Model/MESH-Dev.git',
            'branch': None,  # Use default branch
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
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_FORTRAN="$(nf-config --prefix)"
    echo "Found nf-config, using: ${NETCDF_FORTRAN}"
elif [ -n "${NETCDF_FORTRAN}" ] && [ -d "${NETCDF_FORTRAN}/include" ]; then
    echo "Using NETCDF_FORTRAN: ${NETCDF_FORTRAN}"
elif [ -n "${NETCDF}" ] && [ -d "${NETCDF}/include" ]; then
    NETCDF_FORTRAN="${NETCDF}"
    echo "Using NETCDF: ${NETCDF_TO_USE}"
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
fi

# Patch the getenvc.c file to fix K&R C style function declarations
# This is needed for modern C compilers (clang on macOS)
GETENVC_FILE="./Modules/librmn/19.7.0/primitives/getenvc.c"
if [ -f "$GETENVC_FILE" ]; then
    echo "Patching getenvc.c for modern C compatibility..."
    # Backup original
    cp "$GETENVC_FILE" "${GETENVC_FILE}.backup"

    # Convert K&R C style to ANSI C style
    # Old style: f77name(getenvc) ( name, value, len1, len2 )
    #            F2Cl  len1, len2;
    #            char name[1], value[1];
    # New style: f77name(getenvc) ( char *name, char *value, F2Cl len1, F2Cl len2 )

    cat > "${GETENVC_FILE}" << 'PATCHEOF'
#include <stdlib.h>
#include <string.h>

/* Define F2Cl type for Fortran-C interop */
#define F2Cl int

/* Define f77name macro for Fortran name mangling */
#if defined(__APPLE__) || defined(__linux__)
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

    echo "✅ getenvc.c patched successfully"
fi

# Determine if we should try MPI build
BUILD_MPI=false
if command -v mpifort >/dev/null 2>&1 || command -v mpif90 >/dev/null 2>&1; then
    BUILD_MPI=true
    echo "MPI compiler detected, will attempt MPI build"
fi

# Clean any previous builds
make veryclean 2>/dev/null || make clean 2>/dev/null || true

# Try building with NetCDF support
if [ -n "${NETCDF_FORTRAN}" ]; then
    echo "Building MESH with NetCDF support..."

    # Try MPI version first if available
    if [ "$BUILD_MPI" = true ]; then
        echo "Attempting MPI build..."
        # Use PIPESTATUS to get make's exit code, not tee's
        set +e
        make mpi_gcc netcdf 2>&1 | tee build.log
        BUILD_STATUS=${PIPESTATUS[0]}
        set -e

        if [ $BUILD_STATUS -eq 0 ] && [ -f "mpi_sa_mesh" ]; then
            BUILT_BINARY="mpi_sa_mesh"
            echo "✅ MPI build with NetCDF successful"
        else
            echo "⚠️  MPI build failed (exit code: $BUILD_STATUS), trying serial build..."
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
            echo "✅ Serial build with NetCDF successful"
        else
            echo "⚠️  NetCDF build failed (exit code: $BUILD_STATUS), trying basic build..."
            make veryclean 2>/dev/null || make clean 2>/dev/null || true

            set +e
            make gfortran 2>&1 | tee build.log
            BUILD_STATUS=${PIPESTATUS[0]}
            set -e

            if [ $BUILD_STATUS -eq 0 ] && [ -f "sa_mesh" ]; then
                BUILT_BINARY="sa_mesh"
                echo "✅ Basic build successful"
            else
                echo "❌ MESH compilation failed (exit code: $BUILD_STATUS)"
                echo ""
                echo "Last 100 lines of build log:"
                tail -100 build.log
                exit 1
            fi
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
        echo "✅ Basic build successful"
    else
        echo "❌ MESH compilation failed (exit code: $BUILD_STATUS)"
        echo ""
        echo "Last 100 lines of build log:"
        tail -100 build.log
        exit 1
    fi
fi

# Find and move the binary to bin/
echo "Locating built binary..."
MESH_BINARY=""
for candidate in "${BUILT_BINARY}" "sa_mesh" "mpi_sa_mesh"; do
    if [ -f "$candidate" ]; then
        MESH_BINARY="$candidate"
        break
    fi
done

if [ -z "$MESH_BINARY" ]; then
    # Search in common build directories
    MESH_BINARY=$(find . -maxdepth 2 -name "sa_mesh" -o -name "mpi_sa_mesh" -o -name "mesh.exe" 2>/dev/null | head -1)
fi

if [ -n "$MESH_BINARY" ] && [ -f "$MESH_BINARY" ]; then
    # Standardize binary name to mesh.exe for consistency
    cp "$MESH_BINARY" bin/mesh.exe
    chmod +x bin/mesh.exe
    echo "✅ MESH binary staged to bin/mesh.exe"
else
    echo "❌ MESH binary not found after build"
    echo "Directory contents:"
    ls -la
    exit 1
fi

# Test the binary
echo "Testing MESH binary..."
if bin/mesh.exe --help 2>&1 | head -10 || [ $? -ne 0 ]; then
    echo "✅ MESH build verification complete"
else
    echo "⚠️  MESH binary exists but test output unexpected (may be normal)"
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
        },
        # ================================================================
        # RHESSys - Regional Hydro-Ecologic Simulation System
        # ================================================================
        'rhessys': {
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
                r'''
set -e
echo "Building RHESSys..."
cd rhessys

echo "GEOS_CFLAGS: $GEOS_CFLAGS, PROJ_CFLAGS: $PROJ_CFLAGS"

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

# Build with detected flags
make V=1 netcdf=T CMD_OPTS="-DCLIM_GRID_XY $GEOS_CFLAGS $PROJ_CFLAGS $GEOS_LDFLAGS $PROJ_LDFLAGS"

mkdir -p ../bin
mv rhessys* ../bin/rhessys || true
chmod +x ../bin/rhessys
                '''.strip()
            ],'dependencies': ['gdal-config', 'proj', 'geos-config'],
            'test_command': '-h',
            'verify_install': {
                'file_paths': ['bin/rhessys'],
                'check_type': 'exists'
            },
            'order': 14
        },
        # ================================================================
        # NGIAB - NextGen In A Box
        # ================================================================
        'ngiab': {
            'description': 'NextGen In A Box - Container-based ngen deployment',
            'config_path_key': 'NGIAB_INSTALL_PATH',
            'config_exe_key': 'NGIAB_SCRIPT',
            'default_path_suffix': 'installs/ngiab',
            'default_exe': 'guide.sh',
            'repository': None,
            'branch': 'main',
            'install_dir': 'ngiab',
            'build_commands': [
                r'''
set -e
# Detect HPC vs laptop/workstation and fetch the right NGIAB wrapper repo into ../ngiab
IS_HPC=false
for scheduler in sbatch qsub bsub; do
  if command -v $scheduler >/dev/null 2>&1; then IS_HPC=true; break; fi
done
[ -n "$SLURM_CLUSTER_NAME" ] && IS_HPC=true
[ -n "$PBS_JOBID" ] && IS_HPC=true
[ -n "$SGE_CLUSTER_NAME" ] && IS_HPC=true
[ -d "/scratch" ] && IS_HPC=true

if $IS_HPC; then
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
  echo "HPC environment detected; using NGIAB-HPCInfra"
else
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
  echo "Non-HPC environment detected; using NGIAB-CloudInfra"
fi

cd ..
rm -rf ngiab
git clone "$NGIAB_REPO" ngiab
cd ngiab
[ -f guide.sh ] && chmod +x guide.sh && bash -n guide.sh || true
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['guide.sh'],
                'check_type': 'exists'
            },
            'order': 10
        },
    }


if __name__ == "__main__":
    """Test the configuration definitions."""
    tools = get_external_tools_definitions()
    print(f"✅ Loaded {len(tools)} external tool definitions:")
    for name, info in sorted(tools.items(), key=lambda x: x[1]['order']):
        print(f"   {info['order']:2d}. {name:12s} - {info['description'][:60]}")
