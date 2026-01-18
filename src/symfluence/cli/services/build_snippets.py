"""
Shared shell snippets for external tool builds.

This module contains reusable shell script fragments for detecting
system libraries (NetCDF, HDF5, GEOS, PROJ) across different platforms.
These are lightweight (no heavy dependencies) and can be safely imported
by the CLI without loading pandas, xarray, etc.
"""

from typing import Dict


def get_common_build_environment() -> str:
    """
    Get common build environment setup used across multiple tools.

    Returns:
        Shell script snippet for environment configuration.
    """
    return r'''
set -e

# ================================================================
# 2i2c / JupyterHub Compiler Fix
# ================================================================
# Detect if conda's gcc is broken (common in 2i2c/JupyterHub environments)
# and fall back to system compilers if so.
detect_and_fix_compilers() {
    local use_system_compilers=false

    # Check for 2i2c environment indicators
    if [ -d "/srv/conda/envs/notebook" ]; then
        # Test if conda gcc can link a simple program
        local test_file="/tmp/compiler_test_$$.c"
        echo 'int main() { return 0; }' > "$test_file"

        if [ -x "/srv/conda/envs/notebook/bin/gcc" ]; then
            if ! /srv/conda/envs/notebook/bin/gcc "$test_file" -o /tmp/compiler_test_$$ 2>/dev/null; then
                echo "2i2c: Conda gcc is broken, using system compilers"
                use_system_compilers=true
            fi
        fi
        rm -f "$test_file" /tmp/compiler_test_$$
    fi

    # Apply system compilers if needed
    if [ "$use_system_compilers" = true ] || [ "${SYMFLUENCE_USE_SYSTEM_COMPILERS:-}" = "true" ]; then
        [ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc
        [ -x /usr/bin/g++ ] && export CXX=/usr/bin/g++
        echo "Using system compilers: CC=$CC, CXX=${CXX:-not set}"
    fi
}
detect_and_fix_compilers

# ================================================================
# Fortran Compiler Detection
# ================================================================
# Compiler: force absolute path if possible to satisfy CMake/Makefile
if [ -n "$FC" ] && [ -x "$FC" ]; then
    export FC="$FC"
elif command -v gfortran >/dev/null 2>&1; then
    export FC="$(command -v gfortran)"
else
    export FC="${FC:-gfortran}"
fi
export FC_EXE="$FC"

# ================================================================
# Library Discovery
# ================================================================
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


def get_udunits2_detection_and_build() -> str:
    """
    Get reusable UDUNITS2 detection and build-from-source snippet.

    Sets UDUNITS2_DIR, UDUNITS2_INCLUDE_DIR, UDUNITS2_LIBRARY environment variables.
    If UDUNITS2 is not found system-wide, builds it from source in a local directory.

    Returns:
        Shell script snippet for UDUNITS2 detection and building.
    """
    return r'''
# === UDUNITS2 Detection and Build ===
detect_or_build_udunits2() {
    UDUNITS2_FOUND=false

    # Try pkg-config first (system install)
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists udunits2 2>/dev/null; then
        UDUNITS2_DIR="$(pkg-config --variable=prefix udunits2)"
        UDUNITS2_INCLUDE_DIR="$(pkg-config --variable=includedir udunits2)"
        UDUNITS2_LIBRARY="$(pkg-config --variable=libdir udunits2)/libudunits2.so"
        echo "Found UDUNITS2 via pkg-config at: ${UDUNITS2_DIR}"
        UDUNITS2_FOUND=true
    fi

    # Try common system locations
    if [ "$UDUNITS2_FOUND" = false ]; then
        for try_path in /usr /usr/local /opt/udunits2 $HOME/.local; do
            if [ -f "$try_path/include/udunits2.h" ] && \
               ([ -f "$try_path/lib/libudunits2.so" ] || [ -f "$try_path/lib/libudunits2.dylib" ] || [ -f "$try_path/lib/libudunits2.a" ]); then
                UDUNITS2_DIR="$try_path"
                UDUNITS2_INCLUDE_DIR="$try_path/include"
                if [ -f "$try_path/lib/libudunits2.so" ]; then
                    UDUNITS2_LIBRARY="$try_path/lib/libudunits2.so"
                elif [ -f "$try_path/lib/libudunits2.dylib" ]; then
                    UDUNITS2_LIBRARY="$try_path/lib/libudunits2.dylib"
                else
                    UDUNITS2_LIBRARY="$try_path/lib/libudunits2.a"
                fi
                echo "Found UDUNITS2 at: $try_path"
                UDUNITS2_FOUND=true
                break
            fi
        done
    fi

    # If not found, build from source
    if [ "$UDUNITS2_FOUND" = false ]; then
        echo "UDUNITS2 not found system-wide, building from source..."

        # Save original directory before building
        UDUNITS2_ORIGINAL_DIR="$(pwd)"

        UDUNITS2_VERSION="2.2.28"
        UDUNITS2_BUILD_DIR="${UDUNITS2_ORIGINAL_DIR}/udunits2_build"
        UDUNITS2_INSTALL_DIR="${UDUNITS2_ORIGINAL_DIR}/udunits2"

        # Check if already built locally
        if [ -f "${UDUNITS2_INSTALL_DIR}/include/udunits2.h" ] && \
           ([ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so" ] || [ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.a" ]); then
            echo "Using previously built UDUNITS2 at: ${UDUNITS2_INSTALL_DIR}"
        else
            # Download and build UDUNITS2
            mkdir -p "${UDUNITS2_BUILD_DIR}"
            cd "${UDUNITS2_BUILD_DIR}"

            if [ ! -f "udunits-${UDUNITS2_VERSION}.tar.gz" ]; then
                echo "Downloading UDUNITS2 ${UDUNITS2_VERSION}..."
                wget -q "https://downloads.unidata.ucar.edu/udunits/${UDUNITS2_VERSION}/udunits-${UDUNITS2_VERSION}.tar.gz" \
                  || curl -fsSL -o "udunits-${UDUNITS2_VERSION}.tar.gz" "https://downloads.unidata.ucar.edu/udunits/${UDUNITS2_VERSION}/udunits-${UDUNITS2_VERSION}.tar.gz"
            fi

            if [ ! -d "udunits-${UDUNITS2_VERSION}" ]; then
                echo "Extracting UDUNITS2..."
                tar -xzf "udunits-${UDUNITS2_VERSION}.tar.gz"
            fi

            cd "udunits-${UDUNITS2_VERSION}"
            echo "Configuring UDUNITS2..."
            ./configure --prefix="${UDUNITS2_INSTALL_DIR}" --disable-shared --enable-static

            echo "Building UDUNITS2..."
            make -j ${NCORES:-4}

            echo "Installing UDUNITS2 to ${UDUNITS2_INSTALL_DIR}..."
            make install

            # Return to original directory
            cd "${UDUNITS2_ORIGINAL_DIR}"

            echo "UDUNITS2 built successfully"
        fi

        UDUNITS2_DIR="${UDUNITS2_INSTALL_DIR}"
        UDUNITS2_INCLUDE_DIR="${UDUNITS2_INSTALL_DIR}/include"
        if [ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so" ]; then
            UDUNITS2_LIBRARY="${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so"
        else
            UDUNITS2_LIBRARY="${UDUNITS2_INSTALL_DIR}/lib/libudunits2.a"
        fi
    fi

    export UDUNITS2_DIR UDUNITS2_INCLUDE_DIR UDUNITS2_LIBRARY

    # Also set CMAKE-specific variables
    export UDUNITS2_ROOT="$UDUNITS2_DIR"
    export CMAKE_PREFIX_PATH="${UDUNITS2_DIR}:${CMAKE_PREFIX_PATH:-}"

    echo "UDUNITS2 configuration:"
    echo "  UDUNITS2_DIR: ${UDUNITS2_DIR}"
    echo "  UDUNITS2_INCLUDE_DIR: ${UDUNITS2_INCLUDE_DIR}"
    echo "  UDUNITS2_LIBRARY: ${UDUNITS2_LIBRARY}"
}
detect_or_build_udunits2
    '''.strip()


def get_all_snippets() -> Dict[str, str]:
    """
    Return all snippets as a dictionary for easy access.

    Returns:
        Dictionary mapping snippet names to their shell script content.
    """
    return {
        'common_env': get_common_build_environment(),
        'netcdf_detect': get_netcdf_detection(),
        'hdf5_detect': get_hdf5_detection(),
        'netcdf_lib_detect': get_netcdf_lib_detection(),
        'geos_proj_detect': get_geos_proj_detection(),
        'udunits2_detect_build': get_udunits2_detection_and_build(),
    }
