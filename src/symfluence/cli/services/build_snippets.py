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
# 2i2c / JupyterHub Compiler Configuration
# ================================================================
# Respect pre-configured compilers for ABI compatibility with conda libraries.
# The symfluence shell script sets CC/CXX to conda compilers when available.
configure_compilers() {
    # If CC/CXX are already set to conda compilers, trust them
    # (symfluence --install sets these for ABI compatibility)
    if [ -n "$CC" ] && [[ "$CC" == *conda* ]]; then
        echo "Using pre-configured conda compiler: CC=$CC"
        [ -n "$CXX" ] && echo "  CXX=$CXX"
        return 0
    fi

    # Only use system compilers if explicitly requested
    if [ "${SYMFLUENCE_USE_SYSTEM_COMPILERS:-}" = "true" ]; then
        [ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc
        [ -x /usr/bin/g++ ] && export CXX=/usr/bin/g++
        echo "Using system compilers (requested via SYMFLUENCE_USE_SYSTEM_COMPILERS)"
        echo "  CC=$CC, CXX=${CXX:-not set}"
        return 0
    fi

    # If no compiler is set but we're in a conda env, try to find conda compilers
    if [ -n "$CONDA_PREFIX" ] && [ -z "$CC" ]; then
        local conda_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        local conda_gxx="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        if [ -x "$conda_gcc" ]; then
            export CC="$conda_gcc"
            [ -x "$conda_gxx" ] && export CXX="$conda_gxx"
            echo "Auto-detected conda compilers: CC=$CC"
        fi
    fi

    # Report current compiler configuration
    if [ -n "$CC" ]; then
        echo "Compiler configuration: CC=$CC"
        [ -n "$CXX" ] && echo "  CXX=$CXX"
    fi
}
configure_compilers

# ================================================================
# Fortran Compiler Detection
# ================================================================
# Look for conda gfortran first (for ABI compatibility), then system gfortran
configure_fortran() {
    # Already set and valid
    if [ -n "$FC" ] && [ -x "$FC" ]; then
        echo "Using FC=$FC"
        export FC_EXE="$FC"
        return 0
    fi

    # Try conda gfortran first (from compilers package)
    if [ -n "$CONDA_PREFIX" ]; then
        local conda_fc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
        if [ -x "$conda_fc" ]; then
            export FC="$conda_fc"
            export FC_EXE="$FC"
            echo "Using conda Fortran compiler: FC=$FC"
            return 0
        fi
    fi

    # Fall back to system gfortran
    if command -v gfortran >/dev/null 2>&1; then
        export FC="$(command -v gfortran)"
        export FC_EXE="$FC"
        echo "Using system Fortran compiler: FC=$FC"
        return 0
    fi

    # Last resort
    export FC="${FC:-gfortran}"
    export FC_EXE="$FC"
    echo "Warning: gfortran not found, set FC=$FC"
}
configure_fortran

# ================================================================
# Library Discovery
# ================================================================
# Discover libraries - prefer conda prefix if available
configure_libraries() {
    # NetCDF: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/nc-config" ]; then
        export NETCDF="$CONDA_PREFIX"
        echo "Using conda NetCDF: $NETCDF"
    else
        export NETCDF="${NETCDF:-$(nc-config --prefix 2>/dev/null || echo /usr)}"
    fi

    # NetCDF-Fortran: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/nf-config" ]; then
        export NETCDF_FORTRAN="$CONDA_PREFIX"
        echo "Using conda NetCDF-Fortran: $NETCDF_FORTRAN"
    else
        export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$(nf-config --prefix 2>/dev/null || echo /usr)}"
    fi

    # HDF5: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ] && ls "$CONDA_PREFIX/lib"/libhdf5* >/dev/null 2>&1; then
        export HDF5_ROOT="$CONDA_PREFIX"
        echo "Using conda HDF5: $HDF5_ROOT"
    else
        export HDF5_ROOT="${HDF5_ROOT:-$(h5cc -showconfig 2>/dev/null | awk -F': ' "/Installation point/{print \$2}" || echo /usr)}"
    fi
}
configure_libraries

# Threads
export NCORES="${NCORES:-4}"
    '''.strip()


def get_netcdf_detection() -> str:
    """
    Get reusable NetCDF detection shell snippet.

    Sets NETCDF_FORTRAN and NETCDF_C environment variables.
    Works on Linux (apt), macOS (Homebrew), conda environments, and HPC systems.

    Returns:
        Shell script snippet for NetCDF detection.
    """
    return r'''
# === NetCDF Detection (reusable snippet) ===
detect_netcdf() {
    # Check conda environment first (highest priority for ABI compatibility)
    if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/nf-config" ]; then
        NETCDF_FORTRAN="${CONDA_PREFIX}"
        echo "Found conda NetCDF-Fortran at: ${NETCDF_FORTRAN}"
    # Try nf-config (NetCDF Fortran config tool)
    elif command -v nf-config >/dev/null 2>&1; then
        local nf_prefix
        nf_prefix="$(nf-config --prefix 2>/dev/null)"
        if [ -n "$nf_prefix" ] && [ -d "$nf_prefix" ]; then
            NETCDF_FORTRAN="$nf_prefix"
            echo "Found nf-config, NetCDF-Fortran at: ${NETCDF_FORTRAN}"
        fi
    fi

    # Fallback checks if not yet found
    if [ -z "${NETCDF_FORTRAN}" ] || [ ! -d "${NETCDF_FORTRAN}/include" ]; then
        if [ -n "${NETCDF_FORTRAN}" ] && [ -d "${NETCDF_FORTRAN}/include" ]; then
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
    fi

    # Find NetCDF C library (may be separate from Fortran on macOS)
    if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/nc-config" ]; then
        NETCDF_C="${CONDA_PREFIX}"
    elif command -v nc-config >/dev/null 2>&1; then
        local nc_prefix
        nc_prefix="$(nc-config --prefix 2>/dev/null)"
        if [ -n "$nc_prefix" ] && [ -d "$nc_prefix" ]; then
            NETCDF_C="$nc_prefix"
        fi
    elif [ -d "/opt/homebrew/opt/netcdf" ]; then
        NETCDF_C="/opt/homebrew/opt/netcdf"
    else
        NETCDF_C="${NETCDF_FORTRAN}"
    fi

    export NETCDF_FORTRAN NETCDF_C
    echo "NetCDF detection complete: NETCDF_FORTRAN=${NETCDF_FORTRAN}, NETCDF_C=${NETCDF_C}"
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


def get_bison_detection_and_build() -> str:
    """
    Get reusable bison detection and build-from-source snippet.

    Checks if bison (parser generator) is available, and if not, builds it
    from source in a local directory.

    Returns:
        Shell script snippet for bison detection and building.
    """
    return r'''
# === Bison Detection and Build ===
detect_or_build_bison() {
    BISON_FOUND=false

    # Check if bison is already available
    if command -v bison >/dev/null 2>&1; then
        echo "Found bison: $(command -v bison)"
        bison --version | head -1
        BISON_FOUND=true
        return 0
    fi

    # If not found, build from source
    echo "Bison not found system-wide, building from source..."

    # Save original directory before building
    BISON_ORIGINAL_DIR="$(pwd)"

    BISON_VERSION="3.8.2"
    BISON_BUILD_DIR="${BISON_ORIGINAL_DIR}/bison_build"
    BISON_INSTALL_DIR="${BISON_ORIGINAL_DIR}/bison"

    # Check if already built locally
    if [ -x "${BISON_INSTALL_DIR}/bin/bison" ]; then
        echo "Using previously built bison at: ${BISON_INSTALL_DIR}/bin/bison"
        export PATH="${BISON_INSTALL_DIR}/bin:$PATH"
        bison --version | head -1
        return 0
    fi

    # Download and build bison
    mkdir -p "${BISON_BUILD_DIR}"
    cd "${BISON_BUILD_DIR}"

    if [ ! -f "bison-${BISON_VERSION}.tar.xz" ]; then
        echo "Downloading bison ${BISON_VERSION}..."
        wget -q "https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.xz" \
          || curl -fsSL -o "bison-${BISON_VERSION}.tar.xz" "https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.xz"
    fi

    if [ ! -d "bison-${BISON_VERSION}" ]; then
        echo "Extracting bison..."
        tar -xJf "bison-${BISON_VERSION}.tar.xz"
    fi

    cd "bison-${BISON_VERSION}"
    echo "Configuring bison..."
    ./configure --prefix="${BISON_INSTALL_DIR}"

    echo "Building bison..."
    make -j ${NCORES:-4}

    echo "Installing bison to ${BISON_INSTALL_DIR}..."
    make install

    # Return to original directory
    cd "${BISON_ORIGINAL_DIR}"

    # Add to PATH
    export PATH="${BISON_INSTALL_DIR}/bin:$PATH"

    echo "Bison built successfully"
    bison --version | head -1
}
detect_or_build_bison
    '''.strip()


def get_flex_detection_and_build() -> str:
    """
    Get reusable flex detection and build-from-source snippet.

    Checks if flex (lexical analyzer generator) is available, and if not,
    builds it from source in a local directory.

    Returns:
        Shell script snippet for flex detection and building.
    """
    return r'''
# === Flex Detection and Build ===
detect_or_build_flex() {
    FLEX_FOUND=false
    LIBFL_FOUND=false

    # Check if flex binary is available
    if command -v flex >/dev/null 2>&1; then
        echo "Found flex: $(command -v flex)"
        flex --version | head -1
        FLEX_FOUND=true

        # Check if libfl is available for linking
        # Try multiple methods to find libfl
        if ldconfig -p 2>/dev/null | grep -q libfl; then
            echo "System libfl found via ldconfig"
            LIBFL_FOUND=true
        elif [ -f /usr/lib/libfl.a ] || [ -f /usr/lib/x86_64-linux-gnu/libfl.a ] || [ -f /usr/lib64/libfl.a ]; then
            echo "System libfl.a found"
            LIBFL_FOUND=true
        elif [ -f /usr/lib/libfl.so ] || [ -f /usr/lib/x86_64-linux-gnu/libfl.so ] || [ -f /usr/lib64/libfl.so ]; then
            echo "System libfl.so found"
            LIBFL_FOUND=true
        fi

        if [ "$LIBFL_FOUND" = "true" ]; then
            return 0
        else
            echo "Warning: flex found but libfl not found - will build flex from source for the library"
        fi
    fi

    # If flex or libfl not found, build from source
    if [ "$FLEX_FOUND" = "true" ]; then
        echo "Building flex from source to get libfl library..."
    else
        echo "Flex not found system-wide, building from source..."
    fi

    # Save original directory before building
    FLEX_ORIGINAL_DIR="$(pwd)"

    FLEX_VERSION="2.6.4"
    FLEX_BUILD_DIR="${FLEX_ORIGINAL_DIR}/flex_build"
    FLEX_INSTALL_DIR="${FLEX_ORIGINAL_DIR}/flex"

    # Check if already built locally
    if [ -x "${FLEX_INSTALL_DIR}/bin/flex" ]; then
        echo "Using previously built flex at: ${FLEX_INSTALL_DIR}/bin/flex"
        export PATH="${FLEX_INSTALL_DIR}/bin:$PATH"
        # Export library path for linking (LIBRARY_PATH is used by gcc at link time)
        export FLEX_LIB_DIR="${FLEX_INSTALL_DIR}/lib"
        export LDFLAGS="${LDFLAGS} -L${FLEX_LIB_DIR}"
        export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${FLEX_LIB_DIR}:${LD_LIBRARY_PATH}"
        echo "FLEX_LIB_DIR set to: ${FLEX_LIB_DIR}"
        flex --version | head -1
        return 0
    fi

    # Download and build flex
    mkdir -p "${FLEX_BUILD_DIR}"
    cd "${FLEX_BUILD_DIR}"

    if [ ! -f "flex-${FLEX_VERSION}.tar.gz" ]; then
        echo "Downloading flex ${FLEX_VERSION}..."
        wget -q "https://github.com/westes/flex/releases/download/v${FLEX_VERSION}/flex-${FLEX_VERSION}.tar.gz" \
          || curl -fsSL -o "flex-${FLEX_VERSION}.tar.gz" "https://github.com/westes/flex/releases/download/v${FLEX_VERSION}/flex-${FLEX_VERSION}.tar.gz"
    fi

    if [ ! -d "flex-${FLEX_VERSION}" ]; then
        echo "Extracting flex..."
        tar -xzf "flex-${FLEX_VERSION}.tar.gz"
    fi

    cd "flex-${FLEX_VERSION}"
    echo "Configuring flex..."
    ./configure --prefix="${FLEX_INSTALL_DIR}"

    echo "Building flex..."
    make -j ${NCORES:-4}

    echo "Installing flex to ${FLEX_INSTALL_DIR}..."
    make install

    # Return to original directory
    cd "${FLEX_ORIGINAL_DIR}"

    # Add to PATH
    export PATH="${FLEX_INSTALL_DIR}/bin:$PATH"
    # Export library path for linking (LIBRARY_PATH is used by gcc at link time)
    export FLEX_LIB_DIR="${FLEX_INSTALL_DIR}/lib"
    export LDFLAGS="${LDFLAGS} -L${FLEX_LIB_DIR}"
    export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${FLEX_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "FLEX_LIB_DIR set to: ${FLEX_LIB_DIR}"

    echo "Flex built successfully"
    flex --version | head -1
}
detect_or_build_flex
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
        'bison_detect_build': get_bison_detection_and_build(),
        'flex_detect_build': get_flex_detection_and_build(),
    }
