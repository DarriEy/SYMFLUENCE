#!/usr/bin/env python3
"""
SYMFLUENCE External Tools Configuration - Complete Enhanced Version

This module defines external tool configurations required by SYMFLUENCE,
including repositories, build instructions, and validation criteria.

Tools include:
- SUNDIALS: Differential equation solver library
- SUMMA: Hydrological model with SUNDIALS integration
- mizuRoute: River network routing model
- T-route: NOAA's OWP river network routing model
- FUSE: Framework for Understanding Structural Errors
- TauDEM: Terrain Analysis Using Digital Elevation Models
- GIStool: Geospatial data extraction tool
- Datatool: Meteorological data processing tool
- NGEN: NextGen National Water Model Framework
- NGIAB: NextGen-In-A-Box container environment
"""

from typing import Dict, Any


def get_common_build_environment() -> str:
    """
    Get common build environment setup with comprehensive platform detection and CI support.
    """
    return r'''
set -e

# ================================================================
# PLATFORM AND CI DETECTION
# ================================================================
detect_platform() {
    PLATFORM="unknown"
    PLATFORM_VERSION="unknown"
    IS_CI=false
    IS_HPC=false
    
    # Detect OS
    if [ "$(uname -s)" = "Darwin" ]; then
        PLATFORM="macos"
        PLATFORM_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
    elif [ "$(uname -s)" = "Linux" ]; then
        PLATFORM="linux"
        if [ -f /etc/os-release ]; then
            PLATFORM_VERSION=$(grep "^PRETTY_NAME=" /etc/os-release | cut -d= -f2 | tr -d '"')
        else
            PLATFORM_VERSION="unknown"
        fi
    else
        PLATFORM="$(uname -s)"
        PLATFORM_VERSION="unknown"
    fi
    
    # Detect CI
    if [ -n "$GITHUB_ACTIONS" ] || [ -n "$CI" ]; then
        IS_CI=true
    fi
    
    # Detect HPC (very heuristic)
    if command -v module >/dev/null 2>&1 || \
       [ -n "$SLURM_JOB_ID" ] || [ -n "$PBS_JOBID" ] || [ -d "/cvmfs" ]; then
        IS_HPC=true
    fi
    
    echo "           🤖 CI environment detected" && IS_CI=true || true
    
    echo "           📍 Platform: $PLATFORM $PLATFORM_VERSION ($(uname -m))"
    
    if [ "$IS_CI" = true ]; then
        echo "           🧪 Running in CI environment"
    fi
    if [ "$IS_HPC" = true ]; then
        echo "           🖥️  HPC-like environment detected"
    fi
}

# ================================================================
# NETCDF / HDF5 DETECTION
# ================================================================
detect_netcdf() {
    # Reset variables
    NETCDF=""
    NETCDF_FORTRAN=""
    HDF5=""

    # Strategy 1: Use explicit environment variables if provided
    if [ -n "$NETCDF" ] && [ -d "$NETCDF" ]; then
        echo "           ✓ Using NETCDF from environment: $NETCDF"
    fi

    if [ -n "$NETCDF_FORTRAN" ] && [ -d "$NETCDF_FORTRAN" ]; then
        echo "           ✓ Using NETCDF_FORTRAN from environment: $NETCDF_FORTRAN"
    fi

    if [ -n "$HDF5" ] && [ -d "$HDF5" ]; then
        echo "           ✓ Using HDF5 from environment: $HDF5"
    fi

    # Strategy 2: Use nf-config/nc-config if available
    if [ -z "$NETCDF_FORTRAN" ] && command -v nf-config >/dev/null 2>&1; then
        NETCDF_FORTRAN=$(nf-config --prefix 2>/dev/null || echo "")
        if [ -n "$NETCDF_FORTRAN" ] && [ -d "$NETCDF_FORTRAN" ]; then
            echo "           ✓ NetCDF-Fortran found via nf-config: $NETCDF_FORTRAN"
        fi
    fi

    if [ -z "$NETCDF" ] && command -v nc-config >/dev/null 2>&1; then
        NETCDF=$(nc-config --prefix 2>/dev/null || echo "")
        if [ -n "$NETCDF" ] && [ -d "$NETCDF" ]; then
            echo "           ✓ NetCDF C found via nc-config: $NETCDF"
        fi
    fi

    # Strategy 3: Check common system directories on macOS and Linux
    if [ -z "$NETCDF" ] || [ -z "$NETCDF_FORTRAN" ] || [ -z "$HDF5" ]; then
        COMMON_PATHS="
/usr
/usr/local
/opt
/opt/homebrew
/opt/homebrew/opt/netcdf
/opt/homebrew/opt/netcdf-fortran
/opt/homebrew/opt/hdf5
"
        for base in $COMMON_PATHS; do
            if [ -z "$NETCDF" ] && [ -d "$base" ] && ls "$base" 2>/dev/null | grep -qi "netcdf"; then
                NETCDF="$base"
                echo "           ✓ NetCDF found: $NETCDF"
            fi
            if [ -z "$NETCDF_FORTRAN" ] && [ -d "$base" ] && ls "$base" 2>/dev/null | grep -qi "netcdf-fortran"; then
                NETCDF_FORTRAN="$base"
                echo "           ✓ NetCDF-Fortran found: $NETCDF_FORTRAN"
            fi
            if [ -z "$HDF5" ] && [ -d "$base" ] && ls "$base" 2>/dev/null | grep -qi "hdf5"; then
                HDF5="$base"
                echo "           ✓ HDF5 found: $HDF5"
            fi
        done
    fi

    # Final fallbacks
    if [ -z "$NETCDF" ]; then
        NETCDF="/usr"
        echo "           ℹ️ Defaulting NETCDF to /usr"
    fi
    if [ -z "$NETCDF_FORTRAN" ]; then
        NETCDF_FORTRAN="$NETCDF"
        echo "           ℹ️ Defaulting NETCDF_FORTRAN to NETCDF: $NETCDF_FORTRAN"
    fi
    if [ -z "$HDF5" ]; then
        HDF5="$NETCDF"
        echo "           ℹ️ Defaulting HDF5 to NETCDF: $HDF5"
    fi

    export NETCDF NETCDF_FORTRAN HDF5
}

# ================================================================
# COMPILER DETECTION
# ================================================================
detect_compilers() {
    if [ -n "$SYMFLUENCE_COMPILERS_DETECTED" ]; then
        return 0  # Already detected
    fi
    
    echo "🔍 Detecting compilers with enhanced platform support..."
    
    # First detect platform
    detect_platform
    
    # Strategy 0: If compilers are already set and valid, use them
    if [ -n "$CC" ] && [ -n "$FC" ] && command -v "$CC" >/dev/null 2>&1 && command -v "$FC" >/dev/null 2>&1; then
        echo "  ✓ Using pre-configured compilers: CC=$CC, FC=$FC"
        export CXX="${CXX:-g++}"
        export FC_EXE="$FC"
    
    # Strategy 1: GitHub Actions CI (Ubuntu with preinstalled GCC and MPI)
    elif [ "$IS_CI" = true ] && [ "$PLATFORM" = "linux" ]; then
        echo "  📦 Applying CI-specific settings..."
        if command -v mpicc >/dev/null 2>&1 && command -v mpicxx >/dev/null 2>&1; then
            export CC="mpicc"
            export CXX="mpicxx"
        else
            export CC="gcc"
            export CXX="g++"
        fi
        export FC="gfortran"
        export FC_EXE="$FC"
    
    # Strategy 2: macOS with Homebrew (ARM or Intel)
    elif [ "$PLATFORM" = "macos" ]; then
        # Prefer Homebrew GCC for Fortran (gfortran)
        if command -v gfortran >/dev/null 2>&1; then
            export FC="$(command -v gfortran)"
            export FC_EXE="$FC"
        fi
        
        # Use Apple Clang or Homebrew gcc for C/C++
        if command -v gcc-14 >/dev/null 2>&1; then
            export CC="gcc-14"
        elif command -v gcc-13 >/dev/null 2>&1; then
            export CC="gcc-13"
        elif command -v gcc >/dev/null 2>&1; then
            export CC="gcc"
        else
            export CC="cc"
        fi
        
        if command -v g++-14 >/dev/null 2>&1; then
            export CXX="g++-14"
        elif command -v g++-13 >/dev/null 2>&1; then
            export CXX="g++-13"
        elif command -v g++ >/dev/null 2>&1; then
            export CXX="g++"
        else
            export CXX="c++"
        fi
    
    # Strategy 3: HPC with EasyBuild or module-provided GCC
    elif [ "$IS_HPC" = true ]; then
        if [ -n "$EBVERSIONGCC" ] || [ -n "$EBROOTGCC" ]; then
            echo "  ✓ Found EasyBuild GCC module: ${EBVERSIONGCC:-unknown}"
            export CC="${CC:-gcc}"
            export CXX="${CXX:-g++}"
            export FC="${FC:-gfortran}"
            export FC_EXE="$FC"
        else
            if command -v module >/dev/null 2>&1; then
                if module list 2>&1 | grep -qi "gcc"; then
                    echo "  ✓ GCC module appears to be loaded"
                    export CC="${CC:-gcc}"
                    export CXX="${CXX:-g++}"
                    export FC="${FC:-gfortran}"
                    export FC_EXE="$FC"
                fi
            fi
        fi
    fi
    
    # Final fallback for any platform
    if [ -z "$CC" ]; then
        if command -v gcc >/dev/null 2>&1; then
            export CC="gcc"
        else
            export CC="cc"
        fi
    fi
    if [ -z "$CXX" ]; then
        if command -v g++ >/dev/null 2>&1; then
            export CXX="g++"
        else
            export CXX="c++"
        fi
    fi
    if [ -z "$FC" ]; then
        if command -v gfortran >/dev/null 2>&1; then
            export FC="gfortran"
        else
            echo "  ⚠️ No Fortran compiler found in PATH; some tools may not build"
        fi
    fi
    if [ -z "$FC_EXE" ] && [ -n "$FC" ]; then
        export FC_EXE="$FC"
    fi
    
    SYMFLUENCE_COMPILERS_DETECTED=1
    export SYMFLUENCE_COMPILERS_DETECTED
    
    # Print final compiler configuration
    echo "  ✅ Compilers configured: CC=$CC | CXX=$CXX | FC=$FC"
    echo "  📍 Compiler versions:"
    { "$CC" --version 2>/dev/null || echo "   (CC version not available)"; } | head -n 1
    { "$CXX" --version 2>/dev/null || echo "   (CXX version not available)"; } | head -n 1
    { "$FC" --version 2>/dev/null || echo "   (FC version not available)"; } | head -n 1
}

# ================================================================
# PYTHON DETECTION
# ================================================================
detect_python() {
    # Respect explicitly provided Python from environment if available
    if [ -n "$SYMFLUENCE_PYTHON" ] && [ -x "$SYMFLUENCE_PYTHON" ]; then
        echo "  🐍 Using SYMFLUENCE_PYTHON from environment: $SYMFLUENCE_PYTHON"
        return 0
    fi
    
    # Try venv/ directory relative to current working directory
    if [ -x "venv/bin/python" ]; then
        export SYMFLUENCE_PYTHON="$(pwd)/venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        export SYMFLUENCE_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        export SYMFLUENCE_PYTHON="$(command -v python)"
    else
        echo "  ❌ No suitable Python interpreter found"
        exit 1
    fi
    
    echo "  🐍 Using Python: $SYMFLUENCE_PYTHON"
}

# ================================================================
# NCORES DETECTION
# ================================================================
detect_ncores() {
    if [ -n "$NCORES" ]; then
        return 0
    fi
    
    if command -v nproc >/dev/null 2>&1; then
        NCORES=$(nproc)
    elif [ "$PLATFORM" = "macos" ]; then
        NCORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
    else
        NCORES=2
    fi
    
    if [ "$IS_CI" = true ]; then
        # Be conservative in CI to avoid over-subscribing
        NCORES=2
    fi
    
    export NCORES
    echo "  🔧 Using $NCORES cores for parallel builds"
}

# ================================================================
# GLOBAL SETUP ENTRYPOINT
# ================================================================
detect_platform
detect_compilers
detect_netcdf
detect_python
detect_ncores

'''


def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Define all external tools required by SYMFLUENCE with enhanced build scripts.
    
    Returns:
        Dictionary mapping tool names to their complete configuration
    """
    common_env = get_common_build_environment()
    
    return {
        # ================================================================
        # SUNDIALS - Solver Library (Enhanced Build)
        # ================================================================
        'sundials': {
            'description': 'SUNDIALS - SUite of Nonlinear and DIfferential/ALgebraic equation Solvers',
            'config_path_key': 'SUNDIALS_INSTALL_PATH',
            'config_exe_key': 'SUNDIALS_DIR',
            # Root directory for this tool under SYMFLUENCE_DATA_DIR
            'default_path_suffix': 'installs/sundials/',
            'default_exe': 'install/sundials/lib/libsundials_core.a',
            'repository': None,
            'branch': None,
            'install_dir': 'sundials',
            'build_commands': [
                common_env + r'''
# Enhanced SUNDIALS build with better error handling and CI support
SUNDIALS_VER=7.1.1  # Using stable version for better compatibility
SUNDIALSDIR="$(pwd)/install/sundials"

echo "📦 Building SUNDIALS v${SUNDIALS_VER}..."

# Clean up any previous attempts
rm -rf sundials-${SUNDIALS_VER} build v${SUNDIALS_VER}.tar.gz || true

# Download with retry logic
for attempt in 1 2 3; do
    echo "  📥 Download attempt $attempt..."
    if wget -q --timeout=30 https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz || \
       curl -fsSL --connect-timeout 30 -o v${SUNDIALS_VER}.tar.gz https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz; then
        echo "  ✓ Download successful"
        break
    elif [ $attempt -eq 3 ]; then
        echo "  ❌ Failed to download SUNDIALS after 3 attempts"
        exit 1
    else
        echo "  ⚠️ Download failed, retrying in 2 seconds..."
        sleep 2
    fi
done

# Extract
tar -xzf v${SUNDIALS_VER}.tar.gz
cd sundials-${SUNDIALS_VER}

# Create build directory
rm -rf build && mkdir build && cd build

# Detect MPI availability
USE_MPI="OFF"
if command -v mpicc >/dev/null 2>&1; then
    echo "  ✓ MPI detected, enabling MPI support"
    USE_MPI="ON"
    CC_COMPILER="$(command -v mpicc)"
    CXX_COMPILER="$(command -v mpicxx || command -v mpic++ || command -v mpicc)"
else
    echo "  ℹ️ MPI not detected, building without MPI support"
    CC_COMPILER="$(command -v "$CC" || command -v gcc || command -v cc)"
    CXX_COMPILER="$(command -v "$CXX" || command -v g++ || command -v c++)"
fi

FC_COMPILER="$(command -v "$FC" || command -v gfortran || true)"

echo "📋 Configuration:"
echo "  CC: $CC_COMPILER"
echo "  CXX: $CXX_COMPILER"
echo "  FC: $FC_COMPILER"
echo "  Install: $SUNDIALSDIR"
echo "  MPI: $USE_MPI"

if [ -z "$CC_COMPILER" ] || [ -z "$CXX_COMPILER" ]; then
    echo "  ❌ No C/C++ compiler found (CC_COMPILER/CXX_COMPILER empty)"
    exit 1
fi

if [ -z "$FC_COMPILER" ]; then
    echo "  ❌ No Fortran compiler found (FC_COMPILER empty)"
    echo "  💡 Make sure gfortran (or your Fortran compiler) is in PATH and visible"
    exit 1
fi

# Configure with appropriate options
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$SUNDIALSDIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DCMAKE_Fortran_COMPILER="$FC_COMPILER" \
  -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_STATIC_LIBS=ON \
  -DEXAMPLES_ENABLE=OFF \
  -DBUILD_TESTING=OFF \
  -DENABLE_MPI=$USE_MPI \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  2>&1 | tee cmake_config.log

# Check if configuration succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ CMake configuration failed"
    echo "  📋 Last 30 lines of configuration log:"
    tail -30 cmake_config.log
    exit 1
fi

# Build
echo "  🔨 Building SUNDIALS..."
make -j${NCORES} install 2>&1 | tee build.log

# Check if build succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ Build failed"
    echo "  📋 Last 30 lines of build log:"
    tail -30 build.log
    exit 1
fi

# Verify installation
echo "  🔍 Verifying SUNDIALS installation..."
if [ -d "$SUNDIALSDIR/lib64" ]; then
    LIBDIR="$SUNDIALSDIR/lib64"
elif [ -d "$SUNDIALSDIR/lib" ]; then
    LIBDIR="$SUNDIALSDIR/lib"
else
    echo "  ❌ Library directory not found"
    exit 1
fi

# Check for core libraries
REQUIRED_LIBS="sundials_core sundials_nvecserial"
MISSING_LIBS=""
for lib in $REQUIRED_LIBS; do
    if [ ! -f "$LIBDIR/lib${lib}.a" ] && [ ! -f "$LIBDIR/lib${lib}.so" ] && [ ! -f "$LIBDIR/lib${lib}.dylib" ]; then
        MISSING_LIBS="$MISSING_LIBS $lib"
    fi
done

if [ -n "$MISSING_LIBS" ]; then
    echo "  ❌ Missing required libraries:$MISSING_LIBS"
    echo "  📋 Available libraries:"
    ls -la "$LIBDIR" | grep -E "\.a|\.so|\.dylib"
    exit 1
fi

echo "  ✅ SUNDIALS installation verified"
echo "  📁 Libraries in: $LIBDIR"
ls -la "$LIBDIR" | head -10
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                # Paths are relative to SUNDIALS_INSTALL_PATH, which is the tool root (installs/sundials/)
                'file_paths': [
                    'install/sundials/lib/libsundials_core.a',
                    'install/sundials/lib64/libsundials_core.a',
                    'install/sundials/lib/libsundials_core.so',
                    'install/sundials/lib64/libsundials_core.so',
                    'install/sundials/lib/libsundials_core.dylib',
                    'install/sundials/lib64/libsundials_core.dylib',
                    'install/sundials/include/sundials/sundials_config.h'
                ],
                'check_type': 'exists_any'
            },
            'order': 1
        },

        # ================================================================
        # SUMMA - Hydrological model with SUNDIALS integration
        # ================================================================
        'summa': {
            'description': 'Structure for Unifying Multiple Modeling Alternatives',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'config_exe_key': 'SUMMA_EXE',
            'default_path_suffix': 'installs/summa/bin/',
            'default_exe': 'summa_sundials.exe',
            'repository': 'https://github.com/CH-Earth/summa.git',
            'branch': 'develop',
            'install_dir': 'summa',
            'requires': ['sundials'],
            'build_commands': [
                common_env + r'''
# Enhanced SUMMA build with better SUNDIALS detection
echo "🔨 Building SUMMA..."

# Find SUNDIALS installation with multiple search strategies
SUNDIALS_BASE=""

# Strategy 1: Expected relative location (tool layout)
if [ -d "$(dirname $(pwd))/sundials/install/sundials" ]; then
    SUNDIALS_BASE="$(cd $(dirname $(pwd))/sundials/install/sundials && pwd)"
    echo "  ✓ Found SUNDIALS at expected location"
fi

# Strategy 2: Search common relative paths
if [ -z "$SUNDIALS_BASE" ]; then
    echo "  🔍 Searching for SUNDIALS installation..."
    for search_dir in \
        ../sundials/install/sundials \
        ../../sundials/install/sundials \
        ../../../sundials/install/sundials \
        $HOME/SYMFLUENCE_data/installs/sundials/install/sundials \
        $SYMFLUENCE_DATA_DIR/installs/sundials/install/sundials; do
        if [ -d "$search_dir" ]; then
            SUNDIALS_BASE="$(cd $search_dir && pwd)"
            echo "  ✓ Found SUNDIALS at: $SUNDIALS_BASE"
            break
        fi
    done
fi

# Strategy 3: Use environment variable if set
if [ -z "$SUNDIALS_BASE" ] && [ -n "$SUNDIALS_PATH" ]; then
    if [ -d "$SUNDIALS_PATH" ]; then
        SUNDIALS_BASE="$SUNDIALS_PATH"
        echo "  ✓ Using SUNDIALS_PATH environment variable"
    fi
fi

if [ -z "$SUNDIALS_BASE" ] || [ ! -d "$SUNDIALS_BASE" ]; then
    echo "  ❌ Cannot find SUNDIALS installation"
    echo "  💡 Please install SUNDIALS first: python symfluence.py --get_executables sundials"
    exit 1
fi

# Determine SUNDIALS library directory (lib vs lib64)
if [ -d "$SUNDIALS_BASE/lib64" ]; then
    SUNDIALS_LIB="$SUNDIALS_BASE/lib64"
elif [ -d "$SUNDIALS_BASE/lib" ]; then
    SUNDIALS_LIB="$SUNDIALS_BASE/lib"
else
    echo "  ❌ SUNDIALS library directory not found"
    echo "  📁 Contents of $SUNDIALS_BASE:"
    ls -la "$SUNDIALS_BASE" | head -20
    exit 1
fi

echo "  ✓ Using SUNDIALS from: $SUNDIALS_BASE"
echo "  ✓ SUNDIALS libraries: $SUNDIALS_LIB"

# Configure environment
export FC_EXE="${FC}"
export FC="${FC}"
export SUNDIALS_PATH="$SUNDIALS_BASE"

# Move to build directory
cd build || { echo "  ❌ build directory not found"; exit 1; }

# SUMMA Makefile expects F_MASTER to point to the parent of the build directory
export F_MASTER="$(cd .. && pwd)"
echo "  ✓ F_MASTER set to: $F_MASTER"

# Create Makefile configuration
echo "  📝 Creating Makefile configuration..."
cat > Makefile.config <<EOF
# Auto-generated Makefile configuration for SUMMA
# Platform: $(uname -s)
# Compiler: ${FC}

# Compiler settings
FC = ${FC}
FC_EXE = ${FC_EXE}
CC = ${CC}

# Include directories
INCLUDES = -I${NETCDF}/include -I${NETCDF_FORTRAN}/include -I${SUNDIALS_BASE}/include

# Library directories
LIBRARIES = -L${NETCDF}/lib -L${NETCDF_FORTRAN}/lib -L${SUNDIALS_LIB}

# Additional library search paths
LIBRARIES += -L${NETCDF}/lib64 -L${NETCDF_FORTRAN}/lib64 2>/dev/null || true

# Libraries to link
LIBS = -lnetcdff -lnetcdf -lsundials_nvecserial -lsundials_core

# Add system libraries
LIBS += -lblas -llapack -lm

# Compiler flags
FFLAGS = -O3 -fPIC -fbacktrace -ffree-line-length-none
CFLAGS = -O3 -fPIC

# Full linking flags
LDFLAGS = \$(LIBRARIES) \$(LIBS)

# Installation directory
INSTALL_DIR = ../bin
EOF

echo "  📋 Makefile configuration created"
cat Makefile.config

# Clean previous builds
echo "  🧹 Cleaning previous builds..."
make clean 2>/dev/null || true
rm -f *.o *.mod *.exe 2>/dev/null || true

# Build SUMMA
echo "  🔨 Compiling SUMMA..."
make -j${NCORES} 2>&1 | tee build.log

# Check build result
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ❌ Build failed"
    echo "  📋 Checking for common issues..."
    
    # Check for missing libraries
    if grep -q "cannot find -l" build.log; then
        echo "  ⚠️ Missing libraries detected:"
        grep "cannot find -l" build.log | head -10
    fi
    
    # Check for compilation errors
    if grep -q "Error:" build.log; then
        echo "  ⚠️ Compilation errors:"
        grep -A2 -B2 "Error:" build.log | head -30
    fi
    
    # Try alternative build if main fails
    echo "  🔧 Attempting alternative build configuration..."
    make FC=${FC} FC_EXE=${FC_EXE} -j${NCORES} 2>&1 | tee build_retry.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ❌ Alternative build also failed"
        exit 1
    fi
fi

# Install executable
echo "  📦 Installing SUMMA executable..."
mkdir -p ../bin

# Try different possible executable names
SUMMA_EXE=""
for exe_name in summa_sundials.exe summa.exe summa; do
    if [ -f "$exe_name" ]; then
        SUMMA_EXE="$exe_name"
        echo "  ✓ Found executable: $SUMMA_EXE"
        break
    fi
done

if [ -z "$SUMMA_EXE" ]; then
    echo "  ❌ SUMMA executable not found after build"
    echo "  📋 Build directory contents:"
    ls -la | grep -E "\.exe|summa" || ls -la | head -20
    exit 1
fi

# Copy and rename to standard name
cp "$SUMMA_EXE" ../bin/summa_sundials.exe
chmod +x ../bin/summa_sundials.exe
echo "  ✅ SUMMA installed to ../bin/summa_sundials.exe"
ls -la ../bin/

# Verify it can run
echo "  🔍 Verifying SUMMA executable..."
if ../bin/summa_sundials.exe --help 2>/dev/null | grep -q "SUMMA"; then
    echo "  ✅ SUMMA executable verified"
elif ldd ../bin/summa_sundials.exe 2>&1 | grep -q "not found"; then
    echo "  ⚠️ SUMMA has missing library dependencies:"
    ldd ../bin/summa_sundials.exe | grep "not found"
else
    echo "  ✓ SUMMA executable created (runtime verification skipped)"
fi

echo "  ✅ SUMMA build complete"
                '''
            ],
            'dependencies': ['netcdf', 'netcdf-fortran', 'sundials'],
            'test_command': None,
            'verify_install': {
                'file_paths': [
                    'bin/summa_sundials.exe'
                ],
                'check_type': 'exists_any'
            },
            'order': 2
        },

        # ================================================================
        # mizuRoute - River network routing model
        # ================================================================
        'mizuroute': {
            'description': 'River network routing model',
            'config_path_key': 'MIZUROUTE_INSTALL_PATH',
            'config_exe_key': 'MIZUROUTE_EXE',
            'default_path_suffix': 'installs/mizuRoute/route/bin/',
            'default_exe': 'mizuRoute.exe',
            'repository': 'https://github.com/ESCOMP/mizuRoute.git',
            'branch': 'main',
            'install_dir': 'mizuRoute',
            'build_commands': [
                common_env + r'''
# Build mizuRoute
echo "Building mizuRoute..."
cd route/build/

# mizuRoute Makefile expects F_MASTER to be the parent of the build directory
export F_MASTER="$(cd .. && pwd)"
echo "  ✓ F_MASTER set to: $F_MASTER"

# Create/update Makefile configuration
cat > Makefile.config <<EOF
FC = ${FC}
FC_EXE = \$(FC)
FLAGS_DEBUG = -g -O0 -ffree-line-length-none -fbacktrace -fcheck=all
FLAGS_OPT = -O3 -ffree-line-length-none -fbacktrace
INCLUDES = -I${NETCDF}/include -I${NETCDF_FORTRAN}/include
LIBRARIES = -L${NETCDF}/lib -L${NETCDF_FORTRAN}/lib
LIBS = -lnetcdff -lnetcdf
EOF

# Build
make clean 2>/dev/null || true
make FC=${FC} FC_EXE=${FC} -j ${NCORES:-4}

# Verify and install
mkdir -p ../bin
if [ -f "mizuRoute.exe" ]; then
    cp mizuRoute.exe ../bin/
    chmod +x ../bin/mizuRoute.exe
    echo "✅ mizuRoute built successfully"
else
    echo "❌ mizuRoute build failed - executable not found"
    echo "📋 Build directory contents:"
    ls -la
    exit 1
fi
                '''
            ],
            'dependencies': ['netcdf', 'netcdf-fortran'],
            'test_command': None,
            'verify_install': {
                'file_paths': ['route/bin/mizuRoute.exe'],
                'check_type': 'exists'
            },
            'order': 3
        },

        # ================================================================
        # T-route - NOAA Next Generation river routing model
        # ================================================================
        'troute': {
            'description': "NOAA's Next Generation river routing model",
            'config_path_key': 'TROUTE_INSTALL_PATH',
            'config_exe_key': 'TROUTE_MODULE',
            'default_path_suffix': 'installs/t-route/src/troute-network/',
            'default_exe': 'troute/network/__init__.py',
            'repository': 'https://github.com/NOAA-OWP/t-route.git',
            'branch': None,
            'install_dir': 't-route',
            'build_commands': [
                common_env + r'''
# Install Python dependencies for t-route
echo "Setting up t-route..."
${SYMFLUENCE_PYTHON} -m pip install --upgrade pip setuptools wheel
cd src/troute-network/
${SYMFLUENCE_PYTHON} -m pip install -e . || {
  echo "⚠️  Full installation failed, trying minimal setup..."
  ${SYMFLUENCE_PYTHON} -m pip install -e . --no-deps
}
cd ../..
echo "✅ T-route setup complete"
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['src/troute-network/troute/network/__init__.py'],
                'check_type': 'exists'
            },
            'order': 4
        },

        # ================================================================
        # FUSE - Framework for Understanding Structural Errors
        # ================================================================
        'fuse': {
            'description': 'Framework for Understanding Structural Errors in hydrological models',
            'config_path_key': 'FUSE_INSTALL_PATH',
            'config_exe_key': 'FUSE_EXE',
            'default_path_suffix': 'installs/fuse/bin/',
            'default_exe': 'fuse.exe',
            'repository': 'https://github.com/CH-Earth/fuse.git',
            'branch': None,
            'install_dir': 'fuse',
            'build_commands': [
                common_env + r'''
# Build FUSE
echo "Building FUSE..."
cd build/

# Configure build
cat > Makefile.config <<EOF
# Auto-generated FUSE Makefile.config
FC = ${FC}
CC = ${CC}
FFLAGS = -O3 -ffree-line-length-none -fbacktrace
CFLAGS = -O3
LDFLAGS = -L${NETCDF}/lib -L${NETCDF_FORTRAN}/lib
INCLUDES = -I${NETCDF}/include -I${NETCDF_FORTRAN}/include
LIBS = -lnetcdff -lnetcdf -lm
EOF

# Build
make clean 2>/dev/null || true
make -j${NCORES} 2>&1 | tee build.log

# Install
mkdir -p ../bin

# Try to find a FUSE executable
FUSE_EXE=""
for exe in fuse.exe fuse FUSE.exe FUSE; do
    if [ -f "$exe" ]; then
        FUSE_EXE="$exe"
        break
    fi
done

if [ -z "$FUSE_EXE" ]; then
    # Fallback: any executable starting with "fuse"
    FUSE_EXE=$(find . -maxdepth 1 -type f -perm -111 -name "fuse*" 2>/dev/null | head -n 1 || echo "")
fi

if [ -z "$FUSE_EXE" ]; then
    echo "❌ Could not find FUSE executable after build"
    echo "📋 Directory contents:"
    ls -la
    exit 1
fi

cp "$FUSE_EXE" ../bin/fuse.exe
chmod +x ../bin/fuse.exe

echo "🧪 Testing binary..."
../bin/fuse.exe --help >/dev/null 2>&1 || true
echo "✅ FUSE build successful"
                '''
            ],
            'dependencies': ['netcdf', 'netcdf-fortran'],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/fuse.exe'],
                'check_type': 'exists'
            },
            'order': 5
        },

        # ================================================================
        # TauDEM - Terrain Analysis Using DEMs
        # ================================================================
        'taudem': {
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'config_exe_key': 'TAUDEM_BIN',
            'default_path_suffix': 'installs/TauDEM/bin/',
            'default_exe': 'pitremove',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': None,
            'install_dir': 'TauDEM',
            'build_commands': [
                common_env + r'''
# Build TauDEM
echo "Building TauDEM..."

# First install any Python requirements
if [ -f requirements.txt ]; then
    ${SYMFLUENCE_PYTHON} -m pip install -r requirements.txt
fi

# Build core tools
if [ -f "build.sh" ]; then
    bash build.sh
elif [ -f "CMakeLists.txt" ]; then
    mkdir -p build && cd build
    cmake .. -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
    make -j${NCORES}
    cd ..
else
    echo "  ⚠️ No recognized build system for TauDEM; assuming prebuilt binaries or manual install"
fi

# Stage executables
mkdir -p bin
# Common TauDEM executables (include inundepth too)
for exe in pitremove d8flowdir d8contributingarea aread8 dinfupgrid dinfdistdownstream streamnet threshold inundepth; do
    for loc in "$exe" "src/$exe" "build/$exe" "build/src/$exe"; do
        if [ -f "$loc" ]; then
            cp "$loc" bin/
            chmod +x "bin/$exe"
            break
        fi
    done
done

echo "✅ TauDEM executables staged"
                '''
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
        # GIStool - Geospatial data extraction tool
        # ================================================================
        'gistool': {
            'description': 'Geospatial data extraction and processing tool',
            'config_path_key': 'GISTOOL_INSTALL_PATH',
            'config_exe_key': 'GISTOOL_SCRIPT',
            'default_path_suffix': 'installs/gistool/',
            'default_exe': 'extract-gis.sh',
            'repository': 'https://github.com/kasra-keshavarz/gistool.git',
            'branch': None,
            'install_dir': 'gistool',
            'build_commands': [
                r'''
# GIStool requires no compilation; just ensure script is executable
echo "Configuring GIStool..."
chmod +x extract-gis.sh || true
echo "✅ GIStool configured"
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['extract-gis.sh'],
                'check_type': 'exists'
            },
            'order': 7
        },

        # ================================================================
        # Datatool - Meteorological data processing tool
        # ================================================================
        'datatool': {
            'description': 'Meteorological data extraction and processing tool',
            'config_path_key': 'DATATOOL_INSTALL_PATH',
            'config_exe_key': 'DATATOOL_SCRIPT',
            'default_path_suffix': 'installs/datatool/',
            'default_exe': 'extract-dataset.sh',
            'repository': 'https://github.com/kasra-keshavarz/datatool.git',
            'branch': None,
            'install_dir': 'datatool',
            'build_commands': [
                r'''
# Datatool requires no compilation; just ensure script is executable
echo "Configuring Datatool..."
chmod +x extract-dataset.sh || true
echo "✅ Datatool configured"
                '''
            ],
            'dependencies': [],
            'test_command': None,
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
            'default_path_suffix': 'installs/ngen/cmake_build/',
            'default_exe': 'ngen',
            'repository': 'https://github.com/CIROH-UA/ngen',
            'branch': 'ngiab',
            'install_dir': 'ngen',
            'build_commands': [
                common_env + r'''
set -e
echo "Building ngen..."

# Make sure CMake sees a supported NumPy
export PYTHONNOUSERSITE=1
${SYMFLUENCE_PYTHON} -m pip install --upgrade "pip<24.1" >/dev/null 2>&1 || true
${SYMFLUENCE_PYTHON} -m pip install "numpy<2" "setuptools<70" 2>/dev/null || true

# Get Boost (local installation)
if [ ! -d "boost_1_79_0" ]; then
    echo "Fetching Boost 1.79.0..."
    wget -q https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 -O boost_1_79_0.tar.bz2 || \
    curl -fsSL -o boost_1_79_0.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2
    tar -xjf boost_1_79_0.tar.bz2 && rm -f boost_1_79_0.tar.bz2
fi
export BOOST_ROOT="$(pwd)/boost_1_79_0"

# Update submodules
git submodule update --init --recursive -- test/googletest extern/pybind11 2>/dev/null || true

# Clean previous builds
rm -rf cmake_build

# Configure with CMake - try with Python first, fall back without
if cmake \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="$BOOST_ROOT" \
    -DNGEN_WITH_PYTHON=ON \
    -DNGEN_WITH_SQLITE3=ON \
    -S . -B cmake_build 2>&1 | tee cmake.log; then
    echo "Configured with Python support"
else
    echo "Retrying without Python support..."
    rm -rf cmake_build
    cmake \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBOOST_ROOT="$BOOST_ROOT" \
        -DNGEN_WITH_PYTHON=OFF \
        -DNGEN_WITH_SQLITE3=ON \
        -S . -B cmake_build
fi

# Build ngen
cmake --build cmake_build --target ngen -j ${NCORES:-4}

# Verify
if [ -f "cmake_build/ngen" ]; then
    ./cmake_build/ngen --help >/dev/null 2>&1 || true
    echo "✅ ngen built successfully"
else
    echo "❌ ngen build failed"
    exit 1
fi
                '''
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['cmake_build/ngen'],
                'check_type': 'exists'
            },
            'order': 9
        },

        # ================================================================
        # NGIAB - NextGen-In-A-Box (container-based ngen deployment)
        # ================================================================
        'ngiab': {
            'description': 'NextGen In A Box - Container-based ngen deployment',
            'config_path_key': 'NGIAB_INSTALL_PATH',
            'config_exe_key': 'NGIAB_GUIDE',
            'default_path_suffix': 'installs/ngiab/',
            'default_exe': 'guide.sh',
            'repository': None,
            'branch': None,
            'install_dir': 'ngiab',
            'build_commands': [
                r'''
# NGIAB is environment-specific; here we just provide a placeholder
echo "Configuring NGIAB..."

# Simple heuristic: if we detect an HPC environment, point to NGIAB-HPC docs,
# otherwise use NGIAB-CloudInfra
if command -v module >/dev/null 2>&1 || [ -n "$SLURM_JOB_ID" ] || [ -d "/cvmfs" ]; then
    echo "HPC environment detected; using NGIAB-HPC"
    echo "#!/usr/bin/env bash" > guide.sh
    echo 'echo "See NGIAB-HPC documentation for deployment instructions."' >> guide.sh
else
    echo "Non-HPC environment detected; using NGIAB-CloudInfra"
    echo "#!/usr/bin/env bash" > guide.sh
    echo 'echo "See NGIAB-CloudInfra documentation for deployment instructions."' >> guide.sh
fi
chmod +x guide.sh
echo "✅ NGIAB configured"
                '''
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
