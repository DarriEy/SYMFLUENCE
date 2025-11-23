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
- NGIAB: NextGen In A Box deployment system
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
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        PLATFORM="${ID:-unknown}"
        PLATFORM_VERSION="${VERSION_ID:-unknown}"
    elif [ "$(uname)" = "Darwin" ]; then
        PLATFORM="macos"
        PLATFORM_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
    fi
    
    # Detect CI environment
    if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ] || [ -n "$GITLAB_CI" ] || [ -n "$JENKINS_HOME" ]; then
        IS_CI=true
        echo "  🤖 CI environment detected"
    fi
    
    # Detect HPC environment
    if command -v sbatch >/dev/null 2>&1 || command -v qsub >/dev/null 2>&1 || [ -n "$SLURM_CLUSTER_NAME" ]; then
        IS_HPC=true
        echo "  🖥️ HPC environment detected"
    fi
    
    echo "  📍 Platform: $PLATFORM $PLATFORM_VERSION ($(uname -m))"
}

# ================================================================
# ENHANCED COMPILER DETECTION - PLATFORM-AGNOSTIC
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
    
    # Strategy 1: Use module environment if available (HPC systems)
    elif command -v module >/dev/null 2>&1; then
        echo "  🔍 Checking for loaded compiler modules..."
        # Try to load gcc module if not loaded
        if ! module list 2>&1 | grep -qE "gcc|intel|pgi"; then
            for gcc_module in gcc/11 gcc/10 gcc; do
                if module avail 2>&1 | grep -q "$gcc_module"; then
                    echo "  Loading module: $gcc_module"
                    module load $gcc_module 2>/dev/null || true
                    break
                fi
            done
        fi
        # Now set compilers
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        export FC="${FC:-gfortran}"
        export FC_EXE="$FC"
    
    # Strategy 2: Use EasyBuild if available
    elif [ -n "$EBVERSIONGCC" ]; then
        echo "  ✓ Found EasyBuild GCC module: $EBVERSIONGCC"
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        export FC="${FC:-gfortran}"
        export FC_EXE="$FC"
    
    # Strategy 3: Detect from NetCDF installation
    elif command -v nf-config >/dev/null 2>&1; then
        NETCDF_FC=$(nf-config --fc 2>/dev/null | awk '{print $1}' || true)
        if [ -n "$NETCDF_FC" ]; then
            echo "  ✓ NetCDF compiled with: $NETCDF_FC"
            # Try to match compiler family
            case "$NETCDF_FC" in
                *gfortran*)
                    if [[ "$NETCDF_FC" =~ gfortran-([0-9]+) ]]; then
                        GCC_VER="${BASH_REMATCH[1]}"
                        export CC="gcc-${GCC_VER}"
                        export CXX="g++-${GCC_VER}"
                        export FC="gfortran-${GCC_VER}"
                    else
                        export CC="${CC:-gcc}"
                        export CXX="${CXX:-g++}"
                        export FC="${FC:-gfortran}"
                    fi
                    ;;
                *ifort*)
                    export CC="${CC:-icc}"
                    export CXX="${CXX:-icpc}"
                    export FC="${FC:-ifort}"
                    ;;
                *)
                    export CC="${CC:-gcc}"
                    export CXX="${CXX:-g++}"
                    export FC="${FC:-gfortran}"
                    ;;
            esac
            export FC_EXE="$FC"
        fi
    
    # Strategy 4: Platform-specific search
    else
        echo "  🔍 Searching for available compilers..."
        
        # Try to find best available compiler
        CC_FOUND=""
        FC_FOUND=""
        
        # Search for C compiler
        for gcc_ver in gcc-13 gcc-12 gcc-11 gcc-10 gcc-9 gcc clang; do
            if command -v "$gcc_ver" >/dev/null 2>&1; then
                CC_FOUND="$gcc_ver"
                echo "  ✓ Found C compiler: $CC_FOUND"
                break
            fi
        done
        
        # Search for Fortran compiler
        for fc_ver in gfortran-13 gfortran-12 gfortran-11 gfortran-10 gfortran-9 gfortran ifort; do
            if command -v "$fc_ver" >/dev/null 2>&1; then
                FC_FOUND="$fc_ver"
                echo "  ✓ Found Fortran compiler: $FC_FOUND"
                break
            fi
        done
        
        # Set compilers
        export CC="${CC:-${CC_FOUND:-gcc}}"
        export CXX="${CXX:-${CXX_FOUND:-g++}}"
        export FC="${FC:-${FC_FOUND:-gfortran}}"
        export FC_EXE="$FC"
    fi
    
    # CI-specific adjustments
    if [ "$IS_CI" = "true" ]; then
        echo "  📦 Applying CI-specific settings..."
        # Conservative flags for CI
        export CMAKE_C_FLAGS="${CMAKE_C_FLAGS:-} -fPIC"
        export CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} -fPIC"
        export CMAKE_Fortran_FLAGS="${CMAKE_Fortran_FLAGS:-} -fPIC"
        export NCORES="${NCORES:-2}"
        
        # Try to install missing compilers in Ubuntu CI
        if [ "$PLATFORM" = "ubuntu" ] && [ "$IS_CI" = "true" ]; then
            if ! command -v "$CC" >/dev/null 2>&1 || ! command -v "$FC" >/dev/null 2>&1; then
                echo "  🔧 Attempting to install compilers in CI..."
                if command -v apt-get >/dev/null 2>&1; then
                    sudo apt-get update -qq 2>/dev/null || true
                    sudo apt-get install -y gcc g++ gfortran 2>/dev/null || true
                    # Reset after installation
                    export CC="gcc"
                    export CXX="g++"
                    export FC="gfortran"
                    export FC_EXE="gfortran"
                fi
            fi
        fi
    else
        # Standard flags for non-CI
        export CMAKE_C_FLAGS="${CMAKE_C_FLAGS:-} -static-libgcc -fPIC"
        export CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} -static-libgcc -static-libstdc++ -fPIC"
        export CMAKE_Fortran_FLAGS="${CMAKE_Fortran_FLAGS:-} -fPIC"
        export NCORES="${NCORES:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
    fi
    
    export CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-}"
    
    # Configure MPI compiler overrides
    export OMPI_CC="$CC"
    export OMPI_CXX="$CXX"
    export OMPI_FC="$FC"
    export MPICH_CC="$CC"
    export MPICH_CXX="$CXX"
    export MPICH_F90="$FC"
    
    export SYMFLUENCE_COMPILERS_DETECTED="yes"
    echo "  ✅ Compilers configured: CC=$CC | CXX=$CXX | FC=$FC"
    
    # Verify compilers exist
    local missing=""
    if ! command -v "$CC" >/dev/null 2>&1; then
        missing="$missing CC($CC)"
    fi
    if ! command -v "$CXX" >/dev/null 2>&1; then
        missing="$missing CXX($CXX)"
    fi
    if ! command -v "$FC" >/dev/null 2>&1; then
        missing="$missing FC($FC)"
    fi
    
    if [ -n "$missing" ]; then
        echo "  ❌ ERROR: Required compilers not found:$missing"
        echo "  💡 Install missing compilers:"
        case "$PLATFORM" in
            ubuntu|debian)
                echo "     sudo apt-get update && sudo apt-get install -y gcc g++ gfortran"
                ;;
            centos|rhel|fedora)
                echo "     sudo yum install -y gcc gcc-c++ gcc-gfortran"
                ;;
            macos)
                echo "     brew install gcc"
                ;;
            *)
                echo "     Please install GCC compiler suite (gcc, g++, gfortran)"
                ;;
        esac
        
        # Don't fail immediately in CI - try to continue
        if [ "$IS_CI" = "false" ]; then
            return 1
        fi
    fi
    
    # Show actual compiler versions for debugging
    echo "  📍 Compiler versions:"
    "$CC" --version 2>&1 | head -1 || echo "    $CC: version unknown"
    "$CXX" --version 2>&1 | head -1 || echo "    $CXX: version unknown"
    "$FC" --version 2>&1 | head -1 || echo "    $FC: version unknown"
}

# ================================================================
# LIBRARY DISCOVERY - ENHANCED
# ================================================================

# Call compiler detection immediately
detect_compilers || {
    echo "  ⚠️ Compiler detection had issues, continuing with defaults..."
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    export FC="${FC:-gfortran}"
    export FC_EXE="$FC"
}

# Discover NetCDF with better fallback
if command -v nc-config >/dev/null 2>&1; then
    export NETCDF="$(nc-config --prefix 2>/dev/null)"
    echo "  ✓ NetCDF found: $NETCDF"
elif pkg-config --exists netcdf 2>/dev/null; then
    export NETCDF="$(pkg-config --variable=prefix netcdf)"
    echo "  ✓ NetCDF found via pkg-config: $NETCDF"
else
    # Search common locations
    for dir in /usr /usr/local /opt/netcdf $HOME/local $HOME/.local; do
        if [ -f "$dir/include/netcdf.h" ]; then
            export NETCDF="$dir"
            echo "  ✓ NetCDF found: $NETCDF"
            break
        fi
    done
    if [ -z "$NETCDF" ]; then
        export NETCDF="/usr"
        echo "  ⚠️ NetCDF not detected, using default: $NETCDF"
    fi
fi

# Discover NetCDF-Fortran
if command -v nf-config >/dev/null 2>&1; then
    export NETCDF_FORTRAN="$(nf-config --prefix 2>/dev/null)"
    echo "  ✓ NetCDF-Fortran found: $NETCDF_FORTRAN"
else
    # Check if it's in the same location as NetCDF C
    if [ -f "$NETCDF/include/netcdf.mod" ] || [ -f "$NETCDF/include/NETCDF.mod" ]; then
        export NETCDF_FORTRAN="$NETCDF"
    else
        export NETCDF_FORTRAN="${NETCDF_FORTRAN:-/usr}"
    fi
fi

# Discover HDF5
if command -v h5cc >/dev/null 2>&1; then
    export HDF5_ROOT="$(h5cc -showconfig 2>/dev/null | awk -F': ' '/Installation point/{print $2}' || echo /usr)"
    echo "  ✓ HDF5 found: $HDF5_ROOT"
else
    export HDF5_ROOT="${HDF5_ROOT:-/usr}"
fi

# Python environment
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    export SYMFLUENCE_PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    export SYMFLUENCE_PYTHON="python3"
else
    export SYMFLUENCE_PYTHON="python"
fi
echo "  🐍 Using Python: ${SYMFLUENCE_PYTHON}"

echo "  🔧 Using $NCORES cores for parallel builds"
    '''.strip()


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
            'default_path_suffix': 'installs/sundials/install/sundials/',
            'default_exe': 'lib/libsundials_core.a',
            'repository': None,
            'branch': None,
            'install_dir': 'sundials',
            'build_commands': [
                common_env,
                r'''
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
    CC_COMPILER="$(which mpicc)"
    CXX_COMPILER="$(which mpicxx || which mpic++)"
else
    echo "  ℹ️ MPI not detected, building without MPI support"
    CC_COMPILER="$(which $CC)"
    CXX_COMPILER="$(which $CXX)"
fi

FC_COMPILER="$(which $FC)"

echo "📋 Configuration:"
echo "  CC: $CC_COMPILER"
echo "  CXX: $CXX_COMPILER"  
echo "  FC: $FC_COMPILER"
echo "  Install: $SUNDIALSDIR"
echo "  MPI: $USE_MPI"

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
    if [ ! -f "$LIBDIR/lib${lib}.a" ] && [ ! -f "$LIBDIR/lib${lib}.so" ]; then
        MISSING_LIBS="$MISSING_LIBS $lib"
    fi
done

if [ -n "$MISSING_LIBS" ]; then
    echo "  ❌ Missing required libraries:$MISSING_LIBS"
    echo "  📋 Available libraries:"
    ls -la "$LIBDIR" | grep -E "\.a|\.so"
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
                'file_paths': [
                    'lib64/libsundials_core.a',
                    'lib/libsundials_core.a',
                    'lib64/libsundials_core.so',
                    'lib/libsundials_core.so',
                    'include/sundials/sundials_config.h'
                ],
                'check_type': 'exists_any'
            },
            'order': 1
        },

        # ================================================================
        # SUMMA - Enhanced Build Script
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
                common_env,
                r'''
# Enhanced SUMMA build with better SUNDIALS detection
echo "🔨 Building SUMMA..."

# Find SUNDIALS installation with multiple search strategies
SUNDIALS_BASE=""

# Strategy 1: Expected relative location
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
                'file_paths': ['bin/summa_sundials.exe'],
                'check_type': 'exists'
            },
            'order': 2
        },

        # ================================================================
        # mizuRoute - River Network Routing Model
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
                common_env,
                r'''
# Build mizuRoute
echo "Building mizuRoute..."
cd route/build/

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
        # T-route - NOAA's River Routing
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
                r'''
# Install Python dependencies for t-route
echo "Setting up t-route..."
${SYMFLUENCE_PYTHON} -m pip install --upgrade pip setuptools wheel
cd src/troute-network/
${SYMFLUENCE_PYTHON} -m pip install -e . || {
    echo "⚠️ Full installation failed, trying minimal setup..."
    ${SYMFLUENCE_PYTHON} setup.py build_ext --inplace || true
}
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
            'branch': 'main',
            'install_dir': 'fuse',
            'build_commands': [
                common_env,
                r'''
# Build FUSE
echo "Building FUSE..."
cd build/

# Configure build
cat > Makefile.config <<EOF
FC = ${FC}
FC_EXE = \$(FC)
FLAGS_DEBUG = -g -O0 -ffree-line-length-none -fbacktrace
FLAGS_OPT = -O3 -ffree-line-length-none -fbacktrace
INCLUDES = -I${NETCDF}/include -I${NETCDF_FORTRAN}/include
LIBRARIES = -L${NETCDF}/lib -L${NETCDF_FORTRAN}/lib
LIBS = -lnetcdff -lnetcdf
EOF

# Build
make clean 2>/dev/null || true
make FC=${FC} FC_EXE=${FC} -j ${NCORES:-4}

# Install
mkdir -p ../bin
if [ -f "fuse.exe" ]; then
    cp fuse.exe ../bin/
    chmod +x ../bin/fuse.exe
    echo "✅ FUSE built successfully"
else
    echo "❌ FUSE build failed"
    exit 1
fi
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
        # TauDEM - Terrain Analysis Using Digital Elevation Models
        # ================================================================
        'taudem': {
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'config_exe_key': 'TAUDEM_BIN',
            'default_path_suffix': 'installs/TauDEM/bin/',
            'default_exe': 'pitremove',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': 'develop',
            'install_dir': 'TauDEM',
            'build_commands': [
                common_env,
                r'''
# Build TauDEM
echo "Building TauDEM..."

# First install any Python requirements
if [ -f requirements.txt ]; then
    ${SYMFLUENCE_PYTHON} -m pip install -r requirements.txt || true
fi

# Build with CMake
rm -rf build && mkdir build && cd build

# Configure - try with MPI if available, otherwise without
if command -v mpicc >/dev/null 2>&1; then
    echo "Building with MPI support..."
    cmake .. \
        -DCMAKE_C_COMPILER="$(which mpicc)" \
        -DCMAKE_CXX_COMPILER="$(which mpicxx || which mpic++)" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=.. \
        2>&1 | tee cmake.log
else
    echo "Building without MPI..."
    cmake .. \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=.. \
        2>&1 | tee cmake.log
fi

# Build
make -j ${NCORES:-4} 2>&1 | tee build.log

# Install or copy executables
make install 2>/dev/null || {
    echo "Make install failed, copying executables manually..."
    mkdir -p ../bin
    find . -type f -executable -name "*.exe" -o -name "pitremove" -o -name "d8flowdir" \
        -o -name "aread8" -o -name "gridnet" -o -name "threshold" -o -name "streamnet" | \
    while read exe; do
        cp "$exe" ../bin/ 2>/dev/null || true
    done
}

# Verify critical executables
cd ..
if [ -f "bin/pitremove" ] || [ -f "bin/pitremove.exe" ]; then
    echo "✅ TauDEM built successfully"
    ls -la bin/ | head -20
else
    echo "⚠️ Some TauDEM executables may be missing"
fi
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
echo "✅ GIStool configured"
                '''
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
echo "✅ Datatool configured"
                '''
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
                common_env,
                r'''
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
            'test_command': '--help',
            'verify_install': {
                'file_paths': ['cmake_build/ngen'],
                'check_type': 'exists'
            },
            'order': 9
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
# Detect HPC vs laptop/workstation and fetch the right NGIAB wrapper
IS_HPC=false
for scheduler in sbatch qsub bsub; do
    if command -v $scheduler >/dev/null 2>&1; then 
        IS_HPC=true
        break
    fi
done
[ -n "$SLURM_CLUSTER_NAME" ] && IS_HPC=true
[ -n "$PBS_JOBID" ] && IS_HPC=true
[ -d "/scratch" ] && IS_HPC=true

if $IS_HPC; then
    NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
    echo "HPC environment detected; using NGIAB-HPCInfra"
else
    NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
    echo "Non-HPC environment detected; using NGIAB-CloudInfra"
fi

# Clone appropriate repository
cd ..
rm -rf ngiab
git clone "$NGIAB_REPO" ngiab
cd ngiab

# Make guide.sh executable
if [ -f guide.sh ]; then
    chmod +x guide.sh
    echo "✅ NGIAB configured"
else
    echo "⚠️ guide.sh not found"
fi
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