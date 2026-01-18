"""
NGEN build instructions for SYMFLUENCE.

This module defines how to build NGEN from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

NGEN is the NextGen National Water Model Framework developed by NOAA/NWS.
It supports multiple BMI-compliant model modules including CFE, PET,
NOAH-OWP-Modular, and SLOTH.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
    get_udunits2_detection_and_build,
)


@BuildInstructionsRegistry.register('ngen')
def get_ngen_build_instructions():
    """
    Get NGEN build instructions.

    NGEN uses CMake and requires Boost, NetCDF, and optionally Python
    and Fortran support for various BMI modules.

    Returns:
        Dictionary with complete build configuration for NGEN.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()
    udunits2_detect = get_udunits2_detection_and_build()

    return {
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
            netcdf_detect,
            udunits2_detect,
            r'''
set -e
echo "Building ngen with full BMI support (C, C++, Fortran)..."

# Ensure system tools are preferred (fix for 2i2c environments)
export PATH="/usr/bin:$PATH"

# Prevent any Makefile from being auto-triggered during git operations
export MAKEFLAGS=""
export MAKELEVEL=""

# Fix for conda GCC 14: ensure libstdc++ is found
# GCC 14 from conda-forge requires explicit library path for C++ runtime
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    echo "Added conda lib to library paths: $CONDA_PREFIX/lib"
fi

# Detect venv Python - prefer VIRTUAL_ENV, otherwise use which python3
if [ -n "$VIRTUAL_ENV" ]; then
  PYTHON_EXE="$VIRTUAL_ENV/bin/python3"
elif [ -n "$CONDA_PREFIX" ]; then
  PYTHON_EXE="$CONDA_PREFIX/bin/python3"
else
  PYTHON_EXE=$(which python3)
fi
echo "Using Python: $PYTHON_EXE"
$PYTHON_EXE -c "import numpy as np; print('Using NumPy:', np.__version__)"

# Boost (local)
if [ ! -d "boost_1_79_0" ]; then
  echo "Fetching Boost 1.79.0..."
  (wget -q https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 -O boost_1_79_0.tar.bz2 \
    || curl -fsSL -o boost_1_79_0.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2)
  tar -xjf boost_1_79_0.tar.bz2 && rm -f boost_1_79_0.tar.bz2
fi
export BOOST_ROOT="$(pwd)/boost_1_79_0"
export CXX=${CXX:-g++}

# Initialize ALL submodules needed for full BMI support
echo "Initializing submodules for ngen and external BMI modules..."
git submodule update --init --recursive -- test/googletest extern/pybind11 || true
git submodule update --init --recursive -- extern/cfe extern/evapotranspiration extern/sloth extern/noah-owp-modular || true

# Initialize t-route submodule for routing support
echo "Initializing t-route submodule for routing..."
git submodule update --init --recursive -- extern/t-route || true

# Initialize iso_c_fortran_bmi for Fortran BMI support (required for NOAH-OWP)
echo "Initializing iso_c_fortran_bmi submodule..."
git submodule update --init --recursive -- extern/iso_c_fortran_bmi || true

# Verify Fortran compiler
echo "Checking Fortran compiler..."
if command -v gfortran >/dev/null 2>&1; then
  export FC=$(command -v gfortran)
  echo "Using Fortran compiler: $FC"
  $FC --version | head -1
else
  echo "WARNING: gfortran not found, Fortran BMI modules will be disabled"
  export NGEN_WITH_BMI_FORTRAN=OFF
fi

rm -rf cmake_build

# Debug: show environment
echo "=== Environment Debug ==="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
echo "UDUNITS2_DIR: ${UDUNITS2_DIR:-not set}"
echo "UDUNITS2_INCLUDE_DIR: ${UDUNITS2_INCLUDE_DIR:-not set}"
echo "UDUNITS2_LIBRARY: ${UDUNITS2_LIBRARY:-not set}"
echo "========================="

# Build ngen with full BMI support including Fortran
echo "Configuring ngen with BMI C, C++, and Fortran support..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DBOOST_ROOT=$BOOST_ROOT"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_SQLITE3=ON"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_C=ON"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_CPP=ON"

# Fix for conda GCC 14: explicitly link libstdc++ to resolve __cxa_call_terminate
# This is needed because conda's GCC uses a separate libstdc++ that may not be auto-linked
if [ -n "$CONDA_PREFIX" ]; then
    EXTRA_LIBS="-lstdc++"
    echo "Adding -lstdc++ for conda GCC 14 compatibility"
fi

# Add UDUNITS2 paths if available (from detection/build snippet)
if [ -n "${UDUNITS2_INCLUDE_DIR:-}" ] && [ -n "${UDUNITS2_LIBRARY:-}" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_ROOT=$UDUNITS2_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_INCLUDE_DIR=$UDUNITS2_INCLUDE_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_LIBRARY=$UDUNITS2_LIBRARY"

  # Also add to compiler flags
  export CXXFLAGS="${CXXFLAGS:-} -I${UDUNITS2_INCLUDE_DIR}"
  export CFLAGS="${CFLAGS:-} -I${UDUNITS2_INCLUDE_DIR}"

  echo "Using UDUNITS2 from: $UDUNITS2_DIR"
fi

# Add extra linker flags for conda GCC 14 and expat (needed by UDUNITS2)
# Use CMAKE_EXE_LINKER_FLAGS which properly handles space-separated flags
EXTRA_LDFLAGS="${EXTRA_LIBS:-}"
if [ -n "${UDUNITS2_LIBRARY:-}" ]; then
  EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lexpat"
fi
if [ -n "$EXTRA_LDFLAGS" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_EXE_LINKER_FLAGS='$EXTRA_LDFLAGS'"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_SHARED_LINKER_FLAGS='$EXTRA_LDFLAGS'"
  echo "Adding extra linker flags: $EXTRA_LDFLAGS"
fi

# Add Fortran support if compiler is available
if [ "${NGEN_WITH_BMI_FORTRAN:-ON}" = "ON" ] && [ -n "$FC" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_FORTRAN=ON"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"

  # Configure iso_c_fortran_bmi (C wrapper for Fortran BMI modules)
  # This provides the register_bmi function that NGEN needs to load Fortran modules
  ISO_C_BMI_DIR="$(pwd)/extern/iso_c_fortran_bmi/cmake_build"
  CMAKE_ARGS="$CMAKE_ARGS -DBMI_FORTRAN_ISO_C_LIB_DIR=$ISO_C_BMI_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DBMI_FORTRAN_ISO_C_LIB_NAME=iso_c_bmi"

  echo "Enabling Fortran BMI support with iso_c_bmi wrapper"
fi

# Check NumPy version - ngen doesn't support NumPy 2.x yet
NUMPY_VERSION=$($PYTHON_EXE -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0")
NUMPY_MAJOR=$(echo "$NUMPY_VERSION" | cut -d. -f1)
if [ "$NUMPY_MAJOR" -ge 2 ] 2>/dev/null; then
  echo "NumPy $NUMPY_VERSION detected (>=2.0). Disabling Python and routing support (not yet compatible with ngen)."
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_PYTHON=OFF"
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_ROUTING=OFF"
else
  # Add Python support for NumPy 1.x
  echo "NumPy $NUMPY_VERSION detected. Enabling Python and t-route routing support."
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_PYTHON=ON"
  CMAKE_ARGS="$CMAKE_ARGS -DPython_EXECUTABLE=$PYTHON_EXE"
  CMAKE_ARGS="$CMAKE_ARGS -DPython3_EXECUTABLE=$PYTHON_EXE"
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_ROUTING=ON"
fi

# Configure ngen
echo "Running CMake with args: $CMAKE_ARGS"
if cmake $CMAKE_ARGS -S . -B cmake_build 2>&1 | tee cmake_config.log; then
  echo "ngen configured successfully"
else
  echo "CMake configuration failed, checking log..."
  tail -30 cmake_config.log
  echo ""
  echo "Retrying with Python OFF but keeping Fortran support..."
  rm -rf cmake_build

  # Keep Fortran support in fallback - it's required for NOAH-OWP!
  FALLBACK_ARGS="-DCMAKE_BUILD_TYPE=Release"
  FALLBACK_ARGS="$FALLBACK_ARGS -DBOOST_ROOT=$BOOST_ROOT"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_PYTHON=OFF"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_ROUTING=OFF"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_SQLITE3=ON"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_C=ON"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_CPP=ON"

  # Add UDUNITS2 paths to fallback as well
  if [ -n "${UDUNITS2_INCLUDE_DIR:-}" ] && [ -n "${UDUNITS2_LIBRARY:-}" ]; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_ROOT=$UDUNITS2_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_INCLUDE_DIR=$UDUNITS2_INCLUDE_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_LIBRARY=$UDUNITS2_LIBRARY"
  fi

  # Add extra linker flags to fallback as well
  if [ -n "$EXTRA_LDFLAGS" ]; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DCMAKE_EXE_LINKER_FLAGS='$EXTRA_LDFLAGS'"
    FALLBACK_ARGS="$FALLBACK_ARGS -DCMAKE_SHARED_LINKER_FLAGS='$EXTRA_LDFLAGS'"
  fi

  # Keep Fortran in fallback if compiler is available
  if [ -n "$FC" ]; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_FORTRAN=ON"
    FALLBACK_ARGS="$FALLBACK_ARGS -DCMAKE_Fortran_COMPILER=$FC"

    # Include iso_c_bmi configuration in fallback
    ISO_C_BMI_DIR="$(pwd)/extern/iso_c_fortran_bmi/cmake_build"
    FALLBACK_ARGS="$FALLBACK_ARGS -DBMI_FORTRAN_ISO_C_LIB_DIR=$ISO_C_BMI_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DBMI_FORTRAN_ISO_C_LIB_NAME=iso_c_bmi"

    echo "Fallback: keeping Fortran BMI support with iso_c_bmi wrapper"
  fi

  cmake $FALLBACK_ARGS -S . -B cmake_build
fi

# Build ngen executable
echo "Building ngen..."
cmake --build cmake_build --target ngen -j ${NCORES:-4}

# Verify ngen binary
if [ -x "cmake_build/ngen" ]; then
  echo "ngen built successfully"
  ./cmake_build/ngen --help 2>/dev/null | head -5 || true
else
  echo "ngen binary not found"
  exit 1
fi

# ================================================================
# Build External BMI Modules (CFE, PET, SLOTH, NOAH-OWP-Modular)
# ================================================================
echo ""
echo "Building external BMI modules..."

# --- Build SLOTH (C++ module for soil/ice fractions) ---
if [ -d "extern/sloth" ]; then
  echo "Building SLOTH..."
  cd extern/sloth
  git submodule update --init --recursive || true
  rm -rf cmake_build && mkdir -p cmake_build
  cmake -DCMAKE_BUILD_TYPE=Release -S . -B cmake_build
  cmake --build cmake_build -j ${NCORES:-4}
  if [ -f cmake_build/libslothmodel.* ]; then
    echo "SLOTH built successfully"
  else
    echo "SLOTH library not found (non-fatal)"
  fi
  cd ../..
fi

# --- Build CFE (C module - Conceptual Functional Equivalent) ---
if [ -d "extern/cfe" ]; then
  echo "Building CFE..."
  cd extern/cfe
  git submodule update --init --recursive || true
  rm -rf cmake_build && mkdir -p cmake_build
  cmake -DCMAKE_BUILD_TYPE=Release -S . -B cmake_build
  cmake --build cmake_build -j ${NCORES:-4}
  if [ -f cmake_build/libcfebmi.* ]; then
    echo "CFE built successfully"
  else
    echo "CFE library not found (non-fatal)"
  fi
  cd ../..
fi

# --- Build evapotranspiration/PET (C module) ---
if [ -d "extern/evapotranspiration" ]; then
  echo "Building PET (evapotranspiration)..."
  cd extern/evapotranspiration/evapotranspiration
  git submodule update --init --recursive 2>/dev/null || true
  rm -rf cmake_build && mkdir -p cmake_build
  cmake -DCMAKE_BUILD_TYPE=Release -S . -B cmake_build
  cmake --build cmake_build -j ${NCORES:-4}
  if [ -f cmake_build/libpetbmi.* ]; then
    echo "PET built successfully"
  else
    echo "PET library not found (non-fatal)"
  fi
  cd ../../..
fi

# --- Build iso_c_fortran_bmi (C wrapper for Fortran BMI) ---
# This must be built BEFORE NOAH-OWP as it provides the registration interface
if [ -d "extern/iso_c_fortran_bmi" ] && [ -n "$FC" ]; then
  echo "Building iso_c_fortran_bmi (C wrapper for Fortran BMI)..."
  cd extern/iso_c_fortran_bmi
  git submodule update --init --recursive || true
  rm -rf cmake_build && mkdir -p cmake_build

  ISO_C_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
  ISO_C_CMAKE_ARGS="$ISO_C_CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"

  cmake $ISO_C_CMAKE_ARGS -S . -B cmake_build
  cmake --build cmake_build -j ${NCORES:-4}

  if [ -f cmake_build/libiso_c_bmi.* ]; then
    echo "iso_c_fortran_bmi built successfully"
  else
    echo "WARNING: iso_c_bmi library not found - NOAH-OWP will fail"
  fi
  cd ../..
fi

# --- Build NOAH-OWP-Modular (Fortran module) ---
if [ -d "extern/noah-owp-modular" ] && [ -n "$FC" ]; then
  echo "Building NOAH-OWP-Modular (Fortran)..."
  cd extern/noah-owp-modular
  git submodule update --init --recursive || true
  rm -rf cmake_build && mkdir -p cmake_build

  # Configure with NGEN support
  # NGEN_IS_MAIN_PROJECT=ON triggers compile definitions (NGEN_ACTIVE, etc.)
  # and builds iso_c_fortran_bmi as a subdirectory for Fortran BMI support
  NOAH_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
  NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"
  NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DNGEN_IS_MAIN_PROJECT=ON"

  if [ -n "$NETCDF_FORTRAN" ]; then
    NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DNETCDF_PATH=$NETCDF_FORTRAN"
  fi

  cmake $NOAH_CMAKE_ARGS -S . -B cmake_build
  cmake --build cmake_build -j ${NCORES:-4}

  if [ -f cmake_build/libsurfacebmi.* ]; then
    echo "NOAH-OWP-Modular built successfully"
  else
    echo "NOAH-OWP library not found (non-fatal)"
  fi
  cd ../..
else
  if [ ! -d "extern/noah-owp-modular" ]; then
    echo "NOAH-OWP-Modular submodule not found - skipping"
  elif [ -z "$FC" ]; then
    echo "No Fortran compiler available - skipping NOAH-OWP build"
  fi
fi

# ================================================================
# Install t-route (Python packages for routing)
# ================================================================
if [ -d "extern/t-route/src" ]; then
  echo ""
  echo "Installing t-route Python packages..."

  # Install python_routing_v02 (core troute routing package)
  if [ -d "extern/t-route/src/python_routing_v02" ]; then
    echo "Installing python_routing_v02 (troute)..."
    cd extern/t-route/src/python_routing_v02
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: python_routing_v02 installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install python_framework_v02 (troute framework)
  if [ -d "extern/t-route/src/python_framework_v02" ]; then
    echo "Installing python_framework_v02..."
    cd extern/t-route/src/python_framework_v02
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: python_framework_v02 installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install nwm_routing (required dependency for ngen_routing)
  if [ -d "extern/t-route/src/nwm_routing" ]; then
    echo "Installing nwm_routing..."
    cd extern/t-route/src/nwm_routing
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: nwm_routing installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install ngen_routing (main routing interface)
  if [ -d "extern/t-route/src/ngen_routing" ]; then
    echo "Installing ngen_routing..."
    cd extern/t-route/src/ngen_routing
    $PYTHON_EXE -m pip install -e . --no-deps || {
      echo "WARNING: ngen_routing installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Verify installations
  if $PYTHON_EXE -c "import nwm_routing" 2>/dev/null; then
    echo "t-route nwm_routing installed successfully"
  else
    echo "t-route nwm_routing not available (non-fatal)"
  fi

  if $PYTHON_EXE -c "import ngen_routing" 2>/dev/null; then
    echo "t-route ngen_routing installed successfully"
  else
    echo "t-route ngen_routing not available (non-fatal)"
  fi
else
  echo "t-route submodule not found - routing will not be available"
fi

echo ""
echo "=============================================="
echo "ngen build summary:"
echo "=============================================="
echo "ngen binary: $([ -x cmake_build/ngen ] && echo 'OK' || echo 'MISSING')"
echo "SLOTH:       $([ -f extern/sloth/cmake_build/libslothmodel.* ] 2>/dev/null && echo 'OK' || echo 'Not built')"
echo "CFE:         $([ -f extern/cfe/cmake_build/libcfebmi.* ] 2>/dev/null && echo 'OK' || echo 'Not built')"
echo "PET:         $([ -f extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi.* ] 2>/dev/null && echo 'OK' || echo 'Not built')"
echo "NOAH-OWP:    $([ -f extern/noah-owp-modular/cmake_build/libsurfacebmi.* ] 2>/dev/null && echo 'OK' || echo 'Not built')"
echo "t-route:     $($PYTHON_EXE -c 'import ngen_routing; print(\"OK\")' 2>/dev/null || echo 'Not installed')"
echo "=============================================="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--help',
        'verify_install': {
            'file_paths': ['cmake_build/ngen'],
            'check_type': 'exists'
        },
        'order': 9
    }
