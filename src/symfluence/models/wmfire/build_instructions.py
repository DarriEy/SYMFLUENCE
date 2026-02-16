"""
WMFire build instructions for SYMFLUENCE.

This module defines how to build WMFire (Wildfire-Model Fire) from source.
WMFire is a fire spread model that couples with RHESSys for simulating
wildfire effects on ecohydrological processes.

The build produces a shared library (libwmfire.so on Linux, libwmfire.dylib on macOS)
that RHESSys links against when fire spread simulation is enabled.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('wmfire')
def get_wmfire_build_instructions():
    """
    Get WMFire build instructions.

    WMFire is built as a shared library from C++ source in the RHESSys FIRE directory.
    It requires Boost headers for random number generation.

    Returns:
        Dictionary with complete build configuration for WMFire.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'WMFire - Wildfire spread model for RHESSys coupling',
        'config_path_key': 'WMFIRE_INSTALL_PATH',
        'config_exe_key': 'WMFIRE_LIB',
        'default_path_suffix': 'installs/wmfire/lib',
        'default_exe': 'libwmfire.so',
        'repository': 'https://github.com/RHESSys/RHESSys.git',
        'branch': None,
        'install_dir': 'wmfire',
        'build_commands': [
            common_env,
            r'''
set -e
echo "Building WMFire library..."

# Save install root for output paths
INSTALL_ROOT="$(pwd)"

# The WMFire source is in the FIRE directory of the RHESSys repo.
# When cloned by tool_installer, repo contents are directly in the install dir.
if [ -d "FIRE" ]; then
    cd FIRE
elif [ -d "RHESSys/FIRE" ]; then
    cd RHESSys/FIRE
else
    echo "ERROR: FIRE directory not found (checked FIRE/ and RHESSys/FIRE/)"
    exit 1
fi

# Detect platform
OS=$(uname -s)
case "$OS" in
    Darwin)
        WMFIRE_LIB="libwmfire.dylib"
        SHARED_FLAG="-dynamiclib"
        ;;
    MSYS*|MINGW*|CYGWIN*)
        WMFIRE_LIB="libwmfire.dll"
        SHARED_FLAG="-shared"
        ;;
    *)
        WMFIRE_LIB="libwmfire.so"
        SHARED_FLAG="-shared"
        ;;
esac

# Find Boost headers
# Check common locations for Boost
BOOST_INCLUDE=""

# Check if boost is in conda environment (CONDA_LIB_PREFIX handles Windows)
clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/include/boost" ]; then
    BOOST_INCLUDE="-I$clp/include"
    echo "Found Boost in conda environment: $clp/include"
fi

# Check homebrew on macOS
if [ -z "$BOOST_INCLUDE" ] && [ "$OS" = "Darwin" ]; then
    if [ -d "/opt/homebrew/include/boost" ]; then
        BOOST_INCLUDE="-I/opt/homebrew/include"
        echo "Found Boost in homebrew (arm64): /opt/homebrew/include"
    elif [ -d "/usr/local/include/boost" ]; then
        BOOST_INCLUDE="-I/usr/local/include"
        echo "Found Boost in homebrew (x86_64): /usr/local/include"
    fi
fi

# Check system locations
if [ -z "$BOOST_INCLUDE" ]; then
    for boost_dir in /usr/include /usr/local/include; do
        if [ -d "$boost_dir/boost" ]; then
            BOOST_INCLUDE="-I$boost_dir"
            echo "Found Boost in system: $boost_dir"
            break
        fi
    done
fi

# Check BOOST_ROOT environment variable
if [ -z "$BOOST_INCLUDE" ] && [ -n "$BOOST_ROOT" ]; then
    if [ -d "$BOOST_ROOT/include/boost" ]; then
        BOOST_INCLUDE="-I$BOOST_ROOT/include"
    elif [ -d "$BOOST_ROOT/boost" ]; then
        BOOST_INCLUDE="-I$BOOST_ROOT"
    fi
    echo "Found Boost via BOOST_ROOT: $BOOST_INCLUDE"
fi

# Search Spack install trees (HPC systems)
if [ -z "$BOOST_INCLUDE" ]; then
    for spack_root in /apps/spack /opt/spack; do
        [ -d "$spack_root" ] || continue
        _boost_inc=$(find "$spack_root" -path "*/boost/*/include/boost/random.hpp" -type f 2>/dev/null | sort -rV | head -1)
        if [ -n "$_boost_inc" ]; then
            # Go from .../include/boost/random.hpp up to .../include
            BOOST_INCLUDE="-I$(dirname "$(dirname "$_boost_inc")")"
            echo "Found Boost in Spack tree: $BOOST_INCLUDE"
            break
        fi
    done
fi

if [ -z "$BOOST_INCLUDE" ]; then
    echo "WARNING: Boost headers not found. WMFire requires Boost for random number generation."
    echo "Install Boost with: brew install boost (macOS) or apt-get install libboost-dev (Linux)"
    echo "Attempting build anyway..."
    BOOST_INCLUDE=""
fi

# Use C++ compiler
CXX=${CXX:-g++}
echo "Using C++ compiler: $CXX"

# Compile object files
echo "Compiling RanNums.cpp..."
$CXX -c -fPIC $BOOST_INCLUDE -O2 -o RanNums.o RanNums.cpp

echo "Compiling WMFire.cpp..."
$CXX -c -fPIC $BOOST_INCLUDE -O2 -o WMFire.o WMFire.cpp

# Link into shared library
echo "Linking $WMFIRE_LIB..."
$CXX $SHARED_FLAG -fPIC -o $WMFIRE_LIB RanNums.o WMFire.o

# Install to destination
mkdir -p $INSTALL_ROOT/lib
mv $WMFIRE_LIB $INSTALL_ROOT/lib/
echo "Installed $WMFIRE_LIB to lib/"

# Clean up object files
rm -f RanNums.o WMFire.o

# Verify build
if [ -f "$INSTALL_ROOT/lib/$WMFIRE_LIB" ]; then
    echo "WMFire library successfully built: lib/$WMFIRE_LIB"
    ls -la $INSTALL_ROOT/lib/$WMFIRE_LIB
else
    echo "ERROR: WMFire library not found after build"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,  # Library, not executable
        'verify_install': {
            'file_paths': ['lib/libwmfire.so', 'lib/libwmfire.dylib', 'lib/libwmfire.dll'],
            'check_type': 'exists_any'
        },
        'order': 13,  # Build before RHESSys (order 14)
        'optional': True,  # Not installed by default with --install
    }
