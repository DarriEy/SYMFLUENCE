"""
CRHM build instructions for SYMFLUENCE.

This module defines how to build CRHM from source, including:
- Repository and branch information
- Build commands (CMake + C++)
- Installation verification criteria

CRHM (Cold Regions Hydrological Model) is built from C++ source
using CMake. It provides physically-based cold-region hydrology
including PBSM (Prairie Blowing Snow Model) and EBSM (Energy
Balance Snow Model).

The source code lives in crhmcode/src/ within the repository,
and requires spdlog (git submodule) and Boost (header-only).
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('crhm')
def get_crhm_build_instructions():
    """
    Get CRHM build instructions.

    CRHM is a C++ project built with CMake. The repo structure has
    source in crhmcode/src/ with spdlog as a git submodule and
    Boost as a header-only dependency.

    Returns:
        Dictionary with complete build configuration for CRHM.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'Cold Regions Hydrological Model',
        'config_path_key': 'CRHM_INSTALL_PATH',
        'config_exe_key': 'CRHM_EXE',
        'default_path_suffix': 'installs/crhm/bin',
        'default_exe': 'crhm',
        'repository': 'https://github.com/CentreForHydrology/CRHM.git',
        'branch': 'master',
        'install_dir': 'crhm',
        'build_commands': [
            common_env,
            r'''
# CRHM Build Script for SYMFLUENCE
# Builds CRHM from C++ source using CMake
# Source is in crhmcode/src/, requires spdlog submodule and Boost headers

set -e

echo "=== CRHM Build Starting ==="
echo "Building CRHM (Cold Regions Hydrological Model)"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Check for CMake
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install cmake."
    exit 1
fi

# Check for C++ compiler
if command -v g++ >/dev/null 2>&1; then
    CXX_COMPILER="g++"
elif command -v c++ >/dev/null 2>&1; then
    CXX_COMPILER="c++"
elif command -v clang++ >/dev/null 2>&1; then
    CXX_COMPILER="clang++"
else
    echo "ERROR: No C++ compiler found (need g++, c++, or clang++)"
    exit 1
fi
echo "Using C++ compiler: $CXX_COMPILER"
echo "CMake version: $(cmake --version | head -1)"

# Initialize git submodules (spdlog)
echo "Initializing git submodules..."
git submodule update --init --recursive 2>/dev/null || {
    echo "WARNING: git submodule init failed, attempting manual spdlog clone"
    if [ ! -f "crhmcode/src/libs/spdlog/CMakeLists.txt" ]; then
        rm -rf crhmcode/src/libs/spdlog
        git clone --depth 1 https://github.com/gabime/spdlog.git crhmcode/src/libs/spdlog
    fi
}

# Check that spdlog is present
if [ ! -d "crhmcode/src/libs/spdlog/include" ]; then
    echo "ERROR: spdlog not found after submodule init"
    exit 1
fi
echo "spdlog: OK"

# Handle Boost dependency (header-only)
CRHM_SRC="crhmcode/src"
BOOST_DIR="$CRHM_SRC/libs/boost_1_75_0"

if [ ! -d "$BOOST_DIR" ]; then
    echo "Boost headers not found in repo, checking system..."

    # Check for system Boost
    SYSTEM_BOOST=""
    if [ -d "/opt/homebrew/include/boost" ]; then
        SYSTEM_BOOST="/opt/homebrew/include"
    elif [ -d "/usr/local/include/boost" ]; then
        SYSTEM_BOOST="/usr/local/include"
    elif [ -d "/usr/include/boost" ]; then
        SYSTEM_BOOST="/usr/include"
    fi

    if [ -n "$SYSTEM_BOOST" ]; then
        echo "Using system Boost from: $SYSTEM_BOOST"
        # Create symlink so CMakeLists.txt LOCAL_BOOST_LOCATION works
        mkdir -p "$CRHM_SRC/libs"
        ln -sf "$SYSTEM_BOOST" "$BOOST_DIR"
    else
        echo "Downloading Boost headers (header-only subset)..."
        BOOST_VER="1.75.0"
        BOOST_URL="https://archives.boost.io/release/${BOOST_VER}/source/boost_1_75_0.tar.gz"
        mkdir -p "$CRHM_SRC/libs"
        curl -sL "$BOOST_URL" | tar xz -C "$CRHM_SRC/libs/"
        if [ ! -d "$BOOST_DIR" ]; then
            echo "ERROR: Boost download/extract failed"
            exit 1
        fi
    fi
fi
echo "Boost: OK"

# Navigate to source and build
cd "$CRHM_SRC"
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    -DCMAKE_INSTALL_PREFIX=../../install

# Build
echo "Building CRHM..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
cmake --build . --config Release -j${NCORES}

echo "Build successful!"

# Install via CMake
cmake --install . 2>/dev/null || true

# Find the built executable
CRHM_EXE=""
for candidate in crhm CRHM crhm.exe CRHM.exe; do
    if [ -f "$candidate" ]; then
        CRHM_EXE="$candidate"
        break
    fi
    if [ -f "src/$candidate" ]; then
        CRHM_EXE="src/$candidate"
        break
    fi
    if [ -f "Release/$candidate" ]; then
        CRHM_EXE="Release/$candidate"
        break
    fi
done

# Check install directory
if [ -z "$CRHM_EXE" ]; then
    for candidate in ../../install/bin/crhm ../../install/bin/CRHM; do
        if [ -f "$candidate" ]; then
            CRHM_EXE="$candidate"
            break
        fi
    done
fi

if [ -z "$CRHM_EXE" ]; then
    echo "Searching recursively..."
    CRHM_EXE=$(find . ../../install -name "crhm" -o -name "CRHM" 2>/dev/null | head -1)
    if [ -z "$CRHM_EXE" ]; then
        echo "ERROR: crhm executable not found after build"
        find . -type f -perm +111 2>/dev/null
        ls -la
        exit 1
    fi
fi

echo "Found executable: $CRHM_EXE"

# Install to top-level bin/
# From crhmcode/src/build, top-level is ../../../
mkdir -p ../../../bin
cp "$CRHM_EXE" ../../../bin/crhm
chmod +x ../../../bin/crhm

echo "=== CRHM Build Complete ==="
echo "Installed to: bin/crhm"

# Verify installation
if [ -f "../../../bin/crhm" ]; then
    echo "Verification: crhm exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'g++'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/crhm'],
            'check_type': 'exists'
        },
        'order': 20  # After core models
    }
