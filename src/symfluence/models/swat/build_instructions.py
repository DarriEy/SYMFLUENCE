"""
SWAT build instructions for SYMFLUENCE.

This module defines how to build SWAT from source, including:
- Repository and branch information
- Build commands (CMake + gfortran)
- Installation verification criteria

SWAT is built from Fortran source code using CMake + gfortran. The build
produces the swat_rel.exe executable for watershed simulations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('swat')
def get_swat_build_instructions():
    """
    Get SWAT build instructions.

    SWAT is compiled from Fortran source using CMake + gfortran.
    The repo includes a CMakeLists.txt that handles module dependency
    ordering (modparm.f must be compiled before other sources).

    Returns:
        Dictionary with complete build configuration for SWAT.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'Soil and Water Assessment Tool',
        'config_path_key': 'SWAT_INSTALL_PATH',
        'config_exe_key': 'SWAT_EXE',
        'default_path_suffix': 'installs/swat/bin',
        'default_exe': 'swat_rel.exe',
        'repository': 'https://github.com/WatershedModels/SWAT.git',
        'branch': 'master',
        'install_dir': 'swat',
        'build_commands': [
            common_env,
            r'''
# SWAT Build Script for SYMFLUENCE
# Builds SWAT using CMake + gfortran (handles module dependencies)

set -e

echo "=== SWAT Build Starting ==="
echo "Building SWAT with CMake + gfortran"

# Check for required tools
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install cmake."
    echo "  macOS: brew install cmake"
    echo "  Ubuntu: sudo apt-get install cmake"
    exit 1
fi

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Please install gfortran."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

echo "CMake version: $(cmake --version | head -1)"
echo "gfortran version: $(gfortran --version | head -1)"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Set up Fortran flags for gfortran 10+ compatibility
EXTRA_FFLAGS="-fno-automatic -fno-align-commons"
GFORTRAN_MAJOR=$(gfortran -dumpversion | cut -d. -f1)
if [ "$GFORTRAN_MAJOR" -ge 10 ] 2>/dev/null; then
    echo "gfortran >= 10 detected, adding -fallow-argument-mismatch"
    EXTRA_FFLAGS="$EXTRA_FFLAGS -fallow-argument-mismatch"
fi

# Patch known source code bugs before building
echo "Applying source code patches..."

# Fix missing comma in std3.f format descriptor (line 47)
# Original: t97'------   Fixed: t97,'------
if grep -q "t97'" src/std3.f 2>/dev/null; then
    sed -i.bak "s/t97'/t97,'/" src/std3.f
    echo "  Patched std3.f: missing comma in format 1300"
fi

# Disable static linking. The upstream SWAT CMakeLists.txt has
# OPTION(ENABLE_STATIC_LINKING ... ON) which loads cmake/StaticLinking.cmake
# and appends -static to the Fortran linker flags. This fails on HPC where
# the system linker can't find static libc/libm even though Spack gfortran
# has its own copies. Dynamic linking works everywhere.
# Also remove -static from cmake module files as a belt-and-suspenders measure.
for cmf in CMakeLists.txt src/CMakeLists.txt cmake/*.cmake; do
    if [ -f "$cmf" ] && grep -q '\-static' "$cmf" 2>/dev/null; then
        sed -i.bak 's/-static //g; s/ -static//g' "$cmf"
        echo "  Patched $cmf: removed -static flag"
    fi
done

# Clean and create build directory (prevents stale CMake cache with old flags)
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "Configuring SWAT with CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_Fortran_COMPILER=gfortran
    -DCMAKE_INSTALL_PREFIX=../install
    -DENABLE_STATIC_LINKING=OFF
    "-DCMAKE_Fortran_FLAGS_RELEASE=-O2 $EXTRA_FFLAGS"
)

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo "Building SWAT..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
cmake --build . --config Release -j${NCORES}

echo "Build successful!"

# Install via CMake install target
cmake --install . 2>/dev/null || true

# Find the built executable
# The upstream CMakeLists names the binary after the version string,
# e.g. swat2012.692.gfort.rel, so we search by pattern not exact name.
SWAT_EXE=""

# 1. Check cmake install directory first (cmake --install puts it here)
SWAT_EXE=$(find ../install -name "swat*" -type f 2>/dev/null | head -1)

# 2. Check build/src directory (where cmake builds it)
if [ -z "$SWAT_EXE" ]; then
    SWAT_EXE=$(find . -name "swat*" -type f ! -path "*/CMakeFiles/*" 2>/dev/null | head -1)
fi

if [ -z "$SWAT_EXE" ]; then
    echo "ERROR: SWAT executable not found after build"
    find . ../install -type f -name "swat*" 2>/dev/null || true
    exit 1
fi

echo "Found executable: $SWAT_EXE"

# Create bin directory and install as swat_rel.exe
mkdir -p ../bin
cp "$SWAT_EXE" ../bin/swat_rel.exe
chmod +x ../bin/swat_rel.exe

echo "=== SWAT Build Complete ==="
echo "Installed to: bin/swat_rel.exe"

# Verify installation
if [ -f "../bin/swat_rel.exe" ]; then
    echo "Verification: swat_rel.exe exists"
    ls -la ../bin/swat_rel.exe
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'gfortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/swat_rel.exe'],
            'check_type': 'exists'
        },
        'order': 16  # After VIC
    }
