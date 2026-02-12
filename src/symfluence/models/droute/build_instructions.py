"""
dRoute build instructions for SYMFLUENCE.

This module defines how to build dRoute from source, including:
- Repository and branch information
- Build commands (CMake with pybind11)
- Installation verification criteria

dRoute is a C++ river routing library with Python bindings and
optional automatic differentiation support for gradient-based calibration.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('droute')
def get_droute_build_instructions():
    """
    Get dRoute build instructions.

    dRoute requires CMake, a C++ compiler (GCC/Clang), and optionally
    pybind11 for Python bindings and CoDiPack/Enzyme for AD support.

    Returns:
        Dictionary with complete build configuration for dRoute.
    """
    return {
        'description': 'dRoute river routing library with AD support',
        'config_path_key': 'DROUTE_INSTALL_PATH',
        'config_exe_key': 'DROUTE_EXE',
        'default_path_suffix': 'installs/droute',
        'default_exe': 'droute',
        'repository': 'https://github.com/DarriEy/dRoute.git',
        'branch': 'main',
        'install_dir': 'droute',
        'build_commands': [
            r'''
# Build dRoute with AD support
set -e

# ── Windows toolchain fixes ──
CMAKE_EXTRA_ARGS=""
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        # Fix 1: TMP/TEMP must point to a writable Windows directory.
        # MSYS2 defaults /tmp which the compiler cannot use.
        _WINTMP="$(cmd.exe //c echo %TEMP% 2>/dev/null | tr -d '\r')"
        if [ -n "$_WINTMP" ] && [ -d "$_WINTMP" ]; then
            export TMP="$_WINTMP"
            export TEMP="$_WINTMP"
            export TMPDIR="$_WINTMP"
            echo "Set TMP=$TMP (Windows temp directory)"
        fi

        # Fix 2: Use MSYS Makefiles generator (default NMake fails)
        CMAKE_EXTRA_ARGS="-G \"MSYS Makefiles\""

        # Fix 3: cmake needs explicit ar/ranlib paths on conda-forge GCC
        _AR="$(find "$CONDA_PREFIX/Library" -name ar.exe 2>/dev/null | head -1)"
        _RANLIB="$(find "$CONDA_PREFIX/Library" -name ranlib.exe 2>/dev/null | head -1)"
        [ -n "$_AR" ] && CMAKE_EXTRA_ARGS="$CMAKE_EXTRA_ARGS -DCMAKE_AR=$_AR"
        [ -n "$_RANLIB" ] && CMAKE_EXTRA_ARGS="$CMAKE_EXTRA_ARGS -DCMAKE_RANLIB=$_RANLIB"
        ;;
esac

# Check for CMake
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install CMake >= 3.14"
    exit 1
fi

# Check for Python - handle Windows conda where python3 triggers MS Store
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/python.exe" ]; then
    PYTHON_EXE="$CONDA_PREFIX/python.exe"
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
    PYTHON_EXE="${PYTHON_EXE:-python}"
else
    PYTHON_EXE="${PYTHON_EXE:-python3}"
fi
if ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "=== Building dRoute ==="
echo "Python: $PYTHON_EXE"

# Create build directory
rm -rf build
mkdir -p build
cd build

# Configure with CMake
# Use DMC_ prefix options (the actual CMake variable names)
eval cmake .. \
    $CMAKE_EXTRA_ARGS \
    -DCMAKE_BUILD_TYPE=Release \
    -DDMC_ENABLE_AD=ON \
    -DDMC_BUILD_PYTHON=OFF \
    -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
    -DCMAKE_INSTALL_PREFIX="../install"

# Fix 4: conda netCDF cmake config injects MSVC-only /utf-8 flag.
# Strip it from generated flags files so GCC doesn't choke.
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        find . -name "flags.make" -exec sed -i 's| /utf-8||g' {} +
        echo "Stripped MSVC /utf-8 flag from GCC build"
        ;;
esac

# Build using cmake --build (avoids MSYS make environment issues)
echo "Building..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . -j$NCORES || cmake --build .

# Install
echo "Installing..."
cmake --install .

# Verify installation
echo "=== Verifying installation ==="
if [ -f "../install/bin/dmc_route_run" ] || [ -f "../install/bin/dmc_route_run.exe" ]; then
    echo "dRoute executable found"
else
    echo "Warning: CLI executable not found (library-only build)"
fi

echo "=== Build complete ==="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['install/bin/dmc_route_run.exe'],
            'check_type': 'exists_any'
        },
        'order': 4,
        'optional': True,  # Not installed by default with --install
        'notes': '''
dRoute build options:
- DROUTE_BUILD_PYTHON: Enable Python bindings (requires pybind11)
- DROUTE_ENABLE_AD: Enable automatic differentiation
- DROUTE_AD_BACKEND: AD backend (codipack or enzyme)

If CMake configuration fails:
1. Ensure pybind11 is installed: pip install pybind11
2. Check C++ compiler: gcc --version or clang --version
3. For AD support, ensure CoDiPack headers are available

Alternative: Install pre-built wheel if available:
    pip install droute
        '''
    }
