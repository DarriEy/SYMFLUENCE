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
# Build dRoute with Python bindings and AD support

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

# Get Python info for pybind11
PYTHON_INCLUDE=$("$PYTHON_EXE" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_SITE=$("$PYTHON_EXE" -c "import site; print(site.getsitepackages()[0])")

echo "=== Building dRoute ==="
echo "Python: $PYTHON_EXE"
echo "Python include: $PYTHON_INCLUDE"
echo "Python site-packages: $PYTHON_SITE"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
# Enable Python bindings and AD (CoDiPack)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDROUTE_BUILD_PYTHON=ON \
    -DDROUTE_ENABLE_AD=ON \
    -DDROUTE_AD_BACKEND=codipack \
    -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
    -DCMAKE_INSTALL_PREFIX="../install"

# Build
echo "Building..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j$NCORES

# Install
echo "Installing..."
make install

# Install Python package
if [ -f "droute*.so" ] || [ -f "droute*.pyd" ]; then
    echo "Installing Python bindings..."
    cp droute*.so "$PYTHON_SITE/" 2>/dev/null || \
    cp droute*.pyd "$PYTHON_SITE/" 2>/dev/null || \
    echo "Warning: Could not copy Python module"
fi

# Verify installation
echo "=== Verifying installation ==="
if [ -f "../install/bin/droute" ]; then
    echo "dRoute executable: ../install/bin/droute"
else
    echo "Warning: Executable not found"
fi

# Test Python import
"$PYTHON_EXE" -c "import droute; print(f'dRoute version: {droute.__version__}')" 2>/dev/null || \
    echo "Warning: Python bindings not importable (may need manual installation)"

echo "=== Build complete ==="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': 'python -c "import droute; print(droute.__version__)"',
        'verify_install': {
            'python_import': 'droute',
            'check_type': 'python_module'
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
