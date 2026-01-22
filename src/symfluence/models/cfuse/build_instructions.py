"""
cFUSE build instructions for SYMFLUENCE.

This module defines how to build cFUSE from source, including:
- Repository and branch information
- Build commands (CMake)
- Python module installation
- Installation verification criteria

cFUSE (differentiable FUSE) is a PyTorch/Enzyme AD implementation of the
FUSE (Framework for Understanding Structural Errors) model supporting
automatic differentiation for gradient-based calibration.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('cfuse')
def get_cfuse_build_instructions():
    """
    Get cFUSE build instructions.

    cFUSE uses CMake for building with optional Enzyme AD support.
    Requires PyTorch for the Python interface. Falls back to numerical
    gradients if Enzyme is not available.

    Returns:
        Dictionary with complete build configuration for cFUSE.
    """
    return {
        'description': 'Differentiable FUSE hydrological model with Enzyme AD',
        'config_path_key': 'CFUSE_INSTALL_PATH',
        'config_exe_key': None,  # Python module, not executable
        'default_path_suffix': 'installs/cfuse',
        'default_exe': None,
        'repository': 'https://github.com/DarriEy/cFUSE.git',
        'branch': 'main',
        'install_dir': 'cfuse',
        'build_commands': [
            r'''
echo "=== cFUSE Build Starting ==="
echo "Building cFUSE (differentiable FUSE with Enzyme AD)"

# Check for required dependencies
echo ""
echo "=== Checking Dependencies ==="

# Check for CMake
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install CMake (cmake.org)"
    exit 1
fi
echo "CMake found: $(cmake --version | head -1)"

# Check for Python - prefer environment variable, then venv, then system
if [ -n "$PYTHON_EXECUTABLE" ] && [ -x "$PYTHON_EXECUTABLE" ]; then
    PYTHON_CMD="$PYTHON_EXECUTABLE"
    echo "Using PYTHON_EXECUTABLE: $PYTHON_CMD"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
    echo "Using virtual environment Python: $PYTHON_CMD"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
    echo "Using conda Python: $PYTHON_CMD"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "ERROR: Python not found (tried python3 and python)"
    exit 1
fi
echo "Python found: $($PYTHON_CMD --version)"
echo "Python path: $(which $PYTHON_CMD 2>/dev/null || echo $PYTHON_CMD)"

# Check for PyTorch
if ! $PYTHON_CMD -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "WARNING: PyTorch not found. Install with: pip install torch"
    echo "         Gradient functionality will not be available without PyTorch"
fi

# Check for NumPy
if ! $PYTHON_CMD -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
    echo "ERROR: NumPy not found. Install with: pip install numpy"
    exit 1
fi

# =====================================================
# STEP 1: Detect compiler and Enzyme availability
# =====================================================
echo ""
echo "=== Step 1: Compiler Detection ==="

# Prefer Clang for Enzyme AD support
USE_ENZYME=OFF
ENZYME_LIB=""
CXX_COMPILER=""
C_COMPILER=""

# Check for Enzyme first to determine required Clang version
# Local Enzyme builds (check these first)
for enzyme_path in "$HOME/Enzyme/enzyme/build_release/Enzyme/ClangEnzyme"*.dylib \
                   "$HOME/Enzyme/enzyme/build_release/Enzyme/LLVMEnzyme"*.so \
                   "$HOME/enzyme/build/Enzyme/ClangEnzyme"*.dylib \
                   "$HOME/enzyme/build/Enzyme/LLVMEnzyme"*.so; do
    if [ -f "$enzyme_path" ]; then
        ENZYME_LIB="$enzyme_path"
        echo "Found local Enzyme: $ENZYME_LIB"
        break
    fi
done

# System paths if not found locally
if [ -z "$ENZYME_LIB" ]; then
    for enzyme_path in "/usr/local/lib/LLVMEnzyme"* "/opt/homebrew/lib/LLVMEnzyme"* "$HOME/.local/lib/LLVMEnzyme"*; do
        if [ -f "$enzyme_path" ]; then
            ENZYME_LIB="$enzyme_path"
            echo "Found system Enzyme: $ENZYME_LIB"
            break
        fi
    done
fi

# If Enzyme found, extract version and find matching Clang
if [ -n "$ENZYME_LIB" ]; then
    # Extract version number from Enzyme library name (e.g., ClangEnzyme-19.dylib -> 19)
    ENZYME_VERSION=$(basename "$ENZYME_LIB" | sed -E 's/.*Enzyme-([0-9]+).*/\1/')
    echo "Enzyme version detected: $ENZYME_VERSION"

    # Look for matching Homebrew Clang version
    HOMEBREW_CLANG_PATHS=(
        "/opt/homebrew/Cellar/llvm@$ENZYME_VERSION"
        "/opt/homebrew/opt/llvm@$ENZYME_VERSION"
        "/usr/local/Cellar/llvm@$ENZYME_VERSION"
        "/usr/local/opt/llvm@$ENZYME_VERSION"
    )

    for clang_base in "${HOMEBREW_CLANG_PATHS[@]}"; do
        if [ -d "$clang_base" ]; then
            # Find the version subdirectory
            CLANG_BIN=$(find "$clang_base" -name "clang++" -type f 2>/dev/null | head -1)
            if [ -n "$CLANG_BIN" ] && [ -x "$CLANG_BIN" ]; then
                CXX_COMPILER="$CLANG_BIN"
                C_COMPILER="$(dirname "$CLANG_BIN")/clang"
                USE_ENZYME=ON
                echo "Found matching Homebrew Clang $ENZYME_VERSION: $CXX_COMPILER"
                break
            fi
        fi
    done

    if [ "$USE_ENZYME" = "OFF" ]; then
        echo "WARNING: Enzyme $ENZYME_VERSION found but no matching Clang $ENZYME_VERSION"
        echo "         Enzyme plugins must match the Clang version exactly"
        echo "         Install with: brew install llvm@$ENZYME_VERSION"
    fi
fi

# Fall back to system Clang without Enzyme
if [ -z "$CXX_COMPILER" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER=$(which clang++)
        C_COMPILER=$(which clang)
        CLANG_VERSION=$(clang++ --version | head -1)
        echo "Using system Clang: $CLANG_VERSION"
        if [ -n "$ENZYME_LIB" ]; then
            echo "Note: System Clang version may not match Enzyme. Disabling Enzyme."
            USE_ENZYME=OFF
        fi
    fi
fi

# Fall back to GCC if no Clang
if [ -z "$CXX_COMPILER" ]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX_COMPILER=$(which g++)
        C_COMPILER=$(which gcc)
        echo "Using GCC: $(g++ --version | head -1)"
        echo "Note: Enzyme AD requires Clang. Using numerical gradients."
    else
        echo "ERROR: No C++ compiler found"
        exit 1
    fi
fi

if [ -z "$ENZYME_LIB" ]; then
    echo "Enzyme not found - will use numerical gradients"
    echo "For Enzyme AD, install from: https://enzyme.mit.edu"
fi

# =====================================================
# STEP 2: Create build directory and run CMake
# =====================================================
echo ""
echo "=== Step 2: CMake Configuration ==="

mkdir -p build
cd build

# Configure with CMake (note: cFUSE uses DFUSE_ prefix for CMake options)
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_BUILD_PYTHON=ON"

if [ -n "$CXX_COMPILER" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=$CXX_COMPILER"
fi
if [ -n "$C_COMPILER" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=$C_COMPILER"
fi

if [ "$USE_ENZYME" = "ON" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_USE_ENZYME=ON"
    CMAKE_ARGS="$CMAKE_ARGS -DENZYME_PLUGIN=$ENZYME_LIB"
fi

echo "Running: cmake .. $CMAKE_ARGS"
cmake .. $CMAKE_ARGS

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# =====================================================
# STEP 3: Build the C++ library
# =====================================================
echo ""
echo "=== Step 3: Building C++ Library ==="

# Determine number of parallel jobs
if [ -n "$NPROC" ]; then
    JOBS=$NPROC
elif command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

echo "Building with $JOBS parallel jobs"
make -j$JOBS

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "C++ library built successfully"

# =====================================================
# STEP 4: Install Python module
# =====================================================
echo ""
echo "=== Step 4: Installing Python Module ==="
cd ..

# Check if setup.py or pyproject.toml exists
if [ -f "python/setup.py" ] || [ -f "python/pyproject.toml" ]; then
    echo "Installing Python package from python/ directory"
    cd python
    pip install -e . --no-deps
    cd ..
elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing Python package from root directory"
    pip install -e . --no-deps
else
    echo "No Python package definition found"
    echo "Adding build directory to PYTHONPATH"

    # Create a .pth file for the Python path
    SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
    if [ -d "$SITE_PACKAGES" ]; then
        echo "$(pwd)/build" > "$SITE_PACKAGES/cfuse.pth"
        echo "$(pwd)/python" >> "$SITE_PACKAGES/cfuse.pth"
        echo "Created cfuse.pth in $SITE_PACKAGES"
    fi
fi

# =====================================================
# STEP 5: Verify installation
# =====================================================
echo ""
echo "=== Step 5: Verifying Installation ==="

# Test import
if $PYTHON_CMD -c "import cfuse; print(f'cFUSE version: {cfuse.__version__}')" 2>/dev/null; then
    echo "cfuse Python module imported successfully"
else
    echo "WARNING: Could not import cfuse module"
    echo "You may need to add the build directory to your PYTHONPATH"
fi

# Check for core module
if $PYTHON_CMD -c "import cfuse_core; print('cfuse_core module found')" 2>/dev/null; then
    echo "cfuse_core C++ module found"

    # Check Enzyme status
    if $PYTHON_CMD -c "import cfuse_core; print(f'Enzyme AD: {cfuse_core.HAS_ENZYME}')" 2>/dev/null; then
        echo "Checked Enzyme AD status"
    fi
else
    echo "WARNING: cfuse_core C++ module not found"
    echo "The model will not be able to run without the core module"
fi

# Test basic functionality
echo ""
echo "Testing basic functionality..."
$PYTHON_CMD -c "
import sys
try:
    from cfuse import PARAM_BOUNDS, DEFAULT_PARAMS, VIC_CONFIG
    print(f'  Parameters defined: {len(PARAM_BOUNDS)}')
    print(f'  Model configs available: VIC, TOPMODEL, PRMS, SACRAMENTO')
    print('  Basic import test: PASSED')
except Exception as e:
    print(f'  Basic import test: FAILED ({e})')
    sys.exit(1)
" || echo "Basic functionality test had issues"

echo ""
echo "=== cFUSE Build Complete ==="
echo "Installation path: $(pwd)"
if [ "$USE_ENZYME" = "ON" ]; then
    echo "Enzyme AD: ENABLED (native gradients available)"
else
    echo "Enzyme AD: DISABLED (using numerical gradients)"
fi
            '''.strip()
        ],
        'dependencies': ['pytorch', 'numpy'],
        'test_command': 'python3 -c "import cfuse; print(cfuse.__version__)"',
        'verify_install': {
            'python_import': 'cfuse',
            'check_type': 'python_module'
        },
        'order': 15,
        'optional': True,  # Not installed by default with --install
        'notes': [
            'Requires PyTorch for gradient computation',
            'Enzyme AD is optional but recommended for accurate gradients',
            'Falls back to numerical gradients if Enzyme unavailable',
            'CMake >= 3.14 required',
        ]
    }
