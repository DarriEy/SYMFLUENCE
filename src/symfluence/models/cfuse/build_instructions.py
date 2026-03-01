# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

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
CFUSE_ROOT="$(pwd)"

# Check for required dependencies
echo ""
echo "=== Checking Dependencies ==="

# Check for CMake
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: CMake not found. Please install CMake (cmake.org)"
    exit 1
fi
echo "CMake found: $(cmake --version | head -1)"

# Check for Git (needed for Enzyme clone)
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: Git not found. Please install Git"
    exit 1
fi

# Check for Python - prefer environment variable, then venv/conda, then system
if [ -n "$PYTHON_EXECUTABLE" ] && [ -x "$PYTHON_EXECUTABLE" ]; then
    PYTHON_CMD="$PYTHON_EXECUTABLE"
    echo "Using PYTHON_EXECUTABLE: $PYTHON_CMD"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
    echo "Using virtual environment Python: $PYTHON_CMD"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
    echo "Using conda Python: $PYTHON_CMD"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/python.exe" ]; then
    PYTHON_CMD="$CONDA_PREFIX/python.exe"
    echo "Using conda Python (Windows): $PYTHON_CMD"
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
    PYTHON_CMD=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
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

# Helper: determine parallel job count
get_jobs() {
    if [ -n "$NPROC" ]; then
        echo "$NPROC"
    elif command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 4
    fi
}

# =====================================================
# STEP 1: Detect LLVM/Clang
# =====================================================
echo ""
echo "=== Step 1: LLVM/Clang Detection ==="

LLVM_DIR=""
LLVM_VERSION=""
CXX_COMPILER=""
C_COMPILER=""
OS_NAME="$(uname -s)"

if [ "$OS_NAME" = "Darwin" ]; then
    # macOS: Check generic Homebrew LLVM first
    for llvm_base in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
        if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
            CXX_COMPILER="$llvm_base/bin/clang++"
            C_COMPILER="$llvm_base/bin/clang"
            if [ -d "$llvm_base/lib/cmake/llvm" ]; then
                LLVM_DIR="$llvm_base/lib/cmake/llvm"
            fi
            LLVM_VERSION=$("$CXX_COMPILER" --version | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
            echo "Found Homebrew LLVM $LLVM_VERSION at $llvm_base"
            break
        fi
    done

    # Check versioned Homebrew LLVM (llvm@19, llvm@18, etc.)
    if [ -z "$CXX_COMPILER" ]; then
        for ver in 21 20 19 18 17; do
            for prefix in /opt/homebrew/opt /usr/local/opt; do
                llvm_base="$prefix/llvm@$ver"
                if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
                    CXX_COMPILER="$llvm_base/bin/clang++"
                    C_COMPILER="$llvm_base/bin/clang"
                    if [ -d "$llvm_base/lib/cmake/llvm" ]; then
                        LLVM_DIR="$llvm_base/lib/cmake/llvm"
                    fi
                    LLVM_VERSION="$ver"
                    echo "Found Homebrew LLVM@$ver at $llvm_base"
                    break 2
                fi
            done
        done
    fi

    # Auto-install LLVM via Homebrew if not found on macOS
    if [ -z "$CXX_COMPILER" ]; then
        if command -v brew >/dev/null 2>&1; then
            echo "No Homebrew LLVM found. Installing via: brew install llvm"
            brew install llvm 2>&1
            if [ $? -eq 0 ]; then
                for llvm_base in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
                    if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
                        CXX_COMPILER="$llvm_base/bin/clang++"
                        C_COMPILER="$llvm_base/bin/clang"
                        if [ -d "$llvm_base/lib/cmake/llvm" ]; then
                            LLVM_DIR="$llvm_base/lib/cmake/llvm"
                        fi
                        LLVM_VERSION=$("$CXX_COMPILER" --version | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
                        echo "Installed Homebrew LLVM $LLVM_VERSION at $llvm_base"
                        break
                    fi
                done
            else
                echo "WARNING: brew install llvm failed"
            fi
        else
            echo "WARNING: Homebrew not found. Cannot auto-install LLVM on macOS."
        fi
    fi
else
    # Linux: check system LLVM (generic llvm-config first, then versioned)
    if command -v llvm-config >/dev/null 2>&1; then
        LLVM_DIR="$(llvm-config --cmakedir 2>/dev/null)"
        LLVM_VERSION="$(llvm-config --version 2>/dev/null | sed -nE 's/^([0-9]+).*/\1/p')"
        if command -v "clang++-$LLVM_VERSION" >/dev/null 2>&1; then
            CXX_COMPILER="$(which "clang++-$LLVM_VERSION")"
            C_COMPILER="$(which "clang-$LLVM_VERSION")"
        elif command -v clang++ >/dev/null 2>&1; then
            CXX_COMPILER="$(which clang++)"
            C_COMPILER="$(which clang)"
        fi
        if [ -n "$CXX_COMPILER" ]; then
            echo "Found system LLVM $LLVM_VERSION (llvm-config)"
        fi
    fi

    # Check versioned llvm-config if generic not found
    if [ -z "$CXX_COMPILER" ]; then
        for ver in 21 20 19 18 17; do
            if command -v "llvm-config-$ver" >/dev/null 2>&1; then
                LLVM_DIR="$(llvm-config-$ver --cmakedir 2>/dev/null)"
                LLVM_VERSION="$ver"
                if command -v "clang++-$ver" >/dev/null 2>&1; then
                    CXX_COMPILER="$(which "clang++-$ver")"
                    C_COMPILER="$(which "clang-$ver")"
                    echo "Found system LLVM $LLVM_VERSION (llvm-config-$ver)"
                fi
                break
            fi
        done
    fi
fi

# Fall back to system Clang without LLVM cmake dir (no Enzyme build possible)
if [ -z "$CXX_COMPILER" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER="$(which clang++)"
        C_COMPILER="$(which clang)"
        echo "Using system Clang (no LLVM cmake dir available)"
        echo "Note: Cannot build Enzyme without LLVM cmake modules"
    fi
fi

# Fall back to GCC if no Clang
if [ -z "$CXX_COMPILER" ]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX_COMPILER="$(which g++)"
        C_COMPILER="$(which gcc)"
        case "$(uname -s 2>/dev/null)" in
            MSYS*|MINGW*|CYGWIN*)
                [ -f "${CXX_COMPILER}.exe" ] && CXX_COMPILER="${CXX_COMPILER}.exe"
                [ -f "${C_COMPILER}.exe" ] && C_COMPILER="${C_COMPILER}.exe"
                ;;
        esac
        echo "Using GCC: $(g++ --version | head -1)"
        echo "Note: Enzyme AD requires Clang/LLVM. Using numerical gradients."
    else
        echo "ERROR: No C++ compiler found"
        exit 1
    fi
fi

echo "C++ compiler: $CXX_COMPILER"
echo "C compiler:   $C_COMPILER"
[ -n "$LLVM_DIR" ] && echo "LLVM cmake dir: $LLVM_DIR"
[ -n "$LLVM_VERSION" ] && echo "LLVM version:   $LLVM_VERSION"

# =====================================================
# STEP 2: Check for existing Enzyme
# =====================================================
echo ""
echo "=== Step 2: Checking for Existing Enzyme ==="

USE_ENZYME=OFF
ENZYME_LIB=""

# Check sibling enzyme install first (from: symfluence binary install enzyme)
ENZYME_SIBLING="$(dirname "$CFUSE_ROOT")/enzyme"
for enzyme_path in "$ENZYME_SIBLING/lib/ClangEnzyme"*.dylib \
                   "$ENZYME_SIBLING/lib/LLVMEnzyme"*.so \
                   "$ENZYME_SIBLING/lib/ClangEnzyme"*.so \
                   "$ENZYME_SIBLING/enzyme/_build/Enzyme/ClangEnzyme"*.dylib \
                   "$ENZYME_SIBLING/enzyme/_build/Enzyme/LLVMEnzyme"*.so; do
    if [ -f "$enzyme_path" ]; then
        ENZYME_LIB="$enzyme_path"
        echo "Found sibling Enzyme install: $ENZYME_LIB"
        break
    fi
done

# Check local builds (inline cfuse deps or user home)
if [ -z "$ENZYME_LIB" ]; then
    for enzyme_path in "$CFUSE_ROOT/deps/Enzyme/enzyme/_build/Enzyme/ClangEnzyme"*.dylib \
                       "$CFUSE_ROOT/deps/Enzyme/enzyme/_build/Enzyme/LLVMEnzyme"*.so \
                       "$HOME/Enzyme/enzyme/build_release/Enzyme/ClangEnzyme"*.dylib \
                       "$HOME/Enzyme/enzyme/build_release/Enzyme/LLVMEnzyme"*.so \
                       "$HOME/enzyme/build/Enzyme/ClangEnzyme"*.dylib \
                       "$HOME/enzyme/build/Enzyme/LLVMEnzyme"*.so; do
        if [ -f "$enzyme_path" ]; then
            ENZYME_LIB="$enzyme_path"
            echo "Found local Enzyme: $ENZYME_LIB"
            break
        fi
    done
fi

# System paths if not found locally
if [ -z "$ENZYME_LIB" ]; then
    for enzyme_path in "/usr/local/lib/LLVMEnzyme"* \
                       "/usr/local/lib/ClangEnzyme"* \
                       "/opt/homebrew/lib/LLVMEnzyme"* \
                       "/opt/homebrew/lib/ClangEnzyme"* \
                       "$HOME/.local/lib/LLVMEnzyme"* \
                       "$HOME/.local/lib/ClangEnzyme"*; do
        if [ -f "$enzyme_path" ]; then
            ENZYME_LIB="$enzyme_path"
            echo "Found system Enzyme: $ENZYME_LIB"
            break
        fi
    done
fi

# Verify existing Enzyme matches our LLVM version
if [ -n "$ENZYME_LIB" ] && [ -n "$LLVM_VERSION" ]; then
    ENZYME_VER=$(basename "$ENZYME_LIB" | sed -E 's/.*Enzyme-([0-9]+).*/\1/')
    if [ "$ENZYME_VER" = "$LLVM_VERSION" ]; then
        USE_ENZYME=ON
        echo "Enzyme $ENZYME_VER matches LLVM $LLVM_VERSION - will use existing build"
    else
        echo "WARNING: Existing Enzyme $ENZYME_VER does not match LLVM $LLVM_VERSION"
        echo "         Will build matching Enzyme from source"
        ENZYME_LIB=""
    fi
elif [ -n "$ENZYME_LIB" ]; then
    USE_ENZYME=ON
    echo "Using existing Enzyme (no LLVM version to cross-check)"
fi

# =====================================================
# STEP 3: Build Enzyme from source (if needed)
# =====================================================
if [ "$USE_ENZYME" = "OFF" ] && [ -n "$LLVM_DIR" ] && [ -n "$LLVM_VERSION" ]; then
    echo ""
    echo "=== Step 3: Building Enzyme AD from Source ==="

    ENZYME_SRC_DIR="$CFUSE_ROOT/deps/Enzyme"
    # Use _build (not build) to avoid conflict with Bazel BUILD file
    # on case-insensitive filesystems (macOS HFS+/APFS)
    ENZYME_BUILD_DIR="$ENZYME_SRC_DIR/enzyme/_build"

    # Clone Enzyme if not already present
    if [ ! -d "$ENZYME_SRC_DIR/.git" ]; then
        echo "Cloning Enzyme AD repository..."
        mkdir -p "$CFUSE_ROOT/deps"
        if git clone --depth 1 https://github.com/EnzymeAD/Enzyme.git "$ENZYME_SRC_DIR" 2>&1; then
            echo "Enzyme repository cloned successfully"
        else
            echo "WARNING: Failed to clone Enzyme repository"
            echo "Continuing without Enzyme (numerical gradients)"
            ENZYME_SRC_DIR=""
        fi
    else
        echo "Enzyme source directory already exists: $ENZYME_SRC_DIR"
        # Fetch latest
        cd "$ENZYME_SRC_DIR"
        git fetch --depth 1 origin 2>/dev/null
        cd "$CFUSE_ROOT"
    fi

    if [ -n "$ENZYME_SRC_DIR" ] && [ -d "$ENZYME_SRC_DIR" ]; then
        # Select branch based on LLVM version
        cd "$ENZYME_SRC_DIR"
        if [ "$LLVM_VERSION" -ge 20 ] 2>/dev/null; then
            ENZYME_BRANCH="main"
        else
            # Try version-specific branch first
            ENZYME_BRANCH="v$LLVM_VERSION"
            git fetch --depth 1 origin "$ENZYME_BRANCH" 2>/dev/null
            if ! git rev-parse "origin/$ENZYME_BRANCH" >/dev/null 2>&1; then
                echo "Branch $ENZYME_BRANCH not found, using main"
                ENZYME_BRANCH="main"
            fi
        fi
        echo "Using Enzyme branch: $ENZYME_BRANCH"
        git checkout "$ENZYME_BRANCH" 2>/dev/null || \
            git checkout -b "$ENZYME_BRANCH" "origin/$ENZYME_BRANCH" 2>/dev/null || \
            echo "Staying on current branch"
        cd "$CFUSE_ROOT"

        # Configure Enzyme with CMake
        echo "Configuring Enzyme with LLVM $LLVM_VERSION..."
        mkdir -p "$ENZYME_BUILD_DIR"
        cd "$ENZYME_BUILD_DIR"

        cmake .. \
            -DLLVM_DIR="$LLVM_DIR" \
            -DCMAKE_BUILD_TYPE=Release 2>&1

        if [ $? -ne 0 ]; then
            echo "WARNING: Enzyme CMake configuration failed (see errors above)"
            echo "Continuing without Enzyme (numerical gradients)"
            cd "$CFUSE_ROOT"
        else
            # Build Enzyme
            ENZYME_JOBS=$(get_jobs)
            echo "Building Enzyme with $ENZYME_JOBS parallel jobs..."
            make -j"$ENZYME_JOBS" 2>&1

            if [ $? -ne 0 ]; then
                echo "WARNING: Enzyme build failed (see errors above)"
                echo "Continuing without Enzyme (numerical gradients)"
                cd "$CFUSE_ROOT"
            else
                # Find the built Enzyme plugin library
                for built_enzyme in "$ENZYME_BUILD_DIR/Enzyme/ClangEnzyme"*.dylib \
                                    "$ENZYME_BUILD_DIR/Enzyme/LLVMEnzyme"*.so \
                                    "$ENZYME_BUILD_DIR/Enzyme/ClangEnzyme"*.so; do
                    if [ -f "$built_enzyme" ]; then
                        ENZYME_LIB="$built_enzyme"
                        USE_ENZYME=ON
                        echo "Enzyme built successfully: $ENZYME_LIB"
                        break
                    fi
                done

                if [ "$USE_ENZYME" = "OFF" ]; then
                    echo "WARNING: Enzyme build completed but plugin library not found"
                    echo "Continuing without Enzyme (numerical gradients)"
                fi
                cd "$CFUSE_ROOT"
            fi
        fi
    fi
else
    echo ""
    if [ "$USE_ENZYME" = "ON" ]; then
        echo "=== Step 3: Enzyme Build Skipped (using existing Enzyme) ==="
    elif [ -z "$LLVM_DIR" ]; then
        echo "=== Step 3: Enzyme Build Skipped ==="
        echo "No LLVM cmake directory found - cannot build Enzyme from source"
        echo "Will use numerical gradients"
    else
        echo "=== Step 3: Enzyme Build Skipped ==="
        echo "Will use numerical gradients"
    fi
fi

# =====================================================
# STEP 3.5: Detect SUNDIALS, NetCDF, pybind11
# =====================================================
echo ""
echo "=== Step 3.5: Additional Dependencies ==="

# ── SUNDIALS ─────────────────────────────────────────
USE_SUNDIALS=OFF
SUNDIALS_ROOT=""

# Check sibling sundials install (from: symfluence binary install sundials)
SUNDIALS_SIBLING="$(dirname "$CFUSE_ROOT")/sundials/install/sundials"
if [ -d "$SUNDIALS_SIBLING/lib" ] && [ -d "$SUNDIALS_SIBLING/include/sundials" ]; then
    SUNDIALS_ROOT="$SUNDIALS_SIBLING"
    USE_SUNDIALS=ON
    echo "SUNDIALS: found sibling install at $SUNDIALS_ROOT"
fi

# Check Homebrew sundials
if [ "$USE_SUNDIALS" = "OFF" ]; then
    for sundials_base in /opt/homebrew/opt/sundials /usr/local/opt/sundials; do
        if [ -d "$sundials_base/lib" ] && [ -d "$sundials_base/include/sundials" ]; then
            SUNDIALS_ROOT="$sundials_base"
            USE_SUNDIALS=ON
            echo "SUNDIALS: found Homebrew install at $SUNDIALS_ROOT"
            break
        fi
    done
fi

# Check system paths
if [ "$USE_SUNDIALS" = "OFF" ]; then
    for sundials_base in /usr/local /usr; do
        if [ -f "$sundials_base/include/sundials/sundials_types.h" ]; then
            SUNDIALS_ROOT="$sundials_base"
            USE_SUNDIALS=ON
            echo "SUNDIALS: found system install at $SUNDIALS_ROOT"
            break
        fi
    done
fi

if [ "$USE_SUNDIALS" = "OFF" ]; then
    echo "SUNDIALS: not found (ODE solvers disabled)"
    echo "  Install with: symfluence binary install sundials"
fi

# ── NetCDF (C++) ─────────────────────────────────────
USE_NETCDF=OFF

# Check for netcdf-cxx4 headers
for nc_inc in /opt/homebrew/include /opt/homebrew/opt/netcdf-cxx4/include \
              /usr/local/include /usr/include; do
    if [ -f "$nc_inc/ncFile.h" ]; then
        USE_NETCDF=ON
        echo "NetCDF C++: found headers at $nc_inc"
        break
    fi
done

# Fall back to NetCDF C (cFUSE CMake can use it)
if [ "$USE_NETCDF" = "OFF" ]; then
    if command -v nc-config >/dev/null 2>&1; then
        NC_VERSION=$(nc-config --version 2>/dev/null)
        USE_NETCDF=ON
        echo "NetCDF C: found ($NC_VERSION) - C++ bindings may be limited"
    else
        echo "NetCDF: not found (I/O disabled)"
        echo "  Install with: brew install netcdf netcdf-cxx4"
    fi
fi

# ── pybind11 ─────────────────────────────────────────
USE_PYTHON=OFF
PYBIND11_DIR=""

if $PYTHON_CMD -c "import pybind11; print(f'pybind11 {pybind11.__version__}')" 2>/dev/null; then
    PYBIND11_DIR=$($PYTHON_CMD -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    USE_PYTHON=ON
    echo "pybind11: found (cmake dir: $PYBIND11_DIR)"
else
    echo "pybind11: not found - attempting install..."
    if $PYTHON_CMD -m pip install pybind11 2>&1; then
        if $PYTHON_CMD -c "import pybind11; print(f'pybind11 {pybind11.__version__}')" 2>/dev/null; then
            PYBIND11_DIR=$($PYTHON_CMD -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
            USE_PYTHON=ON
            echo "pybind11: installed successfully"
        fi
    fi
    if [ "$USE_PYTHON" = "OFF" ]; then
        echo "pybind11: install failed (Python bindings disabled)"
        echo "  Install with: pip install pybind11"
    fi
fi

# =====================================================
# STEP 4: CMake Configuration for cFUSE
# =====================================================
echo ""
echo "=== Step 4: CMake Configuration ==="

mkdir -p build
cd build

# Configure with CMake (note: cFUSE uses DFUSE_ prefix for CMake options)
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

if [ -n "$CXX_COMPILER" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=$CXX_COMPILER"
fi
if [ -n "$C_COMPILER" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=$C_COMPILER"
fi

# Python bindings
if [ "$USE_PYTHON" = "ON" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_BUILD_PYTHON=ON"
    [ -n "$PYBIND11_DIR" ] && CMAKE_ARGS="$CMAKE_ARGS -Dpybind11_DIR=$PYBIND11_DIR"
else
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_BUILD_PYTHON=OFF"
fi

# Enzyme AD
if [ "$USE_ENZYME" = "ON" ] && [ -n "$ENZYME_LIB" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_USE_ENZYME=ON"
    CMAKE_ARGS="$CMAKE_ARGS -DENZYME_PLUGIN=$ENZYME_LIB"
fi

# SUNDIALS
if [ "$USE_SUNDIALS" = "ON" ] && [ -n "$SUNDIALS_ROOT" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_USE_SUNDIALS=ON"
    CMAKE_ARGS="$CMAKE_ARGS -DSUNDIALS_ROOT=$SUNDIALS_ROOT"

    # Guard: CVODES adjoint support requires cvodes_adjoint.hpp which may not
    # exist yet in all cFUSE versions. Patch CMakeLists.txt to skip CVODES
    # linking when the header is absent, preventing a build failure.
    if [ ! -f "$CFUSE_ROOT/include/cfuse/cvodes_adjoint.hpp" ]; then
        if grep -q "DFUSE_USE_CVODES" "$CFUSE_ROOT/CMakeLists.txt" 2>/dev/null; then
            echo "Note: cvodes_adjoint.hpp not found, disabling CVODES adjoint"
            sed 's/SUNDIALS_CVODES_FOUND AND SUNDIALS_CVODES_TARGET/FALSE/' \
                "$CFUSE_ROOT/CMakeLists.txt" > "$CFUSE_ROOT/CMakeLists.txt.tmp" && \
                mv "$CFUSE_ROOT/CMakeLists.txt.tmp" "$CFUSE_ROOT/CMakeLists.txt"
        fi
    fi
fi

# NetCDF
if [ "$USE_NETCDF" = "ON" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDFUSE_USE_NETCDF=ON"
fi

echo ""
echo "Build configuration:"
echo "  Python:    $([ "$USE_PYTHON" = "ON" ] && echo "ON" || echo "OFF")"
echo "  Enzyme AD: $([ "$USE_ENZYME" = "ON" ] && echo "ON ($ENZYME_LIB)" || echo "OFF")"
echo "  SUNDIALS:  $([ "$USE_SUNDIALS" = "ON" ] && echo "ON ($SUNDIALS_ROOT)" || echo "OFF")"
echo "  NetCDF:    $([ "$USE_NETCDF" = "ON" ] && echo "ON" || echo "OFF")"
echo ""

echo "Running: cmake .. $CMAKE_ARGS"
cmake .. $CMAKE_ARGS 2>&1

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# =====================================================
# STEP 5: Build the C++ library
# =====================================================
echo ""
echo "=== Step 5: Building C++ Library ==="

JOBS=$(get_jobs)
echo "Building with $JOBS parallel jobs"
make -j$JOBS 2>&1

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "C++ library built successfully"

# =====================================================
# STEP 6: Install Python module
# =====================================================
echo ""
echo "=== Step 6: Installing Python Module ==="
cd "$CFUSE_ROOT"

# Check if setup.py or pyproject.toml exists
if [ -f "python/setup.py" ] || [ -f "python/pyproject.toml" ]; then
    echo "Installing Python package from python/ directory"
    cd python
    pip install -e . --no-deps 2>&1
    cd "$CFUSE_ROOT"
elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing Python package from root directory"
    pip install -e . --no-deps 2>&1
else
    echo "No Python package definition found"
    echo "Adding build directory to PYTHONPATH"

    # Create a .pth file for the Python path
    SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
    if [ -d "$SITE_PACKAGES" ]; then
        echo "$CFUSE_ROOT/build" > "$SITE_PACKAGES/cfuse.pth"
        echo "$CFUSE_ROOT/python" >> "$SITE_PACKAGES/cfuse.pth"
        echo "Created cfuse.pth in $SITE_PACKAGES"
    fi
fi

# =====================================================
# STEP 7: Verify installation
# =====================================================
echo ""
echo "=== Step 7: Verifying Installation ==="

# Test import
if $PYTHON_CMD -c "import cfuse; print(f'cFUSE version: {cfuse.__version__}')" 2>/dev/null; then
    echo "cfuse Python module imported successfully"
else
    echo "WARNING: Could not import cfuse module"
    echo "You may need to add the build directory to your PYTHONPATH"
fi

# Check for core module and Enzyme status
if $PYTHON_CMD -c "import cfuse_core; print('cfuse_core module found')" 2>/dev/null; then
    echo "cfuse_core C++ module found"

    # Check Enzyme status
    ENZYME_STATUS=$($PYTHON_CMD -c "import cfuse_core; print(cfuse_core.HAS_ENZYME)" 2>/dev/null)
    if [ -n "$ENZYME_STATUS" ]; then
        echo "Enzyme AD (cfuse_core.HAS_ENZYME): $ENZYME_STATUS"
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
except Exception as e:  # noqa: BLE001 — must-not-raise contract
    print(f'  Basic import test: FAILED ({e})')
    sys.exit(1)
" || echo "Basic functionality test had issues"

echo ""
echo "=== cFUSE Build Complete ==="
echo "Installation path: $CFUSE_ROOT"
if [ "$USE_ENZYME" = "ON" ]; then
    echo "Enzyme AD: ENABLED (native gradients available)"
    echo "Enzyme lib: $ENZYME_LIB"
else
    echo "Enzyme AD: DISABLED (using numerical gradients)"
    echo "To enable: ensure LLVM is installed and re-run build"
fi
            '''.strip()
        ],
        'dependencies': [],  # PyTorch/NumPy checked in build script (not system binaries)
        'test_command': 'python -c "import cfuse; print(cfuse.__version__)"',
        'verify_install': {
            'python_import': 'cfuse',
            'check_type': 'python_module',
            'pre_imports': ['torch'],  # torch must load first on Windows (DLL search order)
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
