#!/usr/bin/env python3
"""
SYMFLUENCE External Tools Configuration - Platform Agnostic & Robust

This module defines external tool configurations required by SYMFLUENCE.
Critical Fix: Combines environment setup and build commands into atomic scripts
to ensure variables and functions persist across execution steps.
"""

from typing import Dict, Any

def get_common_header() -> str:
    """
    Returns the bash header used for every tool build.
    Contains platform detection, compiler setup, and helper functions.
    """
    return r'''
set -e

# ================================================================
# 1. HELPER FUNCTIONS
# ================================================================

detect_platform() {
    PLATFORM="unknown"
    OS_NAME="$(uname -s)"
    ARCH="$(uname -m)"
    
    if [ "$OS_NAME" = "Darwin" ]; then
        PLATFORM="macos"
    elif [ -f /etc/os-release ]; then
        . /etc/os-release
        PLATFORM="${ID:-linux}"
    fi
    echo "  📍 Platform: $PLATFORM ($ARCH)"
}

git_clone_safe() {
    local repo=$1
    local dir=$2
    local branch=$3
    
    if [ -d "$dir" ] && [ "$(ls -A $dir)" ]; then
        echo "   ⏭️  Directory exists and is not empty: $dir"
        return 0
    fi
    
    echo "   📥 Cloning $repo..."
    
    # Try specified branch first
    if [ -n "$branch" ]; then
        if git clone --depth 1 -b "$branch" "$repo" "$dir" 2>/dev/null; then
            echo "   ✓ Cloned branch: $branch"
            return 0
        else
            echo "   ⚠️  Branch '$branch' not found, trying default branch..."
        fi
    fi
    
    # Fallback to default
    if git clone --depth 1 "$repo" "$dir"; then
        echo "   ✓ Cloned default branch"
        return 0
    else
        echo "   ❌ Clone failed"
        return 1
    fi
}

# ================================================================
# 2. COMPILER CONFIGURATION
# ================================================================

setup_compilers() {
    detect_platform
    
    # --- MAC-SPECIFIC CONFIGURATION ---
    if [ "$PLATFORM" = "macos" ]; then
        # On macOS, use Apple Clang for C/C++ explicitly to avoid CMake confusion
        export CC="/usr/bin/clang"
        export CXX="/usr/bin/clang++"
        
        # Find Homebrew Gfortran
        if command -v brew >/dev/null 2>&1; then
            HB_PREFIX="$(brew --prefix)"
            
            # Find gfortran
            if [ -z "$FC" ]; then
                # Look for specific versions first (homebrew usually installs gfortran-13 or similar)
                for ver in 14 13 12 11 10 9 ""; do
                    if command -v "gfortran-$ver" >/dev/null 2>&1; then
                        export FC="gfortran-$ver"
                        break
                    elif [ -f "$HB_PREFIX/bin/gfortran-$ver" ]; then
                        export FC="$HB_PREFIX/bin/gfortran-$ver"
                        break
                    fi
                done
                # Fallback
                [ -z "$FC" ] && export FC="gfortran"
            fi
            
            # Add Homebrew paths to flags
            export CFLAGS="-I${HB_PREFIX}/include ${CFLAGS}"
            export CPPFLAGS="-I${HB_PREFIX}/include ${CPPFLAGS}"
            export LDFLAGS="-L${HB_PREFIX}/lib ${LDFLAGS}"
            export PKG_CONFIG_PATH="${HB_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
            
            # Link gfortran library if using clang + gfortran
            if [ -n "$FC" ] && command -v $FC >/dev/null 2>&1; then
                GFORT_LIB_PATH=$($FC -print-file-name=libgfortran.dylib)
                if [ -f "$GFORT_LIB_PATH" ]; then
                    GFORT_DIR=$(dirname "$GFORT_LIB_PATH")
                    export LDFLAGS="${LDFLAGS} -L${GFORT_DIR} -Wl,-rpath,${GFORT_DIR}"
                    # Also need to link libgcc_s sometimes on M1/M2
                    GCC_LIB_PATH=$($FC -print-file-name=libgcc_s.1.1.dylib)
                     if [ -f "$GCC_LIB_PATH" ]; then
                        GCC_DIR=$(dirname "$GCC_LIB_PATH")
                        export LDFLAGS="${LDFLAGS} -L${GCC_DIR}"
                    fi
                fi
            fi
        fi
    else
        # Linux/CI defaults
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        export FC="${FC:-gfortran}"
    fi

    # --- MPI HANDLING ---
    # Prioritize MPI wrappers if they exist
    if command -v mpicc >/dev/null 2>&1; then
        export USE_MPI="ON"
        export MPICC="$(which mpicc)"
        export MPICXX="$(which mpicxx || which mpic++)"
        export MPIFC="$(which mpif90 || which mpifort)"
        
        # On Linux CI, sometimes we want to force using the wrappers as the primary compilers
        if [ "$PLATFORM" != "macos" ]; then
             export CC="$MPICC"
             export CXX="$MPICXX"
             export FC="$MPIFC"
        fi
    else
        export USE_MPI="OFF"
    fi

    export FC_EXE="$FC"
    export NCORES="${NCORES:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

    echo "  🔧 Configured: CC=$CC | CXX=$CXX | FC=$FC | MPI=$USE_MPI"
}

# ================================================================
# 3. LIBRARY DISCOVERY
# ================================================================

setup_libraries() {
    # NetCDF
    if command -v nc-config >/dev/null 2>&1; then
        export NETCDF_ROOT="$(nc-config --prefix)"
        export NETCDF_INC="$(nc-config --includedir)"
        export NETCDF_LIB="$(nc-config --libdir)"
    elif pkg-config --exists netcdf; then
        export NETCDF_ROOT="$(pkg-config --variable=prefix netcdf)"
        export NETCDF_INC="$(pkg-config --cflags-only-I | sed 's/-I//')"
        export NETCDF_LIB="$(pkg-config --libs-only-L | sed 's/-L//')"
    else
        # Fallbacks
        for p in /usr/local /usr /opt/homebrew /opt/local; do
            if [ -f "$p/include/netcdf.h" ]; then
                export NETCDF_ROOT="$p"
                export NETCDF_INC="$p/include"
                export NETCDF_LIB="$p/lib"
                break
            fi
        done
    fi
    
    # NetCDF Fortran
    if command -v nf-config >/dev/null 2>&1; then
        export NETCDFF_ROOT="$(nf-config --prefix)"
        export NETCDFF_INC="$(nf-config --includedir)"
        export NETCDFF_LIB="$(nf-config --libdir)"
    else
        export NETCDFF_ROOT="${NETCDF_ROOT}"
        export NETCDFF_INC="${NETCDF_INC}"
        export NETCDFF_LIB="${NETCDF_LIB}"
    fi

    echo "  📚 Libraries: NetCDF=$NETCDF_ROOT"
}

# Initialize environment immediately
setup_compilers
setup_libraries

# Python
if [ -z "$SYMFLUENCE_PYTHON" ]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        export SYMFLUENCE_PYTHON="$VIRTUAL_ENV/bin/python"
    else
        export SYMFLUENCE_PYTHON="python3"
    fi
fi
'''


def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Define all external tools required by SYMFLUENCE.
    """
    header = get_common_header()
    
    return {
        # ================================================================
        # SUNDIALS
        # ================================================================
        'sundials': {
            'description': 'SUNDIALS Solver (Essential for SUMMA)',
            'config_path_key': 'SUNDIALS_INSTALL_PATH',
            'install_dir': 'sundials',
            'repository': None, 
            'default_exe': 'lib/libsundials_core.a',
            'default_path_suffix': 'installs/sundials/install/sundials/',
            # KEY FIX: Single string combining header + script
            'build_commands': [header + r'''
SUNDIALS_VER=7.1.1
INSTALL_DIR="$(pwd)/install/sundials"

# Check if already installed (Library can be in lib or lib64)
if [ -f "$INSTALL_DIR/lib/libsundials_core.a" ] || [ -f "$INSTALL_DIR/lib64/libsundials_core.a" ]; then
    echo "✅ SUNDIALS already installed"
    exit 0
fi

echo "📦 Downloading SUNDIALS v${SUNDIALS_VER}..."
rm -rf sundials-${SUNDIALS_VER} build sundials.tar.gz

wget -qO sundials.tar.gz https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz || \
curl -L -o sundials.tar.gz https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz

tar -xzf sundials.tar.gz
mkdir build && cd build

echo "🔨 Configuring CMake..."
# Construct CMake command carefully
CMD="cmake ../sundials-${SUNDIALS_VER} \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_STATIC_LIBS=ON \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_TESTING=OFF \
-DEXAMPLES_ENABLE=OFF \
-DBUILD_FORTRAN_MODULE_INTERFACE=ON \
-DCMAKE_C_COMPILER=${CC} \
-DCMAKE_CXX_COMPILER=${CXX} \
-DCMAKE_Fortran_COMPILER=${FC}"

if [ "$USE_MPI" = "ON" ]; then
    CMD="$CMD -DENABLE_MPI=ON -DMPI_C_COMPILER=${MPICC} -DMPI_Fortran_COMPILER=${MPIFC}"
else
    CMD="$CMD -DENABLE_MPI=OFF"
fi

echo "   Command: $CMD"
$CMD || { echo "❌ CMake failed"; exit 1; }

echo "🔨 Building..."
make -j${NCORES} install || { echo "❌ Build failed"; exit 1; }
'''],
            'dependencies': ['cmake'],
            'verify_install': {'file_paths': ['install/sundials/lib/libsundials_core.a', 'install/sundials/lib64/libsundials_core.a'], 'check_type': 'exists_any'},
            'order': 1
        },

        # ================================================================
        # SUMMA
        # ================================================================
        'summa': {
            'description': 'SUMMA Hydrological Model',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'install_dir': 'summa',
            'repository': 'https://github.com/CH-Earth/summa.git',
            'branch': 'master',
            'requires': ['sundials'],
            'default_exe': 'summa_sundials.exe',
            'default_path_suffix': 'installs/summa/bin/',
            'build_commands': [header + r'''
# Call git clone safely (Function now exists in scope)
git_clone_safe "https://github.com/CH-Earth/summa.git" "." "develop"

# Find SUNDIALS
SUNDIALS_DIR=""
for path in ../sundials/install/sundials ../../sundials/install/sundials; do
    if [ -d "$path" ]; then SUNDIALS_DIR=$(cd $path && pwd); break; fi
done

if [ -z "$SUNDIALS_DIR" ]; then
    echo "❌ SUNDIALS not found. Please install sundials first."
    exit 1
fi

echo "🔨 Building SUMMA using SUNDIALS at $SUNDIALS_DIR"

rm -rf build_cmake && mkdir build_cmake && cd build_cmake

cmake -S .. -B . \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_Fortran_COMPILER=${FC} \
 -DCMAKE_C_COMPILER=${CC} \
 -DUSE_SUNDIALS=ON \
 -DSUNDIALS_ROOT=${SUNDIALS_DIR} \
 -DNETCDF_ROOT=${NETCDF_ROOT} \
 -DNETCDF_FORTRAN_ROOT=${NETCDFF_ROOT} \
 -DBUILD_TESTING=OFF

make -j${NCORES}

# Install
mkdir -p ../bin
find . -name "summa_sundials.exe" -exec cp {} ../bin/ \;
# If summa.exe exists but not sundials version (older builds), cp it
if [ ! -f ../bin/summa_sundials.exe ]; then
    find . -name "summa.exe" -exec cp {} ../bin/summa_sundials.exe \;
fi

if [ -f "../bin/summa_sundials.exe" ]; then
    echo "✅ SUMMA built successfully"
else
    echo "❌ SUMMA executable not found after build"
    exit 1
fi
'''],
            'dependencies': ['netcdf', 'netcdf-fortran', 'cmake'],
            'verify_install': {'file_paths': ['bin/summa_sundials.exe'], 'check_type': 'exists'},
            'order': 2
        },

        # ================================================================
        # mizuRoute
        # ================================================================
        'mizuroute': {
            'description': 'mizuRoute River Routing',
            'config_path_key': 'MIZUROUTE_INSTALL_PATH',
            'install_dir': 'mizuRoute',
            'repository': 'https://github.com/ESCOMP/mizuRoute.git',
            'branch': 'main',
            'default_exe': 'mizuRoute.exe',
            'default_path_suffix': 'installs/mizuRoute/route/bin/',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/ESCOMP/mizuRoute.git" "." "main"

cd route/build

# Create config
cat > Makefile.config <<EOF
FC = ${FC}
FC_EXE = ${FC}
FLAGS_OPT = -O3 -ffree-line-length-none -fbacktrace
INCLUDES = -I${NETCDF_INC} -I${NETCDFF_INC}
LIBRARIES = -L${NETCDF_LIB} -L${NETCDFF_LIB}
LIBS = -lnetcdff -lnetcdf
EOF

echo "🔨 Building mizuRoute with FC=${FC}..."
make clean 2>/dev/null || true
make -j${NCORES}

mkdir -p ../bin
if [ -f "mizuRoute.exe" ]; then
    cp mizuRoute.exe ../bin/
    echo "✅ mizuRoute built"
else
    echo "❌ mizuRoute build failed"
    exit 1
fi
'''],
            'dependencies': ['netcdf', 'netcdf-fortran'],
            'verify_install': {'file_paths': ['route/bin/mizuRoute.exe'], 'check_type': 'exists'},
            'order': 3
        },

        # ================================================================
        # T-route
        # ================================================================
        'troute': {
            'description': 'NOAA T-route',
            'config_path_key': 'TROUTE_INSTALL_PATH',
            'install_dir': 't-route',
            'repository': 'https://github.com/NOAA-OWP/t-route.git',
            'branch': 'master',
            'default_exe': 'troute/network/__init__.py',
            'default_path_suffix': 'installs/t-route/src/troute-network/',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/NOAA-OWP/t-route.git" "." "master"

echo "🔨 Installing T-route dependencies..."
cd src/troute-network

# Install deps but ignore errors (like pyarrow on HPC)
$SYMFLUENCE_PYTHON -m pip install . || echo "⚠️  Pip install returned error, checking if critical files exist..."

# Manual check
if [ -f "troute/network/__init__.py" ]; then
    echo "✅ T-route files present"
else
    # Try inplace build as fallback
    $SYMFLUENCE_PYTHON setup.py build_ext --inplace || true
fi
'''],
            'dependencies': [],
            'verify_install': {'file_paths': ['src/troute-network/troute/network/__init__.py'], 'check_type': 'exists'},
            'order': 4
        },

        # ================================================================
        # FUSE
        # ================================================================
        'fuse': {
            'description': 'FUSE Model',
            'config_path_key': 'FUSE_INSTALL_PATH',
            'install_dir': 'fuse',
            'repository': 'https://github.com/CH-Earth/fuse.git',
            'branch': None, 
            'default_exe': 'fuse.exe',
            'default_path_suffix': 'installs/fuse/bin/',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/CH-Earth/fuse.git" "." "main"

cd build
cat > Makefile.config <<EOF
FC = ${FC}
FC_EXE = ${FC}
FLAGS_OPT = -O3 -ffree-line-length-none
INCLUDES = -I${NETCDF_INC} -I${NETCDFF_INC}
LIBRARIES = -L${NETCDF_LIB} -L${NETCDFF_LIB}
LIBS = -lnetcdff -lnetcdf
EOF

make clean 2>/dev/null || true
make -j${NCORES}

mkdir -p ../bin
cp fuse.exe ../bin/ 2>/dev/null || true
'''],
            # Ensure dependencies list is present to prevent KeyError
            'dependencies': ['netcdf', 'netcdf-fortran'],
            'verify_install': {'file_paths': ['bin/fuse.exe'], 'check_type': 'exists'},
            'order': 5
        },

        # ================================================================
        # TauDEM
        # ================================================================
        'taudem': {
            'description': 'TauDEM',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'install_dir': 'TauDEM',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': None,
            'default_exe': 'pitremove',
            'default_path_suffix': 'installs/TauDEM/bin/',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/dtarb/TauDEM.git" "." "develop"

rm -rf build && mkdir build && cd build

if [ "$USE_MPI" = "ON" ]; then
    cmake .. -DCMAKE_C_COMPILER=${MPICC} -DCMAKE_CXX_COMPILER=${MPICXX} -DCMAKE_INSTALL_PREFIX=..
else
    cmake .. -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_INSTALL_PREFIX=..
fi

make -j${NCORES} && make install
'''],
            'dependencies': ['cmake'],
            'verify_install': {'file_paths': ['bin/pitremove'], 'check_type': 'exists'},
            'order': 6
        },

        # ================================================================
        # GIStool
        # ================================================================
        'gistool': {
            'description': 'GIStool',
            'config_path_key': 'INSTALL_PATH_GISTOOL',
            'install_dir': 'gistool',
            'repository': 'https://github.com/kasra-keshavarz/gistool.git',
            'branch': None,
            'default_exe': 'extract-gis.sh',
            'default_path_suffix': 'installs/gistool',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/kasra-keshavarz/gistool.git" "." "" 
chmod +x extract-gis.sh
'''],
            'dependencies': [],
            'verify_install': {'file_paths': ['extract-gis.sh'], 'check_type': 'exists'},
            'order': 7
        },

        # ================================================================
        # Datatool
        # ================================================================
        'datatool': {
            'description': 'Datatool',
            'config_path_key': 'DATATOOL_PATH',
            'install_dir': 'datatool',
            'repository': 'https://github.com/kasra-keshavarz/datatool.git',
            'branch': None,
            'default_exe': 'extract-dataset.sh',
            'default_path_suffix': 'installs/datatool',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/kasra-keshavarz/datatool.git" "." "" 
chmod +x extract-dataset.sh
'''],
            'dependencies': [],
            'verify_install': {'file_paths': ['extract-dataset.sh'], 'check_type': 'exists'},
            'order': 8
        },

        # ================================================================
        # NGEN
        # ================================================================
        'ngen': {
            'description': 'NextGen Framework',
            'config_path_key': 'NGEN_INSTALL_PATH',
            'install_dir': 'ngen',
            'repository': 'https://github.com/CIROH-UA/ngen',
            'branch': 'ngiab',
            'default_exe': 'ngen',
            'default_path_suffix': 'installs/ngen/cmake_build',
            'build_commands': [header + r'''
git_clone_safe "https://github.com/CIROH-UA/ngen" "." "ngiab"

# Fetch Boost manually if needed (HPC often has module boost)
if [ -z "$BOOST_ROOT" ] && [ ! -d "/usr/include/boost" ]; then
    echo "📦 Downloading Boost..."
    wget -qO boost.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 || \
    curl -L -o boost.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2
    
    tar -xjf boost.tar.bz2
    export BOOST_ROOT="$(pwd)/boost_1_79_0"
fi

git submodule update --init --recursive -- test/googletest extern/pybind11

rm -rf cmake_build && mkdir cmake_build
cmake -S . -B cmake_build \
 -DCMAKE_C_COMPILER=${CC} \
 -DCMAKE_CXX_COMPILER=${CXX} \
 -DNGEN_WITH_PYTHON=OFF \
 -DNGEN_WITH_SQLITE3=ON \
 -DBOOST_ROOT=${BOOST_ROOT}

cmake --build cmake_build --target ngen -j${NCORES}
'''],
            'dependencies': ['cmake'],
            'verify_install': {'file_paths': ['cmake_build/ngen'], 'check_type': 'exists'},
            'order': 9
        },

        # ================================================================
        # NGIAB
        # ================================================================
        'ngiab': {
            'description': 'NextGen In A Box',
            'config_path_key': 'NGIAB_INSTALL_PATH',
            'install_dir': 'ngiab',
            'repository': None,
            'default_exe': 'guide.sh',
            'default_path_suffix': 'installs/ngiab',
            'build_commands': [header + r'''
# Heuristic for HPC vs Cloud
if [ -n "$SLURM_JOB_ID" ] || [ -n "$PBS_JOBID" ] || [ -d "/scratch" ]; then
    REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
else
    REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
fi
cd .. && rm -rf ngiab
git clone "$REPO" ngiab && cd ngiab && chmod +x guide.sh
'''],
            'dependencies': [],
            'verify_install': {'file_paths': ['guide.sh'], 'check_type': 'exists'},
            'order': 10
        }
    }

if __name__ == "__main__":
    tools = get_external_tools_definitions()
    print(f"✅ Loaded {len(tools)} external tool definitions.")