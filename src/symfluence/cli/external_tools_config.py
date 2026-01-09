#!/usr/bin/env python3

"""
SYMFLUENCE External Tools Configuration

This module defines external tool configurations required by SYMFLUENCE,
including repositories, build instructions, and validation criteria.
"""

from typing import Dict, Any


def get_common_build_environment() -> str:
    """Get common build environment setup used across multiple tools."""
    return r'''
set -e
# Compiler: force absolute path if possible to satisfy CMake/Makefile
if [ -n "$FC" ] && [ -x "$FC" ]; then
    export FC="$FC"
elif command -v gfortran >/dev/null 2>&1; then
    export FC="$(command -v gfortran)"
else
    export FC="${FC:-gfortran}"
fi
export FC_EXE="$FC"

# Discover libraries (fallback to /usr)
export NETCDF="${NETCDF:-$(nc-config --prefix 2>/dev/null || echo /usr)}"
export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$(nf-config --prefix 2>/dev/null || echo /usr)}"
export HDF5_ROOT="${HDF5_ROOT:-$(h5cc -showconfig 2>/dev/null | awk -F': ' "/Installation point/{print $2}" || echo /usr)}"
# Threads
export NCORES="${NCORES:-4}"
    '''.strip()

def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """Define all external tools required by SYMFLUENCE."""
    common_env = get_common_build_environment()
    
    return {
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
set -e
SUNDIALS_VER=7.4.0
SUNDIALS_ROOT_DIR="$(pwd)"
SUNDIALS_PREFIX="${SUNDIALS_ROOT_DIR}/install/sundials"
mkdir -p "${SUNDIALS_PREFIX}"
rm -f "v${SUNDIALS_VER}.tar.gz" || true
wget -q "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz" || curl -fsSL -o "v${SUNDIALS_VER}.tar.gz" "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz"
tar -xzf "v${SUNDIALS_VER}.tar.gz"
cd "sundials-${SUNDIALS_VER}"
rm -rf build && mkdir build && cd build
cmake .. -DBUILD_FORTRAN_MODULE_INTERFACE=ON -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_INSTALL_PREFIX="${SUNDIALS_PREFIX}" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DEXAMPLES_ENABLE=OFF -DBUILD_TESTING=OFF
cmake --build . --target install -j ${NCORES:-4}
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['lib64/libsundials_core.a', 'lib/libsundials_core.a', 'include/sundials/sundials_config.h'],
                'check_type': 'exists_any'
            },
            'order': 1
        },
        'summa': {
            'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
            'config_path_key': 'SUMMA_INSTALL_PATH',
            'config_exe_key': 'SUMMA_EXE',
            'default_path_suffix': 'installs/summa/bin',
            'default_exe': 'summa_sundials.exe',
            'repository': 'https://github.com/CH-Earth/summa.git',
            'branch': 'develop_sundials',
            'install_dir': 'summa',
            'requires': ['sundials'],
            'build_commands': [
                common_env,
                r'''
set -e
export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
SPECIFY_LINKS=OFF
if [ "$(uname)" == "Darwin" ]; then SPECIFY_LINKS=ON; export LIBRARY_LINKS='-llapack'; elif command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then SPECIFY_LINKS=OFF; elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then SPECIFY_LINKS=OFF; else SPECIFY_LINKS=ON; export LIBRARY_LINKS='-llapack -lblas'; fi
rm -rf cmake_build && mkdir -p cmake_build
cmake -S build -B cmake_build -DUSE_SUNDIALS=ON -DCMAKE_BUILD_TYPE=Release -DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS -DCMAKE_PREFIX_PATH="$SUNDIALS_DIR" -DSUNDIALS_ROOT="$SUNDIALS_DIR" -DNETCDF_PATH="${NETCDF:-/usr}" -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"
cmake --build cmake_build --target all -j ${NCORES:-4}
if [ -f "bin/summa_sundials.exe" ]; then cd bin; ln -sf summa_sundials.exe summa.exe; cd ..; elif [ -f "cmake_build/bin/summa_sundials.exe" ]; then mkdir -p bin; cp cmake_build/bin/summa_sundials.exe bin/; cd bin; ln -sf summa_sundials.exe summa.exe; cd ..; elif [ -f "cmake_build/bin/summa.exe" ]; then mkdir -p bin; cp cmake_build/bin/summa.exe bin/; fi
                '''.strip()
            ],
            'dependencies': [],
            'test_command': '--version',
            'verify_install': {
                'file_paths': ['bin/summa.exe', 'bin/summa_sundials.exe'],
                'check_type': 'exists_any'
            },
            'order': 2
        },
        'mesh': {
            'description': 'MESH 1.5.6 - Environment Canada Hydrology Land-Surface Scheme',
            'config_path_key': 'MESH_INSTALL_PATH',
            'config_exe_key': 'MESH_EXE',
            'default_path_suffix': 'installs/mesh/bin',
            'default_exe': 'mesh.exe',
            'repository': 'https://github.com/MESH-Model/MESH-Dev.git',
            'branch': 'SA_MESH_1.5/SA_MESH_1.5.6',
            'install_dir': 'mesh',
            'build_commands': [
                common_env,
                r'''
set -e
echo "Building MESH 1.5.6 (Serial) with NetCDF4 support..."
mkdir -p bin

# Setup NetCDF paths
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_FORTRAN="$(nf-config --prefix)"
elif [ -n "${NETCDF_FORTRAN}" ]; then
    :
else
    NETCDF_FORTRAN="/usr"
fi

if [ -n "${NETCDF_FORTRAN}" ]; then
    export NCDF_PATH="${NETCDF_FORTRAN}"
    export NETCDF_INC="${NETCDF_FORTRAN}/include"
    if [ -d "${NETCDF_FORTRAN}/lib64" ]; then
        export NETCDF_LIB="${NETCDF_FORTRAN}/lib64"
    else
        export NETCDF_LIB="${NETCDF_FORTRAN}/lib"
    fi
fi

# Find NetCDF C library (mandatory for linking sa_mesh)
if command -v nc-config >/dev/null 2>&1; then
    NCDF_C_LIB="-L$(nc-config --prefix)/lib"
elif [ -d "/opt/homebrew/opt/netcdf/lib" ]; then
    NCDF_C_LIB="-L/opt/homebrew/opt/netcdf/lib"
else
    NCDF_C_LIB=""
fi

# Patch getenvc.c for modern C compatibility (MESH 1.5 still uses librmn)
GETENVC_FILE="./Modules/librmn/19.7.0/primitives/getenvc.c"
if [ -f "$GETENVC_FILE" ]; then
    cat > "$GETENVC_FILE" << 'PATCHEOF'
#include <stdlib.h>
#include <string.h>
#define F2Cl int
#if defined(__APPLE__) || defined(__linux__)
#define f77name(x) x##_
#else
#define f77name(x) x
#endif
void f77name(getenvc) ( char *name, char *value, F2Cl len1, F2Cl len2 )
{
   char *temp, *hold; int size, i;
   size = len1+len2+1 ;
   temp = (char *) malloc(size) ;
   hold = (char *) malloc(size) ;
   for ( i=0 ; i < len1 && name[i] != ' ' ; i++ ) *(temp+i) = name[i] ;
   *(temp+i) = '\0' ;
   if (getenv(temp) != NULL) { strcpy(hold, getenv(temp)) ; size = strlen(hold) ; }
   else { size = 0 ; }
   for ( i=0 ; i < len2 ; i++ ) value[i] = ' ' ;
   if ( size != 0 ) { for ( i=0 ; i < size ; i++ ) value[i] = *(hold+i) ; }
   free (temp) ; free (hold) ;
}
PATCHEOF
fi

# Patch Makefile to fix NetCDF linking if needed
if [ -f "Makefile" ] && [ -n "$NCDF_C_LIB" ]; then
    sed -i.bak "s|^LIBNCL\s*=.*|LIBNCL = $NCDF_C_LIB \$(shell nf-config --flibs)|" Makefile 2>/dev/null || true
fi

make veryclean 2>/dev/null || make clean 2>/dev/null || true
make gfortran netcdf 2>&1 | tee build.log
if [ -f "sa_mesh" ]; then
    cp sa_mesh bin/mesh.exe
    chmod +x bin/mesh.exe
    echo "✅ MESH 1.5.6 serial binary built successfully"
else
    echo "❌ MESH compilation failed"
    exit 1
fi
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/mesh.exe'],
                'check_type': 'exists'
            },
            'order': 12
        },
        'mizuroute': {
            'description': 'Mizukami routing model for river network routing',
            'config_path_key': 'INSTALL_PATH_MIZUROUTE',
            'config_exe_key': 'EXE_NAME_MIZUROUTE',
            'default_path_suffix': 'installs/mizuRoute/route/bin',
            'default_exe': 'mizuRoute.exe',
            'repository': 'https://github.com/ESCOMP/mizuRoute.git',
            'branch': 'serial',
            'install_dir': 'mizuRoute',
            'build_commands': [
                common_env,
                r'''
cd route/build
mkdir -p ../bin
F_MASTER_PATH="$(cd .. && pwd)"
if command -v nf-config >/dev/null 2>&1; then NETCDF_TO_USE="$(nf-config --prefix)"; else NETCDF_TO_USE="/usr"; fi
perl -i -pe "s|^FC\s*=.*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=.*$|FC_EXE = ${FC:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=.*$|EXE = mizuRoute.exe|" Makefile
perl -i -pe "s|^F_MASTER\s*=.*$|F_MASTER = $F_MASTER_PATH/|" Makefile
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$| NCDF_PATH = ${NETCDF_TO_USE}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = no|" Makefile
make clean || true
make 2>&1 | tee build.log
if [ -f "../bin/mizuRoute.exe" ]; then echo "Build successful"; else echo "ERROR: Executable not found"; exit 1; fi
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['route/bin/mizuRoute.exe'],
                'check_type': 'exists'
            },
            'order': 3
        },
        'troute': {
            'description': "NOAA's Next Generation river routing model",
            'config_path_key': 'TROUTE_INSTALL_PATH',
            'config_exe_key': 'TROUTE_PKG_PATH',
            'default_path_suffix': 'installs/t-route/src/troute-network',
            'default_exe': 'troute/network/__init__.py',
            'repository': 'https://github.com/NOAA-OWP/t-route.git',
            'branch': 'master',
            'install_dir': 't-route',
            'build_commands': [
                r'''
set +e
cd src/troute-network 2>/dev/null || exit 0
PYTHON_BIN="${SYMFLUENCE_PYTHON:-python3}"
"$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps || true
exit 0
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['src/troute-network/troute/network/__init__.py'],
                'check_type': 'exists'
            },
            'order': 4
        },
        'fuse': {
            'description': "Framework for Understanding Structural Errors",
            'config_path_key': "FUSE_INSTALL_PATH",
            'config_exe_key': "FUSE_EXE",
            'default_path_suffix': "installs/fuse/bin",
            'default_exe': "fuse.exe",
            'repository': "https://github.com/CH-Earth/fuse.git",
            'branch': None,
            'install_dir': "fuse",
            'build_commands': [
                common_env,
                r'''
cd build
make clean 2>/dev/null || true
export F_MASTER="$(cd .. && pwd)/"
NCDF_LIB_DIR="${NETCDF_FORTRAN}/lib"
HDF_LIB_DIR="${HDF5_ROOT}/lib"
LIBS="-L${HDF_LIB_DIR} -lhdf5 -lhdf5_hl -L${NCDF_LIB_DIR} -lnetcdff -lnetcdf"
INCLUDES="-I${HDF5_ROOT}/include -I${NETCDF_FORTRAN}/include"
FFLAGS_NORMA="-O3 -ffree-line-length-none -fallow-argument-mismatch -std=legacy -cpp"
FFLAGS_FIXED="-O2 -c -ffixed-form -fallow-argument-mismatch -std=legacy"
${FC} ${FFLAGS_FIXED} -o sce_16plus.o "FUSE_SRC/FUSE_SCE/sce_16plus.f"
make FC="${FC}" F_MASTER="${F_MASTER}" LIBS="${LIBS}" INCLUDES="${INCLUDES}" FFLAGS_NORMA="${FFLAGS_NORMA}" FFLAGS_FIXED="${FFLAGS_FIXED}"
mkdir -p ../bin && cp fuse.exe ../bin/
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/fuse.exe'],
                'check_type': 'exists'
            },
            'order': 5
        },
        'taudem': {
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'config_path_key': 'TAUDEM_INSTALL_PATH',
            'config_exe_key': 'TAUDEM_EXE',
            'default_path_suffix': 'installs/TauDEM/bin',
            'default_exe': 'pitremove',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'branch': None,
            'install_dir': 'TauDEM',
            'build_commands': [
                common_env,
                r'''
set -e
export CC=mpicc; export CXX=mpicxx
rm -rf build && mkdir -p build && cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 2
mkdir -p ../bin
find . -type f -perm -111 -name "pitremove" -exec cp {} ../bin/ \;
find . -type f -perm -111 -name "streamnet" -exec cp {} ../bin/ \;
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['bin/pitremove'],
                'check_type': 'exists'
            },
            'order': 6
        },
        'gistool': {
            'description': 'Geospatial data extraction and processing tool',
            'config_path_key': 'INSTALL_PATH_GISTOOL',
            'config_exe_key': 'EXE_NAME_GISTOOL',
            'default_path_suffix': 'installs/gistool',
            'default_exe': 'extract-gis.sh',
            'repository': 'https://github.com/kasra-keshavarz/gistool.git',
            'branch': None,
            'install_dir': 'gistool',
            'build_commands': ['chmod +x extract-gis.sh'],
            'verify_install': {'file_paths': ['extract-gis.sh'], 'check_type': 'exists'},
            'dependencies': [], 'test_command': None, 'order': 7
        },
        'datatool': {
            'description': 'Meteorological data processing tool',
            'config_path_key': 'DATATOOL_PATH',
            'config_exe_key': 'DATATOOL_SCRIPT',
            'default_path_suffix': 'installs/datatool',
            'default_exe': 'extract-dataset.sh',
            'repository': 'https://github.com/kasra-keshavarz/datatool.git',
            'branch': None,
            'install_dir': 'datatool',
            'build_commands': ['chmod +x extract-dataset.sh'],
            'dependencies': [], 'test_command': '--help',
            'verify_install': {'file_paths': ['extract-dataset.sh'], 'check_type': 'exists'},
            'order': 8
        },
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
git submodule update --init --recursive
rm -rf cmake_build
cmake -DCMAKE_BUILD_TYPE=Release -DNGEN_WITH_PYTHON=OFF -DNGEN_WITH_SQLITE3=ON -S . -B cmake_build
cmake --build cmake_build --target ngen -j 4
                '''.strip()
            ],
            'dependencies': [], 'test_command': '--help',
            'verify_install': {'file_paths': ['cmake_build/ngen'], 'check_type': 'exists'},
            'order': 9
        },
        'hype': {
            'description': 'HYPE - Hydrological Predictions for the Environment',
            'config_path_key': 'HYPE_INSTALL_PATH',
            'config_exe_key': 'HYPE_EXE',
            'default_path_suffix': 'installs/hype/bin',
            'default_exe': 'hype',
            'repository': 'git://git.code.sf.net/p/hype/code',
            'branch': None,
            'install_dir': 'hype',
            'build_commands': [
                common_env,
                r'''
mkdir -p bin
make hype FC="${FC:-gfortran}"
mv hype bin/
                '''.strip()
            ],
            'dependencies': [], 'test_command': None,
            'verify_install': {'file_paths': ['bin/hype'], 'check_type': 'exists'},
            'order': 11
        },
        'wmfire': {
            'description': 'WMFire - Wildfire spread module for RHESSys',
            'config_path_key': 'WMFIRE_INSTALL_PATH',
            'config_exe_key': 'WMFIRE_LIB',
            'default_path_suffix': 'installs/wmfire/lib',
            'default_exe': 'libwmfire.so',
            'repository': None,  # Built from RHESSys repo FIRE directory
            'branch': None,
            'install_dir': 'wmfire',
            'build_commands': [
                common_env,
                r'''
set -e
echo "Building WMFire (fire spread library for RHESSys)..."

# Detect OS for library extension
if [ "$(uname)" = "Darwin" ]; then
    LIB_EXT="dylib"
    SHARED_FLAG="-dynamiclib"
else
    LIB_EXT="so"
    SHARED_FLAG="-shared"
fi

mkdir -p lib include

# Check for Boost
BOOST_INC=""
if [ -n "$BOOST_ROOT" ]; then
    BOOST_INC="-I$BOOST_ROOT"
elif [ -d "/usr/include/boost" ]; then
    BOOST_INC="-I/usr/include"
elif [ -d "/usr/local/include/boost" ]; then
    BOOST_INC="-I/usr/local/include"
elif [ -d "/opt/homebrew/include/boost" ]; then
    BOOST_INC="-I/opt/homebrew/include"
fi

if [ -z "$BOOST_INC" ]; then
    echo "WARNING: Boost headers not found. Attempting build anyway..."
    BOOST_INC=""
fi

# Clone RHESSys repo if needed for FIRE source
if [ ! -d "RHESSys_src" ]; then
    git clone --depth 1 https://github.com/RHESSys/RHESSys.git RHESSys_src
fi

cd RHESSys_src/FIRE

# Compile WMFire library (requires C++11 for modern Boost)
CXX="${CXX:-g++}"
# -Wno-c++11-narrowing: WMFire has int-to-double narrowing that's harmless
# -Wno-deprecated-declarations: suppress Boost deprecated warnings
CXXFLAGS="-std=c++11 -fPIC -O2 -Wno-c++11-narrowing -Wno-deprecated-declarations $BOOST_INC"
$CXX $CXXFLAGS -c RanNums.cpp -o RanNums.o
$CXX $CXXFLAGS -c WMFire.cpp -o WMFire.o
$CXX $SHARED_FLAG -o libwmfire.$LIB_EXT RanNums.o WMFire.o

# Install
cp libwmfire.$LIB_EXT ../../lib/
cp WMFire.h RanNums.h ../../include/

echo "✅ WMFire library built: lib/libwmfire.$LIB_EXT"
                '''.strip()
            ],
            'dependencies': [],
            'test_command': None,
            'verify_install': {
                'file_paths': ['lib/libwmfire.so', 'lib/libwmfire.dylib'],
                'check_type': 'exists_any'
            },
            'order': 13
        },
        'rhessys': {
            'description': 'RHESSys - Regional Hydro-Ecologic Simulation System',
            'config_path_key': 'RHESSYS_INSTALL_PATH',
            'config_exe_key': 'RHESSYS_EXE',
            'default_path_suffix': 'installs/rhessys/bin',
            'default_exe': 'rhessys',
            'repository': 'https://github.com/RHESSys/RHESSys.git',
            'branch': 'develop',
            'install_dir': 'rhessys',
            'requires': ['wmfire'],  # WMFire must be built first for fire support
            'build_commands': [
                common_env,
                r'''
set -e
echo "Building RHESSys with NetCDF and optional WMFire support..."

# Detect OS for library extension and paths
if [ "$(uname)" = "Darwin" ]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi

mkdir -p bin lib

# Check for WMFire library (from wmfire tool build)
WMFIRE_LIB="../wmfire/lib/libwmfire.$LIB_EXT"
WMFIRE_FLAG=""
if [ -f "$WMFIRE_LIB" ]; then
    echo "Found WMFire library, enabling fire spread support..."
    cp "$WMFIRE_LIB" lib/
    WMFIRE_FLAG="wmfire=T"
    # Set library path for linking
    export LD_LIBRARY_PATH="$(pwd)/lib:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="$(pwd)/lib:${DYLD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$(pwd)/lib:${LIBRARY_PATH:-}"
else
    echo "WMFire not found, building without fire spread support..."
fi

cd rhessys

# Clean any previous build
make clean 2>/dev/null || true

# Build with NetCDF and optionally WMFire
if [ -n "$WMFIRE_FLAG" ]; then
    make V=1 netcdf=T $WMFIRE_FLAG CC="${CC:-gcc}" 2>&1 | tee build.log || {
        echo "Build with WMFire failed, retrying without..."
        make clean 2>/dev/null || true
        make V=1 netcdf=T CC="${CC:-gcc}" 2>&1 | tee build.log
    }
else
    make V=1 netcdf=T CC="${CC:-gcc}" 2>&1 | tee build.log
fi

# Find and copy the built executable
RHESSYS_BIN=$(find . -maxdepth 1 -name "rhessys*" -type f -perm -111 | head -1)
if [ -n "$RHESSYS_BIN" ]; then
    cp "$RHESSYS_BIN" ../bin/rhessys
    chmod +x ../bin/rhessys
    # Copy WMFire library to bin dir for runtime (macOS needs it in same dir or DYLD_LIBRARY_PATH)
    if [ -f "../lib/libwmfire.$LIB_EXT" ]; then
        cp "../lib/libwmfire.$LIB_EXT" ../bin/
        echo "✅ WMFire library copied to bin/"
    fi
    echo "✅ RHESSys binary installed: bin/rhessys"
else
    echo "❌ RHESSys compilation failed - no executable found"
    exit 1
fi
                '''.strip()
            ],
            'dependencies': ['gdal-config', 'nc-config'],
            'test_command': '-h',
            'verify_install': {'file_paths': ['bin/rhessys'], 'check_type': 'exists'},
            'order': 14
        },
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
git clone https://github.com/CIROH-UA/NGIAB-CloudInfra.git ../ngiab
cd ../ngiab && chmod +x guide.sh
                '''.strip()
            ],
            'dependencies': [], 'test_command': None,
            'verify_install': {'file_paths': ['guide.sh'], 'check_type': 'exists'},
            'order': 10
        },
    }