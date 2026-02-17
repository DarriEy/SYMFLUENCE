"""
WRF-Hydro build instructions for SYMFLUENCE.

This module defines how to build WRF-Hydro from source, including:
- Repository and branch information
- Build commands (compile_offline_NoahMP.sh with NetCDF-Fortran and MPI)
- Installation verification criteria

WRF-Hydro uses compile_offline_NoahMP.sh for building. The build requires
NetCDF-Fortran and MPI libraries. The default build produces the
wrf_hydro.exe executable with NoahMP LSM for distributed hydrological
simulations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('wrfhydro')
def get_wrfhydro_build_instructions():
    """
    Get WRF-Hydro build instructions.

    WRF-Hydro requires NetCDF-Fortran and MPI. The build uses CMake
    and produces the wrf_hydro.exe executable for distributed
    hydrological simulations with NoahMP LSM.

    Returns:
        Dictionary with complete build configuration for WRF-Hydro.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'NCAR WRF-Hydro Coupled Atmosphere-Hydrology Model',
        'config_path_key': 'WRFHYDRO_INSTALL_PATH',
        'config_exe_key': 'WRFHYDRO_EXE',
        'default_path_suffix': 'installs/wrfhydro/bin',
        'default_exe': 'wrf_hydro.exe',
        'repository': 'https://github.com/NCAR/wrf_hydro_nwm_public.git',
        'branch': 'main',
        'install_dir': 'wrfhydro',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# WRF-Hydro Build Script for SYMFLUENCE
# Builds WRF-Hydro with NoahMP LSM using compile_offline_NoahMP.sh

set -e

echo "=== WRF-Hydro Build Starting ==="
echo "Building WRF-Hydro with NoahMP LSM"

# Check for required tools
if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Please install gfortran."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

# Detect MPI Fortran compiler
if command -v mpifort >/dev/null 2>&1; then
    export MPIFORT=$(command -v mpifort)
elif command -v mpif90 >/dev/null 2>&1; then
    export MPIFORT=$(command -v mpif90)
else
    echo "WARNING: No MPI Fortran compiler found, using gfortran"
    export MPIFORT=$(command -v gfortran)
fi
echo "Using MPI Fortran compiler: $MPIFORT"

# Ensure we have NetCDF
if [ -z "$NETCDF_C" ]; then
    echo "ERROR: NetCDF-C not found. Please install netcdf."
    exit 1
fi

# Check for NetCDF-Fortran
NF_CONFIG=$(command -v nf-config 2>/dev/null || true)
if [ -z "$NF_CONFIG" ]; then
    for nf_path in /opt/homebrew /usr/local /usr; do
        if [ -f "$nf_path/lib/libnetcdff.a" ] || [ -f "$nf_path/lib/libnetcdff.dylib" ] || [ -f "$nf_path/lib/libnetcdff.so" ]; then
            export NETCDF_FORTRAN="$nf_path"
            break
        fi
    done
    if [ -z "${NETCDF_FORTRAN:-}" ]; then
        echo "ERROR: NetCDF-Fortran not found. Please install:"
        echo "  macOS: brew install netcdf-fortran"
        echo "  Ubuntu: sudo apt-get install libnetcdff-dev"
        exit 1
    fi
else
    export NETCDF_FORTRAN=$(nf-config --prefix 2>/dev/null || dirname $(dirname $NF_CONFIG))
fi

echo "Using NetCDF from: $NETCDF_C"
echo "Using NetCDF-Fortran from: $NETCDF_FORTRAN"

# Set up NETCDF env vars for WRF-Hydro
export NETCDF=$NETCDF_C
export NETCDF_INC=$NETCDF_C/include
export NETCDF_LIB=$NETCDF_C/lib

# Save top-level install directory as absolute path
INSTALL_TOP=$(pwd)

# Navigate to source directory and save absolute path
cd src
SRC_DIR=$(pwd)
echo "Source directory: $SRC_DIR"

# Create setEnvar.sh with compile options (required by compile_offline_NoahMP.sh)
cat > "$SRC_DIR/setEnvar.sh" << 'SETENV_EOF'
export WRF_HYDRO=1
export HYDRO_D=0
export SPATIAL_SOIL=0
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
export WRF_HYDRO_RAPID=0
export WRF_HYDRO_NUDGING=0
export NCEP_WCOSS=0
export NWM_META=0
SETENV_EOF

echo "Created setEnvar.sh at: $SRC_DIR/setEnvar.sh"

# Configure the build (select gfortran with MPI = option 2)
# The configure script may change CWD, so we restore it afterward
echo "Configuring WRF-Hydro..."
if [ -f "$SRC_DIR/configure" ]; then
    cd "$SRC_DIR"
    echo "2" | ./configure 2>&1 || true
    cd "$SRC_DIR"
fi

# Pre-compile SURFEX modules that are needed by NoahMP but missing from Makefile deps
echo "Pre-compiling SURFEX modules..."
SURFEX_DIR="$SRC_DIR/Land_models/NoahMP/phys/surfex"
if [ -d "$SURFEX_DIR" ]; then
    cd "$SURFEX_DIR"
    # Compile in dependency order with -fallow-argument-mismatch for modern gfortran
    SFFLAGS="-O2 -ffree-form -ffree-line-length-none -fallow-argument-mismatch"
    # Group 1: No dependencies
    for src in modd_csts.F modd_snow_par.F modd_snow_metamo.F modd_surf_atm.F; do
        if [ -f "$src" ]; then
            echo "  Compiling $src..."
            $MPIFORT -c $SFFLAGS "$src" 2>&1
        fi
    done
    # Group 2: Depends on modd_csts
    for src in ini_csts.F mode_thermos.F; do
        if [ -f "$src" ]; then
            echo "  Compiling $src..."
            $MPIFORT -c $SFFLAGS "$src" 2>&1
        fi
    done
    # Group 3: Depends on modd_csts + mode_thermos + modd_snow_par
    for src in mode_snow3l.F mode_surf_coefs.F tridiag_ground_snowcro.F; do
        if [ -f "$src" ]; then
            echo "  Compiling $src..."
            $MPIFORT -c $SFFLAGS "$src" 2>&1 || true
        fi
    done
    # Group 4: Depends on all above
    for src in module_snowcro.F module_snowcro_intercept.F; do
        if [ -f "$src" ]; then
            echo "  Compiling $src..."
            $MPIFORT -c $SFFLAGS "$src" 2>&1 || true
        fi
    done
    echo "  Generated .mod files:"
    find . -name "*.mod" 2>/dev/null || echo "  (none)"
    # Copy .mod and .o files to all include paths used by the build
    cp -f *.mod "$SRC_DIR/Land_models/NoahMP/phys/"  2>/dev/null || true
    cp -f *.o "$SRC_DIR/Land_models/NoahMP/phys/"    2>/dev/null || true
    cp -f *.mod "$SRC_DIR/Land_models/NoahMP/IO_code/" 2>/dev/null || true
    cp -f *.mod "$SRC_DIR/mod/" 2>/dev/null || true
    echo "SURFEX modules pre-compiled"
else
    echo "WARNING: SURFEX directory not found at $SURFEX_DIR"
fi

# Return to source dir for compilation
cd "$SRC_DIR"

# Compile with compile_offline_NoahMP.sh (the standard WRF-Hydro build method)
# WRF-Hydro's SURFEX modules have circular Makefile deps (ini_csts needs modd_csts,
# mode_surf_coefs needs mode_thermos, module_snowcro needs mode_surf_coefs, etc.)
# The phys Makefile compiles surfex files alphabetically, but ini_csts.F needs
# modd_csts.mod which comes later. The compile script runs `make clean` which
# wipes .o/.mod files between passes.
# Fix: Pass 1 generates .mod files, then we patch the compile script to skip
# `make clean` on pass 2 so those .mod files persist.
echo "Compiling NoahMP (two-pass for SURFEX module dependencies)..."
if [ -f "$SRC_DIR/compile_offline_NoahMP.sh" ]; then
    chmod +x "$SRC_DIR/compile_offline_NoahMP.sh"
    cd "$SRC_DIR"

    # Pass 1: Generate .mod files (allow failures from dependency ordering)
    echo "=== Compile Pass 1 (generating .mod files) ==="
    "$SRC_DIR/compile_offline_NoahMP.sh" "$SRC_DIR/setEnvar.sh" 2>&1 || true

    # Check if exe was produced on first pass
    if [ -f "$SRC_DIR/Run/wrf_hydro.exe" ] || [ -f "$SRC_DIR/Run/wrf_hydro_NoahMP.exe" ]; then
        echo "Build succeeded on first pass"
    else
        # Pass 2: Create a patched compile script that skips `make clean`
        # so that .mod files from pass 1 persist for dependency resolution
        echo "=== Compile Pass 2 (with .mod files from pass 1) ==="
        cd "$SRC_DIR"
        python3 -c "
with open('compile_offline_NoahMP.sh') as f:
    content = f.read()
# Remove make clean line to preserve .mod files from pass 1
content = content.replace(
    'make clean; rm -f Run/wrf_hydro_NoahMP ; rm -f Run/*TBL ; rm -f Run/*namelist*',
    '# Skipping make clean to preserve .mod files')
with open('compile_pass2.sh', 'w') as f:
    f.write(content)
"
        chmod +x compile_pass2.sh
        "$SRC_DIR/compile_pass2.sh" "$SRC_DIR/setEnvar.sh" 2>&1
    fi
elif [ -f "$SRC_DIR/Makefile" ]; then
    NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    make -C "$SRC_DIR" -j${NCORES}
fi

# Find the executable (search from src dir)
cd "$SRC_DIR"
WRF_HYDRO_EXE=""
for exe_path in "Run/wrf_hydro_NoahMP" "Run/wrf_hydro.exe" "Run/wrf_hydro_NoahMP.exe" "Run/wrf_hydro" "build/wrf_hydro.exe"; do
    if [ -f "$exe_path" ]; then
        WRF_HYDRO_EXE="$exe_path"
        break
    fi
done

# Fallback: search recursively
if [ -z "$WRF_HYDRO_EXE" ]; then
    WRF_HYDRO_EXE=$(find "$SRC_DIR" -name "wrf_hydro*" -type f ! -name "*.o" ! -name "*.F*" ! -name "*.c" ! -name "*.sh" ! -name "*.bak" 2>/dev/null | head -1)
fi

if [ -z "$WRF_HYDRO_EXE" ]; then
    echo "ERROR: WRF-Hydro executable not found after build"
    ls -la "$SRC_DIR/Run/" 2>/dev/null || true
    exit 1
fi

echo "Build successful! Found: $WRF_HYDRO_EXE"

# Create bin directory and install
mkdir -p "$INSTALL_TOP/bin"
cp "$SRC_DIR/$WRF_HYDRO_EXE" "$INSTALL_TOP/bin/wrf_hydro.exe" 2>/dev/null || cp "$WRF_HYDRO_EXE" "$INSTALL_TOP/bin/wrf_hydro.exe"
chmod +x "$INSTALL_TOP/bin/wrf_hydro.exe"

echo "=== WRF-Hydro Build Complete ==="
echo "Installed to: $INSTALL_TOP/bin/wrf_hydro.exe"

if [ -f "$INSTALL_TOP/bin/wrf_hydro.exe" ]; then
    echo "Verification: wrf_hydro.exe exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'gfortran', 'nc-config'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/wrf_hydro.exe'],
            'check_type': 'exists'
        },
        'order': 21,
        'optional': True,
    }
