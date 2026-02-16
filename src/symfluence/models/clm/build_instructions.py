"""
CLM (CTSM 5.x) build instructions for SYMFLUENCE.

This module defines how to build CLM5 from source via CIME, including:
- Repository and branch information (CTSM 5.3)
- Build commands (CIME case create/setup/build)
- Installation verification criteria

CTSM uses CIME for building. A single-point case is created with
I2000Clm50SpRs compset and f09_g17 resolution (build grid only).
ESMF must be installed (mpiuni serial mode) for the NUOPC driver.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import (
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('clm')
def get_clm_build_instructions():
    """
    Get CLM (CTSM 5.x) build instructions.

    CLM requires NetCDF-Fortran, ESMF, CMake, Python3, and Perl.
    The build uses CIME to create a single-point case and compile
    the cesm.exe executable for standalone CLM runs.

    Returns:
        Dictionary with complete build configuration for CLM.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'Community Land Model (CTSM 5.x)',
        'config_path_key': 'CLM_INSTALL_PATH',
        'config_exe_key': 'CLM_EXE',
        'default_path_suffix': 'installs/clm/bin',
        'default_exe': 'cesm.exe',
        'repository': 'https://github.com/ESCOMP/CTSM.git',
        'branch': 'ctsm5.3.012',
        'install_dir': 'clm',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# CLM (CTSM 5.x) Build Script for SYMFLUENCE
# Builds CTSM with single-point I2000Clm50SpGs compset

echo "=== CLM/CTSM Build Starting ==="
echo "Building CTSM 5.3 with CIME for single-point CLM5"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Check required tools
for tool in cmake perl; do
    if ! command -v $tool >/dev/null 2>&1; then
        echo "ERROR: $tool not found. Please install it."
        exit 1
    fi
done

# Python 3.7+ is required by CTSM git-fleximod and CIME scripts.
# On HPC systems the default python3 may be too old (e.g. RHEL 8 ships 3.6).
# Prefer the venv/conda python which is typically 3.7+.
PYTHON3=""
for py_candidate in python3 python; do
    if command -v "$py_candidate" >/dev/null 2>&1; then
        PY_VER=$("$py_candidate" -c "import sys; print(sys.version_info[:2] >= (3,7))" 2>/dev/null || echo "False")
        if [ "$PY_VER" = "True" ]; then
            PYTHON3="$(command -v "$py_candidate")"
            break
        fi
    fi
done
if [ -z "$PYTHON3" ]; then
    echo "ERROR: Python 3.7 or later is required but not found."
    echo "  On HPC, try: module load python or activate a conda/venv with Python 3.7+"
    exit 1
fi
echo "Using Python: $PYTHON3 ($($PYTHON3 --version 2>&1))"
# Ensure CTSM scripts use the correct python
export PATH="$(dirname "$PYTHON3"):$PATH"

# Check for Fortran compiler
if command -v gfortran >/dev/null 2>&1; then
    export FC=gfortran
    echo "Found Fortran compiler: gfortran"
elif command -v ifort >/dev/null 2>&1; then
    export FC=ifort
    echo "Found Fortran compiler: ifort"
else
    echo "ERROR: No Fortran compiler found (need gfortran or ifort)"
    exit 1
fi

# Ensure we have NetCDF-Fortran
if ! command -v nf-config >/dev/null 2>&1; then
    echo "WARNING: nf-config not found. Trying nc-config for Fortran support..."
    if [ -z "$NETCDF_C" ]; then
        echo "ERROR: NetCDF not found. Please install netcdf-fortran."
        exit 1
    fi
fi

# Navigate to CTSM source
# The clone target IS the install dir, so we may already be in it
if [ -d "manage_externals" ] && [ -d "cime_config" ]; then
    echo "Already in CTSM source directory"
elif [ -d "ctsm" ]; then
    cd ctsm
elif [ -d "CTSM" ]; then
    cd CTSM
else
    echo "ERROR: CTSM source directory not found"
    exit 1
fi

CTSM_ROOT=$(pwd)
echo "CTSM root: $CTSM_ROOT"

# Checkout external dependencies
# CTSM 5.3+ uses git-fleximod instead of manage_externals
echo "Checking out external components..."
if [ -f "./bin/git-fleximod" ]; then
    echo "Using git-fleximod (CTSM 5.3+)"
    ./bin/git-fleximod update
elif [ -f "./manage_externals/checkout_externals" ]; then
    echo "Using checkout_externals (legacy)"
    ./manage_externals/checkout_externals
else
    echo "ERROR: No external checkout tool found"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "ERROR: External checkout failed"
    exit 1
fi

echo "External checkout complete"

# === STEP 1: Locate ESMF FIRST (needed for all subsequent patches) ===
# Use python3 to resolve home dir since $HOME may not be set in build subprocess
_HOME=$($PYTHON3 -c "import pathlib; print(pathlib.Path.home())")
echo "Resolved HOME via python3: $_HOME"
export HOME="${_HOME}"

ESMF_MK_SEARCH=(
    "${_HOME}/.local/esmf/lib/libO/Darwin.gfortranclang.64.mpiuni.default/esmf.mk"
    "${_HOME}/.local/esmf/lib/esmf.mk"
    "/opt/homebrew/opt/esmf/lib/esmf.mk"
    "/usr/local/lib/esmf.mk"
)
for mk in "${ESMF_MK_SEARCH[@]}"; do
    if [ -f "$mk" ]; then
        export ESMFMKFILE="$mk"
        echo "Found ESMFMKFILE: $ESMFMKFILE"
        break
    fi
done
if [ -z "$ESMFMKFILE" ]; then
    echo "ERROR: ESMF not found. Install ESMF first (mpiuni mode)."
    echo "  git clone --depth 1 --branch v8.6.1 https://github.com/esmf-org/esmf.git"
    echo "  cd esmf && export ESMF_COMM=mpiuni ESMF_COMPILER=gfortranclang"
    echo "  make -j\$(nproc) && make install ESMF_INSTALL_PREFIX=~/.local/esmf"
    exit 1
fi

# === STEP 2: Patch PIO for NetCDF 4.9+ compatibility (_FillValue undefined) ===
PIO_NC="${CTSM_ROOT}/libraries/parallelio/src/clib/pio_nc.c"
if [ -f "$PIO_NC" ] && ! grep -q "ifndef _FillValue" "$PIO_NC"; then
    echo "Patching PIO for NetCDF 4.9+ _FillValue compatibility..."
    sed -i.bak '/#include <pio_internal.h>/a\
/* Fix for NetCDF 4.9+ where _FillValue may not be exposed */\
#ifndef _FillValue\
#define _FillValue "_FillValue"\
#endif' "$PIO_NC"
    echo "PIO patched successfully"
fi

# === STEP 3: Inject ESMFMKFILE into homebrew machine config ===
# ESMFMKFILE is now set, so $ESMFMKFILE expands correctly
MACH_XML="${CTSM_ROOT}/ccs_config/machines/homebrew/config_machines.xml"
if [ -f "$MACH_XML" ] && ! grep -q "ESMFMKFILE" "$MACH_XML"; then
    echo "Adding ESMFMKFILE=$ESMFMKFILE to homebrew machine config..."
    $PYTHON3 -c "
with open('$MACH_XML') as f: c = f.read()
c = c.replace('</environment_variables>',
    '      <env name=\"ESMFMKFILE\">$ESMFMKFILE</env>\n    </environment_variables>')
with open('$MACH_XML','w') as f: f.write(c)
"
    echo "ESMFMKFILE added to machine config"
fi

# === STEP 4: Fix gnu_homebrew.cmake: $(NETCDF) is unset ===
GNU_CMAKE="${CTSM_ROOT}/ccs_config/machines/cmake_macros/gnu_homebrew.cmake"
if [ -f "$GNU_CMAKE" ] && grep -q '$(NETCDF)/lib' "$GNU_CMAKE"; then
    echo "Fixing NETCDF rpath in gnu_homebrew.cmake..."
    NETCDF_PREFIX=$(nc-config --prefix 2>/dev/null || echo "/opt/homebrew")
    cat > "$GNU_CMAKE" << CMAKEEOF
execute_process(COMMAND nc-config --prefix OUTPUT_VARIABLE NETCDF_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
string(APPEND LDFLAGS " -framework Accelerate -Wl,-rpath,\${NETCDF_PREFIX}/lib")
CMAKEEOF
    echo "Fixed NETCDF rpath"
fi

# === STEP 5: Create single-point case ===
CASE_DIR="${CTSM_ROOT}/cases/symfluence_build"
COMPSET="I2000Clm50SpRs"
RES="f09_g17"

echo "Creating case: $CASE_DIR"
echo "Compset: $COMPSET (build grid — runtime uses CLM_USRDAT)"
echo "Resolution: $RES"

rm -rf "$CASE_DIR"

if [ "$UNAME_S" = "Darwin" ]; then
    MACH_NAME="homebrew"
else
    MACH_NAME="homebrew"
fi
echo "Using CIME machine: $MACH_NAME"

# Create new case with explicit machine
${CTSM_ROOT}/cime/scripts/create_newcase \
    --case "$CASE_DIR" \
    --compset "$COMPSET" \
    --res "$RES" \
    --run-unsupported \
    --machine "$MACH_NAME" \
    --compiler gnu

if [ $? -ne 0 ]; then
    echo "ERROR: create_newcase failed"
    echo "Available compsets:"
    $PYTHON3 ${CTSM_ROOT}/cime/scripts/query_config --compsets clm 2>/dev/null | head -20
    exit 1
fi

cd "$CASE_DIR"

# Inject ESMFMKFILE into case env_mach_specific.xml so CIME finds it
if ! grep -q "ESMFMKFILE" env_mach_specific.xml 2>/dev/null; then
    $PYTHON3 -c "
with open('env_mach_specific.xml') as f: c = f.read()
c = c.replace('</environment_variables>',
    '  <env name=\"ESMFMKFILE\">$ESMFMKFILE</env>\n    </environment_variables>')
with open('env_mach_specific.xml','w') as f: f.write(c)
"
    echo "Injected ESMFMKFILE=$ESMFMKFILE into env_mach_specific.xml"
fi

# Configure case (standard grid for build, CLM_USRDAT at runtime)
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=1
./xmlchange RUN_STARTDATE=2000-01-01

# Keep build/run directories inside the case to avoid stale scratch conflicts
./xmlchange CIME_OUTPUT_ROOT="${CASE_DIR}/output"
./xmlchange EXEROOT="${CASE_DIR}/bld"
./xmlchange RUNDIR="${CASE_DIR}/run"

# Clean any stale build directory
rm -rf "${CASE_DIR}/bld" "${CASE_DIR}/output" "${CASE_DIR}/run"

# Setup
echo "Running case.setup..."
./case.setup
if [ $? -ne 0 ]; then
    echo "ERROR: case.setup failed"
    exit 1
fi

# Build (CIME case.build uses GMAKE_J for parallelism, not --parallel)
echo "Running case.build..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
./xmlchange GMAKE_J=$NCORES 2>/dev/null || true
./case.build

if [ $? -ne 0 ]; then
    echo "ERROR: case.build failed"
    echo "Check build logs in: ${CASE_DIR}/bld/"
    exit 1
fi

echo "Build successful!"

# Find and install cesm.exe
CESM_EXE=$(find "$CASE_DIR" -name "cesm.exe" -type f 2>/dev/null | head -1)
if [ -z "$CESM_EXE" ]; then
    # Also check bld directory
    CESM_EXE=$(find "${CASE_DIR}/bld" -name "cesm.exe" -o -name "cesm*.exe" -type f 2>/dev/null | head -1)
fi

if [ -z "$CESM_EXE" ]; then
    echo "ERROR: cesm.exe not found after build"
    find "$CASE_DIR" -name "*.exe" -type f 2>/dev/null
    exit 1
fi

# Install to bin directory (install dir = CTSM_ROOT when cloned directly)
INSTALL_DIR="${CTSM_ROOT}"
mkdir -p "${INSTALL_DIR}/bin" "${INSTALL_DIR}/share"

cp "$CESM_EXE" "${INSTALL_DIR}/bin/cesm.exe"
chmod +x "${INSTALL_DIR}/bin/cesm.exe"

# Copy default parameter file
CLM_PARAMS=$(find "$CTSM_ROOT" -name "clm5_params*.nc" -path "*/paramdata/*" 2>/dev/null | head -1)
if [ -n "$CLM_PARAMS" ]; then
    cp "$CLM_PARAMS" "${INSTALL_DIR}/share/clm5_params.nc"
    echo "Copied default parameter file to share/clm5_params.nc"
else
    echo "WARNING: Default clm5_params.nc not found in source tree"
fi

echo "=== CLM Build Complete ==="
echo "Installed to: ${INSTALL_DIR}/bin/cesm.exe"

# Verify installation
if [ -f "${INSTALL_DIR}/bin/cesm.exe" ]; then
    echo "Verification: cesm.exe exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['nc-config', 'gfortran', 'cmake', 'perl'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/cesm.exe'],
            'check_type': 'exists'
        },
        'order': 20  # After VIC
    }
