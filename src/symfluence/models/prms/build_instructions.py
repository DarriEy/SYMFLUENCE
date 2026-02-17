"""
PRMS build instructions for SYMFLUENCE.

This module defines how to build PRMS from source, including:
- Repository and branch information
- Build commands (make + gfortran with NetCDF-Fortran)
- Installation verification criteria

PRMS is built from Fortran source code using make + gfortran. The build
produces the prms executable for watershed-scale hydrological simulations.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('prms')
def get_prms_build_instructions():
    """
    Get PRMS build instructions.

    PRMS is compiled from Fortran source using make + gfortran.
    The build produces the prms executable for HRU-based
    watershed simulations.

    Returns:
        Dictionary with complete build configuration for PRMS.
    """
    common_env = get_common_build_environment()

    return {
        'description': 'USGS Precipitation-Runoff Modeling System',
        'config_path_key': 'PRMS_INSTALL_PATH',
        'config_exe_key': 'PRMS_EXE',
        'default_path_suffix': 'installs/prms/bin',
        'default_exe': 'prms',
        'repository': 'https://github.com/nhm-usgs/prms.git',
        'branch': 'master',
        'install_dir': 'prms',
        'build_commands': [
            common_env,
            r'''
# PRMS Build Script for SYMFLUENCE
# Builds PRMS using make + gfortran

set -e

echo "=== PRMS Build Starting ==="
echo "Building PRMS with make + gfortran"

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Please install gfortran."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

echo "gfortran version: $(gfortran --version | head -1)"

# Detect NetCDF-Fortran paths (needed for nhru_ncf.f90 and nsegment_ncf.f90)
NF_INC=""
NF_FLIBS=""
NC_LIBS=""
if command -v nf-config >/dev/null 2>&1; then
    NF_INC=$(nf-config --includedir 2>/dev/null || true)
    NF_FLIBS=$(nf-config --flibs 2>/dev/null || true)
    echo "NetCDF-Fortran include: $NF_INC"
    echo "NetCDF-Fortran libs: $NF_FLIBS"
else
    # Fallback: search common paths
    for nf_path in /opt/homebrew/Cellar/netcdf-fortran/*/include /opt/homebrew/include /usr/local/include /usr/include; do
        if [ -f "$nf_path/netcdf.mod" ] || [ -f "$nf_path/NETCDF.mod" ]; then
            NF_INC="$nf_path"
            break
        fi
    done
    echo "NetCDF-Fortran include (fallback): ${NF_INC:-not found}"
fi

# Also detect NetCDF-C library path (nf-config --flibs may reference -lnetcdf
# without providing the -L path to the C library, which lives separately)
if command -v nc-config >/dev/null 2>&1; then
    NC_LIBS=$(nc-config --libs 2>/dev/null || true)
    echo "NetCDF-C libs: $NC_LIBS"
    # Merge NetCDF-C -L path into NF_FLIBS if not already present
    NC_LIBDIR=$(echo "$NC_LIBS" | grep -o '\-L[^ ]*' || true)
    if [ -n "$NC_LIBDIR" ] && ! echo "$NF_FLIBS" | grep -q -- "$NC_LIBDIR"; then
        NF_FLIBS="$NC_LIBDIR $NF_FLIBS"
        echo "Merged NetCDF-C lib path into linker flags: $NF_FLIBS"
    fi
fi

# PRMS repo clones directly into install dir; top-level has Makefile, mmf/, prms/
echo "Building PRMS with make..."

# Legacy PRMS Fortran code has type mismatches in getparam() calls;
# modern gfortran (>=10) treats these as errors by default.
# Patch makelist to add -fallow-argument-mismatch and configure NetCDF paths.
if [ -f "makelist" ]; then
    echo "Patching makelist for modern gfortran compatibility..."
    export NF_INC_EXPORT="$NF_INC"
    export NF_FLIBS_EXPORT="$NF_FLIBS"
    python3 -c "
import os
p = 'makelist'
with open(p) as f: txt = f.read()

nf_inc = os.environ.get('NF_INC_EXPORT', '')
nf_flibs = os.environ.get('NF_FLIBS_EXPORT', '')

# Add -fallow-argument-mismatch to FFLAGS (if not already present)
if '-fallow-argument-mismatch' not in txt:
    txt = txt.replace('-fbounds-check -Wall', '-fbounds-check -Wall -fallow-argument-mismatch')

# Replace NETCDF_DIR include reference with actual path
if nf_inc:
    txt = txt.replace(r'-I\$(NETCDF_DIR)/include', '-I' + nf_inc)
    # If there's no NETCDF include in FFLAGS, add it to the active FFLAGS line
    if '-I' + nf_inc not in txt and nf_inc:
        txt = txt.replace('-fno-second-underscore', '-fno-second-underscore -I' + nf_inc)
else:
    txt = txt.replace(r'-I\$(NETCDF_DIR)/include', '')

# Fix GCLIB: replace Cray-specific paths with proper NetCDF libs
import re
# Match the active (uncommented) GCLIB line
gclib_pattern = r'^GCLIB\s*=.*$'
if nf_flibs:
    replacement = 'GCLIB\t\t= -lgfortran ' + nf_flibs
else:
    replacement = 'GCLIB\t\t= -lgfortran'
txt = re.sub(gclib_pattern, replacement, txt, flags=re.MULTILINE)

with open(p, 'w') as f: f.write(txt)
print('makelist patched successfully')
"
fi

if [ -f "Makefile" ] || [ -f "makefile" ]; then
    NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    make -j${NCORES} FC=gfortran CC=gcc 2>&1
else
    echo "ERROR: No Makefile found"
    ls -la
    exit 1
fi

# Find the executable (PRMS builds to prms/ subdirectory as prms_hpc or prms)
PRMS_EXE=""
for exe_path in "prms/prms_hpc" "prms/prms" "build/prms" "bin/prms"; do
    if [ -f "$exe_path" ]; then
        PRMS_EXE="$exe_path"
        break
    fi
done

# Also search with find as fallback
if [ -z "$PRMS_EXE" ]; then
    PRMS_EXE=$(find . -name "prms*" -type f -perm +111 2>/dev/null | grep -v '\.o$' | head -1)
fi

if [ -z "$PRMS_EXE" ]; then
    echo "ERROR: PRMS executable not found after build"
    find . -name "prms*" -type f 2>/dev/null || true
    exit 1
fi

echo "Build successful! Found: $PRMS_EXE"

# Create bin directory and install
mkdir -p bin
cp "$PRMS_EXE" bin/prms
chmod +x bin/prms

echo "=== PRMS Build Complete ==="
echo "Installed to: bin/prms"

# Verify installation
if [ -f "bin/prms" ]; then
    echo "Verification: prms exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'gfortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/prms'],
            'check_type': 'exists'
        },
        'order': 22,
        'optional': True,
    }
