#!/usr/bin/env python3

"""
SYMFLUENCE External Tools Configuration

This module provides build configurations for external tools required by SYMFLUENCE.

Architecture:
    - Infrastructure tools (sundials, taudem, gistool, datatool, ngiab) are defined
      directly in this file and registered via BuildInstructionsRegistry.register_instructions()
    - Model-specific tools (summa, fuse, mizuroute, etc.) are defined in their
      respective model directories (e.g., src/symfluence/models/summa/build_instructions.py)
      and registered via @BuildInstructionsRegistry.register() decorator

Public API:
    get_external_tools_definitions() -> Dict[str, Dict[str, Any]]
        Returns all tool definitions (both infrastructure and model-specific).
        This is the primary interface used by BinaryManager.

Tools Defined Here (Infrastructure):
    - SUNDIALS: Differential equation solver library (required by SUMMA)
    - TauDEM: Terrain Analysis Using Digital Elevation Models
    - GIStool: Geospatial data extraction tool
    - Datatool: Meteorological data processing tool
    - NGIAB: NextGen In A Box deployment system

Tools Defined in Model Directories:
    - SUMMA: src/symfluence/models/summa/build_instructions.py
    - FUSE: src/symfluence/models/fuse/build_instructions.py
    - mizuRoute: src/symfluence/models/mizuroute/build_instructions.py
    - t-route: src/symfluence/models/troute/build_instructions.py
    - NGEN: src/symfluence/models/ngen/build_instructions.py
    - HYPE: src/symfluence/models/hype/build_instructions.py
    - MESH: src/symfluence/models/mesh/build_instructions.py
    - RHESSys: src/symfluence/models/rhessys/build_instructions.py
"""

from typing import Any, Dict

from .services.build_registry import BuildInstructionsRegistry
from .services.build_snippets import (
    get_common_build_environment,
)


def _register_infrastructure_tools() -> None:
    """
    Register infrastructure tool build instructions.

    These are tools that do NOT have model directories and are
    kept centralized here. They include solver libraries, geospatial
    utilities, and deployment wrappers.
    """
    common_env = get_common_build_environment()

    # ================================================================
    # SUNDIALS - Solver Library (Install First - Required by SUMMA)
    # ================================================================
    BuildInstructionsRegistry.register_instructions('sundials', {
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
# Build SUNDIALS from release tarball (shared libs OK; SUMMA will link).
set -e

SUNDIALS_VER=7.4.0

# Tool install root, e.g.  .../SYMFLUENCE_data/installs/sundials
SUNDIALS_ROOT_DIR="$(pwd)"

# Actual install prefix, consistent with default_path_suffix and SUMMA:
#   .../installs/sundials/install/sundials
SUNDIALS_PREFIX="${SUNDIALS_ROOT_DIR}/install/sundials"
mkdir -p "${SUNDIALS_PREFIX}"

rm -f "v${SUNDIALS_VER}.tar.gz" || true
wget -q "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz" \
  || curl -fsSL -o "v${SUNDIALS_VER}.tar.gz" "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz"

tar -xzf "v${SUNDIALS_VER}.tar.gz"
cd "sundials-${SUNDIALS_VER}"

rm -rf build && mkdir build && cd build
cmake .. \
  -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_INSTALL_PREFIX="${SUNDIALS_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DEXAMPLES_ENABLE=OFF \
  -DBUILD_TESTING=OFF

cmake --build . --target install -j ${NCORES:-4}

# Debug: show where the libs landed
[ -d "${SUNDIALS_PREFIX}/lib64" ] && ls -la "${SUNDIALS_PREFIX}/lib64" | head -20 || true
[ -d "${SUNDIALS_PREFIX}/lib" ] && ls -la "${SUNDIALS_PREFIX}/lib" | head -20 || true
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': [
                'install/sundials/lib64/libsundials_core.a',
                'install/sundials/lib/libsundials_core.a',
                'install/sundials/include/sundials/sundials_config.h'
            ],
            'check_type': 'exists_any'
        },
        'order': 1
    })

    # ================================================================
    # TauDEM - Terrain Analysis
    # ================================================================
    BuildInstructionsRegistry.register_instructions('taudem', {
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
# Build TauDEM from GitHub repository
set -e

# On Compute Canada HPC, OpenMPI has broken Level Zero dependency through hwloc.
# The Level Zero library doesn't exist but hwloc was built with it enabled.
# Solution: Use --allow-shlib-undefined to ignore missing symbols in shared libs.
CMAKE_MPI_FLAGS=""
if [ -d "/cvmfs/soft.computecanada.ca" ]; then
    echo "Detected Compute Canada HPC environment"

    # Use system gcc/g++ to avoid broken mpicc dependency chain
    export CC=gcc
    export CXX=g++

    # Tell linker to allow undefined symbols in shared libraries
    # This works around the missing libze_loader.so that hwloc wants
    export LDFLAGS="-Wl,--allow-shlib-undefined ${LDFLAGS:-}"

    # Tell cmake where MPI is
    MPI_ROOT=$(dirname $(dirname $(which mpicc 2>/dev/null))) || true
    if [ -n "$MPI_ROOT" ] && [ -d "$MPI_ROOT" ]; then
        echo "Found MPI at: $MPI_ROOT"
        CMAKE_MPI_FLAGS="-DMPI_HOME=$MPI_ROOT -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
    fi
else
    # On other systems, mpicc/mpicxx as CC/CXX works fine
    export CC=mpicc
    export CXX=mpicxx
fi

rm -rf build && mkdir -p build
cd build

# Let CMake find MPI and GDAL
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release $CMAKE_MPI_FLAGS

# Build everything plus the two tools that sometimes get skipped by default
cmake --build . -j 2
cmake --build . --target moveoutletstostreams gagewatershed -j 2 || true

echo "Staging executables..."
mkdir -p ../bin

# Debug: show what was built
echo "Files in build directory:"
find . -type f -name "pitremove" -o -name "streamnet" -o -name "aread8" 2>/dev/null || true

# List of expected TauDEM tools (superset — some may not exist on older commits)
tools="pitremove d8flowdir d8converge dinfconverge dinfflowdir aread8 areadinf threshold
       streamnet slopearea gridnet peukerdouglas lengtharea moveoutletstostreams gagewatershed"

copied=0
for exe in $tools;
  do
  # Find by name (cross-platform: works on macOS, Linux, WSL)
  # Note: piping through head masks find errors, so use sequential checks
  p=""
  # Try GNU find -executable first (Linux)
  if [ -z "$p" ]; then p="$(find . -type f -executable -name "$exe" 2>/dev/null | head -n1 || true)"; fi
  # Try POSIX -perm /111 (macOS/BSD/Linux)
  if [ -z "$p" ]; then p="$(find . -type f -perm /111 -name "$exe" 2>/dev/null | head -n1 || true)"; fi
  # Fallback: find by name only (then chmod +x below handles permissions)
  if [ -z "$p" ]; then p="$(find . -type f -name "$exe" ! -path "*/CMakeFiles/*" 2>/dev/null | head -n1 || true)"; fi
  if [ -n "$p" ] && [ -f "$p" ]; then
    cp -f "$p" ../bin/
    chmod +x ../bin/$exe
    copied=$((copied+1))
    echo "  Copied: $exe"
  fi
done

echo "Copied $copied executables"

# Final sanity
ls -la ../bin/ || true
if [ ! -f "../bin/pitremove" ] || [ ! -f "../bin/streamnet" ]; then
  echo "TauDEM stage failed: core binaries missing" >&2
  echo "Build directory contents:"
  find . -name "pitremove" -o -name "streamnet" 2>/dev/null || true
  exit 1
fi
echo "TauDEM executables staged"
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/pitremove'],
            'check_type': 'exists'
        },
        'order': 6
    })

    # ================================================================
    # GIStool - Geospatial Data Extraction
    # ================================================================
    BuildInstructionsRegistry.register_instructions('gistool', {
        'description': 'Geospatial data extraction and processing tool',
        'config_path_key': 'INSTALL_PATH_GISTOOL',
        'config_exe_key': 'EXE_NAME_GISTOOL',
        'default_path_suffix': 'installs/gistool',
        'default_exe': 'extract-gis.sh',
        'repository': 'https://github.com/kasra-keshavarz/gistool.git',
        'branch': None,
        'install_dir': 'gistool',
        'build_commands': [
            r'''
set -e
chmod +x extract-gis.sh
            '''.strip()
        ],
        'verify_install': {
            'file_paths': ['extract-gis.sh'],
            'check_type': 'exists'
        },
        'dependencies': [],
        'test_command': None,
        'order': 7
    })

    # ================================================================
    # Datatool - Meteorological Data Processing
    # ================================================================
    BuildInstructionsRegistry.register_instructions('datatool', {
        'description': 'Meteorological data extraction and processing tool',
        'config_path_key': 'DATATOOL_PATH',
        'config_exe_key': 'DATATOOL_SCRIPT',
        'default_path_suffix': 'installs/datatool',
        'default_exe': 'extract-dataset.sh',
        'repository': 'https://github.com/kasra-keshavarz/datatool.git',
        'branch': None,
        'install_dir': 'datatool',
        'build_commands': [
            r'''
set -e
chmod +x extract-dataset.sh
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--help',
        'verify_install': {
            'file_paths': ['extract-dataset.sh'],
            'check_type': 'exists'
        },
        'order': 8
    })

    # ================================================================
    # NGIAB - NextGen In A Box
    # ================================================================
    BuildInstructionsRegistry.register_instructions('ngiab', {
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
# Detect HPC vs laptop/workstation and fetch the right NGIAB wrapper repo into ../ngiab
IS_HPC=false
for scheduler in sbatch qsub bsub; do
  if command -v $scheduler >/dev/null 2>&1; then IS_HPC=true; break; fi
done
[ -n "$SLURM_CLUSTER_NAME" ] && IS_HPC=true
[ -n "$PBS_JOBID" ] && IS_HPC=true
[ -n "$SGE_CLUSTER_NAME" ] && IS_HPC=true
[ -d "/scratch" ] && IS_HPC=true

if $IS_HPC; then
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-HPCInfra.git"
  echo "HPC environment detected; using NGIAB-HPCInfra"
else
  NGIAB_REPO="https://github.com/CIROH-UA/NGIAB-CloudInfra.git"
  echo "Non-HPC environment detected; using NGIAB-CloudInfra"
fi

cd ..
rm -rf ngiab
git clone "$NGIAB_REPO" ngiab
cd ngiab
[ -f guide.sh ] && chmod +x guide.sh && bash -n guide.sh || true
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['guide.sh'],
            'check_type': 'exists'
        },
        'order': 10,
        'optional': True,  # Not installed by default with --install
    })


def _import_model_build_instructions() -> None:
    """
    Import model build instructions to trigger registration.

    This is done lazily to avoid importing heavy model dependencies.
    We only import the build_instructions modules, which are lightweight
    (they only depend on build_snippets and build_registry).
    """
    import importlib

    model_modules = [
        'symfluence.models.summa.build_instructions',
        'symfluence.models.fuse.build_instructions',
        'symfluence.models.cfuse.build_instructions',
        'symfluence.models.droute.build_instructions',
        'symfluence.models.mizuroute.build_instructions',
        'symfluence.models.troute.build_instructions',
        'symfluence.models.ngen.build_instructions',
        'symfluence.models.hype.build_instructions',
        'symfluence.models.mesh.build_instructions',
        'symfluence.models.wmfire.build_instructions',
        'symfluence.models.rhessys.build_instructions',
        'symfluence.models.ignacio.build_instructions',
        'symfluence.models.vic.build_instructions',
    ]

    for module_name in model_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            # Model may not be installed or available
            pass


# Register infrastructure tools on module load
_register_infrastructure_tools()


def get_external_tools_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Get all external tool definitions (both infrastructure and model-specific).

    This function maintains backward compatibility with BinaryManager.
    It aggregates:
    1. Infrastructure tools (sundials, taudem, gistool, datatool, ngiab)
    2. Model-specific tools (summa, fuse, mizuroute, etc.)

    Returns:
        Dictionary mapping tool names to their complete configuration including:
        - description: Human-readable description
        - config_path_key: Key in config file for installation path
        - config_exe_key: Key in config file for executable name
        - default_path_suffix: Default relative path for installation
        - default_exe: Default executable/library filename
        - repository: Git repository URL (None for non-git installs)
        - branch: Git branch to checkout (None for default)
        - install_dir: Directory name for installation
        - requires: List of tool dependencies (other tools)
        - build_commands: Shell commands for building
        - dependencies: System dependencies required
        - test_command: Command argument for testing (None to skip)
        - verify_install: Installation verification criteria
        - order: Installation order (lower numbers first)
    """
    # Trigger lazy loading of model build instructions
    _import_model_build_instructions()

    # Return all aggregated instructions
    return BuildInstructionsRegistry.get_all_instructions()


if __name__ == "__main__":
    """Test the configuration definitions."""
    tools = get_external_tools_definitions()
    print(f"Loaded {len(tools)} external tool definitions:")
    for name, info in sorted(tools.items(), key=lambda x: x[1].get('order', 99)):
        print(f"   {info.get('order', '?'):2}. {name:12s} - {info['description'][:60]}")
