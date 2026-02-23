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
    - Enzyme AD: Automatic differentiation via LLVM (used by cFUSE)

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

from .external_tools_build_commands import (
    ENZYME_BUILD_COMMAND,
    OPENFEWS_BUILD_COMMAND,
    SUNDIALS_BUILD_COMMAND,
    TAUDEM_BUILD_COMMAND,
)
from .services.build_registry import BuildInstructionsRegistry
from .services.build_snippets import (
    get_common_build_environment,
)


def _register_sundials(common_env: str) -> None:
    """Register SUNDIALS build instructions."""
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
            SUNDIALS_BUILD_COMMAND,
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
        'order': 1,
        'library_only': True,
    })

def _register_taudem(common_env: str) -> None:
    """Register TauDEM build instructions."""
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
            TAUDEM_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/pitremove', 'bin/pitremove.exe'],
            'check_type': 'exists_any'
        },
        'order': 6
    })

def _register_gistool() -> None:
    """Register GIStool build instructions."""
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

def _register_datatool() -> None:
    """Register Datatool build instructions."""
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

def _register_openfews() -> None:
    """Register openFEWS build instructions."""
    # ================================================================
    # openFEWS - Delft-FEWS Flood Early Warning System
    # ================================================================
    BuildInstructionsRegistry.register_instructions('openfews', {
        'description': 'openFEWS - Delft Flood Early Warning System (open-source distribution)',
        'config_path_key': 'OPENFEWS_INSTALL_PATH',
        'config_exe_key': 'OPENFEWS_EXE',
        'default_path_suffix': 'installs/openfews',
        'default_exe': 'bin/fews.sh',
        'repository': None,
        'branch': None,
        'install_dir': 'openfews',
        'build_commands': [
            OPENFEWS_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/fews.sh', 'Modules/symfluence_adapter.xml'],
            'check_type': 'exists'
        },
        'order': 11,
        'optional': True,
    })

def _register_ngiab() -> None:
    """Register NGIAB build instructions."""
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
    })


def _register_enzyme() -> None:
    """Register Enzyme AD build instructions."""
    # ================================================================
    # Enzyme AD - Automatic Differentiation (used by cFUSE)
    # ================================================================
    BuildInstructionsRegistry.register_instructions('enzyme', {
        'description': 'Enzyme AD - Automatic Differentiation via LLVM',
        'config_path_key': None,
        'config_exe_key': None,
        'default_path_suffix': 'installs/enzyme',
        'default_exe': None,
        'repository': 'https://github.com/EnzymeAD/Enzyme.git',
        'branch': 'main',
        'install_dir': 'enzyme',
        'build_commands': [
            ENZYME_BUILD_COMMAND,
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': [
                'lib/ClangEnzyme.dylib',
                'lib/LLVMEnzyme.so',
            ],
            'check_type': 'exists_any'
        },
        'order': 14,  # Before cfuse (order=15)
        'optional': True,
        'library_only': True,
    })


def _register_infrastructure_tools() -> None:
    """Register infrastructure tool build instructions."""
    common_env = get_common_build_environment()
    _register_sundials(common_env)
    _register_taudem(common_env)
    _register_gistool()
    _register_datatool()
    _register_openfews()
    _register_ngiab()
    _register_enzyme()


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
        'symfluence.models.clm.build_instructions',
        'symfluence.models.swat.build_instructions',
        'symfluence.models.mhm.build_instructions',
        'symfluence.models.crhm.build_instructions',
        'symfluence.models.prms.build_instructions',
        'symfluence.models.modflow.build_instructions',
        'symfluence.models.gsflow.build_instructions',
        'symfluence.models.watflood.build_instructions',
        'symfluence.models.wflow.build_instructions',
        'symfluence.models.parflow.build_instructions',
        'symfluence.models.clmparflow.build_instructions',
        'symfluence.models.wrfhydro.build_instructions',

        'symfluence.models.pihm.build_instructions',
    ]

    for module_name in model_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            # Model may not be installed or available
            pass
        except Exception as exc:  # noqa: BLE001 — intentional; catch all failures
            # Catch non-ImportError failures (e.g. a dependency in the
            # model's __init__.py raising AttributeError or TypeError).
            # Log so the user can diagnose missing tools.
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load build instructions from %s: %s", module_name, exc
            )


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
