# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB build instructions for SYMFLUENCE."""
from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('pcrglobwb')
def get_pcrglobwb_build_instructions():
    """Get PCR-GLOBWB install instructions.

    PCR-GLOBWB requires PCRaster (C++ spatial modelling library).  Strategy:
    1. Clone the PCR-GLOBWB repository from UU-Hydro.
    2. Create a conda env with PCRaster matching the active Python version.
    3. The runner injects that env's site-packages via PYTHONPATH at runtime
       (same strategy as LISFLOOD).
    """
    return {
        'description': 'PCR-GLOBWB 2.0 Global Water Balance Model (Utrecht University)',
        'config_path_key': 'PCRGLOBWB_INSTALL_PATH',
        'config_exe_key': 'PCRGLOBWB_EXE',
        'default_path_suffix': 'installs/pcrglobwb',
        'default_exe': 'deterministic_runner.py',
        'repository': 'https://github.com/UU-Hydro/PCR-GLOBWB_model.git',
        'branch': 'master',
        'install_dir': 'pcrglobwb',
        'build_commands': [
            r"""
# PCR-GLOBWB Install Script for SYMFLUENCE
set -e
echo "=== PCR-GLOBWB Installation Starting ==="

INSTALL_DIR="$(pwd)"

# --- Locate the runner script ---
if [ -f "deterministic_runner.py" ]; then
    echo "Found deterministic_runner.py"
elif [ -f "model/deterministic_runner.py" ]; then
    echo "Found model/deterministic_runner.py"
    ln -sf model/deterministic_runner.py deterministic_runner.py
else
    echo "ERROR: deterministic_runner.py not found."
    exit 1
fi

# --- Ensure PCRaster is available via conda ---
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PCRASTER_ENV="pcraster${PYVER//./}"
echo "Looking for PCRaster (target conda env: $PCRASTER_ENV)..."

if python -c "import pcraster" 2>/dev/null; then
    echo "PCRaster already available in current environment"
elif ! command -v conda >/dev/null 2>&1; then
    echo "WARNING: PCRaster not found and conda not available."
    echo "Install conda (e.g. Miniforge), then run:"
    echo "  conda create -n $PCRASTER_ENV -c conda-forge python=$PYVER pcraster"
elif conda env list 2>/dev/null | grep -q "$PCRASTER_ENV"; then
    echo "Found existing conda env '$PCRASTER_ENV' with PCRaster"
else
    echo "Creating conda env '$PCRASTER_ENV' with PCRaster..."
    conda create -y -n "$PCRASTER_ENV" -c conda-forge "python=$PYVER" pcraster
    echo "PCRaster conda env created: $PCRASTER_ENV"
fi

echo "=== PCR-GLOBWB Installation Complete ==="
echo "Runner: deterministic_runner.py"
echo "Usage: python deterministic_runner.py <config.ini>"
            """.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['deterministic_runner.py'],
            'check_type': 'exists',
        },
        'order': 22,
        'optional': True,
    }
