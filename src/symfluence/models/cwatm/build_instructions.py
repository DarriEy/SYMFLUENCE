# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM build instructions for SYMFLUENCE."""
from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('cwatm')
def get_cwatm_build_instructions():
    """Get CWatM install instructions.

    CWatM is a pure Python model with no PCRaster dependency.
    Installation: clone the repository and install numpy/scipy/netCDF4.
    """
    return {
        'description': 'CWatM Community Water Model (IIASA)',
        'config_path_key': 'CWATM_INSTALL_PATH',
        'config_exe_key': 'CWATM_EXE',
        'default_path_suffix': 'installs/cwatm',
        'default_exe': 'run_cwatm.py',
        'repository': 'https://github.com/iiasa/CWatM.git',
        'branch': 'main',
        'install_dir': 'cwatm',
        'build_commands': [
            r"""
# CWatM Install Script for SYMFLUENCE
set -e
echo "=== CWatM Installation Starting ==="

# Locate runner script
if [ -f "run_cwatm.py" ]; then
    echo "Found run_cwatm.py"
elif [ -f "cwatm/run_cwatm.py" ]; then
    echo "Found cwatm/run_cwatm.py"
    ln -sf cwatm/run_cwatm.py run_cwatm.py
fi

# Check dependencies
python -c "import numpy, scipy, netCDF4; print('Dependencies OK')" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install numpy scipy netCDF4 2>/dev/null || true
}

echo "=== CWatM Installation Complete ==="
echo "Runner: run_cwatm.py"
echo "Usage: python run_cwatm.py <settings.ini>"
            """.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['run_cwatm.py'],
            'check_type': 'exists',
        },
        'order': 23,
        'optional': True,
    }
