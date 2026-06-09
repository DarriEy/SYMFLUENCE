# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM build instructions for SYMFLUENCE."""
from __future__ import annotations

from symfluence.core.registries import R


@R.build_instructions.add('cwatm')
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

# Compile kinematic wave routing C library for current architecture
# (CWatM ships x86_64 binaries; ARM Macs need recompilation)
T5_DIR="cwatm/hydrological_modules/routing_reservoirs"
if [ -f "$T5_DIR/t5.cpp" ]; then
    echo "Compiling kinematic wave routing library..."
    if [ "$(uname)" = "Darwin" ]; then
        c++ -shared -fPIC -O2 -o "$T5_DIR/t5_mac.so" "$T5_DIR/t5.cpp" 2>/dev/null && echo "Compiled t5_mac.so" || echo "WARNING: Could not compile t5.cpp"
    else
        c++ -shared -fPIC -O2 -o "$T5_DIR/t5_linux.so" "$T5_DIR/t5.cpp" 2>/dev/null && echo "Compiled t5_linux.so" || echo "WARNING: Could not compile t5.cpp"
    fi
fi

# Patch netCDF4 compatibility: modern netCDF4 rejects _FillValue as
# post-creation attribute. CWatM sets metadata attrs via exec() loops
# that include _FillValue from metaNetcdf.xml — skip it.
DH="cwatm/management_modules/data_handling.py"
if [ -f "$DH" ]; then
    echo "Patching netCDF4 compatibility..."
    python -c "
import re
with open('$DH') as f: c = f.read()
c = re.sub(
    r'(\\s+)(exec\\(\\'%s=\\\\\"%s\\\\\"\\'\\s*%\\s*\\(\\\"\\w+\\.\\\" \\+ i)',
    r\"\\1if i != '_FillValue':\\n\\1    \\2\",
    c)
with open('$DH','w') as f: f.write(c)
" 2>/dev/null && echo "Patched data_handling.py" || echo "WARNING: Could not patch data_handling.py"
fi

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
