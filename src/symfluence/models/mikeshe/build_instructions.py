"""
MIKE-SHE build instructions for SYMFLUENCE.

MIKE-SHE is a proprietary model developed by DHI. It cannot be built
from source. A valid license from DHI is required to run MikeSheEngine.exe.

This module registers a build instructions entry that documents the
license requirement and raises an informative error message.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('mikeshe')
def get_mikeshe_build_instructions():
    """
    Get MIKE-SHE build/install instructions.

    MIKE-SHE is proprietary software from DHI. There is no source code
    to compile. Users must obtain a license and install the software
    through DHI's official distribution channels.

    Returns:
        Dictionary with build configuration documenting the proprietary
        nature of MIKE-SHE and the license requirement.
    """
    return {
        'description': (
            'MIKE-SHE (DHI) - PROPRIETARY. '
            'Requires a valid DHI license. '
            'Cannot be built from source. '
            'Install via DHI official distribution at https://www.dhigroup.com/'
        ),
        'config_path_key': 'MIKESHE_INSTALL_PATH',
        'config_exe_key': 'MIKESHE_EXE',
        'default_path_suffix': 'installs/mikeshe/bin',
        'default_exe': 'MikeSheEngine.exe',
        'repository': None,
        'branch': None,
        'install_dir': 'mikeshe',
        'build_commands': [
            r'''
# MIKE-SHE Installation Notice for SYMFLUENCE
# MIKE-SHE is PROPRIETARY software from DHI (Danish Hydraulic Institute).
# There is NO source code to compile.

echo "============================================================"
echo "ERROR: MIKE-SHE is proprietary software from DHI."
echo ""
echo "MIKE-SHE cannot be built from source."
echo "A valid DHI license is required to use MikeSheEngine.exe."
echo ""
echo "To obtain MIKE-SHE:"
echo "  1. Visit https://www.dhigroup.com/"
echo "  2. Purchase a MIKE-SHE license"
echo "  3. Install via DHI's official installer"
echo "  4. Set MIKESHE_INSTALL_PATH in your SYMFLUENCE config"
echo "     to point to the directory containing MikeSheEngine.exe"
echo ""
echo "For Unix/Linux systems:"
echo "  - Set MIKESHE_USE_WINE=True in config to run via WINE"
echo "  - Ensure WINE is installed: sudo apt install wine"
echo "============================================================"
exit 1
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/MikeSheEngine.exe'],
            'check_type': 'exists'
        },
        'order': 50  # Low priority, proprietary
    }
