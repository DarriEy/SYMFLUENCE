"""
t-route build instructions for SYMFLUENCE.

This module defines how to install t-route, including:
- Repository information
- Installation commands (Python package)
- Installation verification criteria

t-route is NOAA's Next Generation river routing model implemented as
a Python package.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('troute')
def get_troute_build_instructions():
    """
    Get t-route build instructions.

    t-route is a Python package that requires pip installation.
    The installation is non-fatal - if it fails, SYMFLUENCE continues.

    Returns:
        Dictionary with complete build configuration for t-route.
    """
    return {
        'description': "NOAA's Next Generation river routing model",
        'config_path_key': 'TROUTE_INSTALL_PATH',
        'config_exe_key': 'TROUTE_PKG_PATH',
        'default_path_suffix': 'installs/t-route/src/troute-network',
        'default_exe': 'troute/network/__init__.py',
        'repository': 'https://github.com/NOAA-OWP/t-route.git',
        'branch': None,
        'install_dir': 't-route',
        'build_commands': [
            r'''
# Non-fatal installation of the t-route Python package.
# t-route intentionally does not stop SYMFLUENCE builds on pip failure.

set +e

echo "Installing t-route Python package (non-fatal if this step fails)..."

cd src/troute-network 2>/dev/null || {
    echo "src/troute-network not found; skipping t-route install."
    exit 0
}

PYTHON_BIN="${SYMFLUENCE_PYTHON:-python3}"

# Upgrade tools quietly; failure is allowed
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# Try installing the package; if it fails, SYMFLUENCE continues
"$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps || true

echo "t-route installation attempt complete (see any errors above)."
exit 0
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': [
                'src/troute-network/troute/network/__init__.py',
                'troute/network/__init__.py',
            ],
            'check_type': 'exists_any'
        },
        'order': 4
    }
