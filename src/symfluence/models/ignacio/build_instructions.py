# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO build instructions for SYMFLUENCE.

This module defines how to install IGNACIO (Fire-Engine-Framework) from source.
IGNACIO is a Python package implementing the Canadian FBP System for fire
spread simulation with Richards' elliptical wave propagation.

Unlike compiled tools, IGNACIO is installed via pip as an editable package.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('ignacio')
def get_ignacio_build_instructions():
    """
    Get IGNACIO build instructions.

    IGNACIO is a Python package that implements the Canadian Forest Fire
    Behavior Prediction (FBP) System. It is installed via pip from the
    cloned repository.

    Returns:
        Dictionary with complete build configuration for IGNACIO.
    """
    return {
        'description': 'IGNACIO - Canadian FBP System fire spread model',
        'config_path_key': 'IGNACIO_INSTALL_PATH',
        'config_exe_key': 'IGNACIO_CLI',
        'default_path_suffix': 'installs/ignacio',
        'default_exe': 'ignacio',  # CLI entry point after pip install
        'repository': 'https://github.com/KatherineHopeReece/Fire-Engine-Framework.git',
        'branch': None,
        'install_dir': 'ignacio',
        'build_commands': [
            r'''
set -e
echo "Installing IGNACIO (Fire-Engine-Framework)..."

# Verify we're in the ignacio directory with pyproject.toml
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found. Are we in the ignacio directory?"
    exit 1
fi

# Use python -m pip to ensure we install to the correct Python environment
# On Windows, python3 triggers MS Store redirect; conda provides python
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/python.exe" ]; then
    PYTHON_CMD="$CONDA_PREFIX/python.exe"
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
    PYTHON_CMD="${PYTHON:-python}"
else
    PYTHON_CMD="${PYTHON:-python3}"
fi
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Patch upstream bug: simulation.py imports rasterize_geometries but io.py
# only defines rasterize_polygons. Add the missing function.
if [ -f "ignacio/io.py" ] && ! grep -q "def rasterize_geometries" ignacio/io.py; then
    echo "Patching upstream bug: adding missing rasterize_geometries to io.py"
    cat >> ignacio/io.py << 'PATCH'


def rasterize_geometries(
    gdf,
    reference,
    value=1,
    fill=0,
    dtype="uint8",
):
    """
    Rasterize geometries from a GeoDataFrame onto a reference grid.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with geometries to rasterize.
    reference : RasterData
        Reference raster for extent and resolution.
    value : int or float
        Value to burn for all geometries.
    fill : int or float
        Fill value for areas outside geometries.
    dtype : str
        Output data type.

    Returns
    -------
    np.ndarray
        Rasterized array matching reference dimensions.
    """
    from rasterio.features import rasterize as _rasterize

    shapes = [(geom, value) for geom in gdf.geometry]
    return _rasterize(
        shapes=shapes,
        out_shape=reference.shape,
        transform=reference.transform,
        fill=fill,
        dtype=dtype,
    )
PATCH
fi

# Install in editable mode with pip
echo "Installing IGNACIO in editable mode..."
$PYTHON_CMD -m pip install -e . --quiet

# Verify installation
echo "Verifying installation..."
if $PYTHON_CMD -c "import ignacio; print(f'IGNACIO version: {getattr(ignacio, \"__version__\", \"0.1.0\")}')" 2>/dev/null; then
    echo "IGNACIO Python package installed successfully"
else
    echo "ERROR: IGNACIO Python package installation failed"
    $PYTHON_CMD -c "import ignacio" 2>&1 | tail -5
    exit 1
fi

# Verify CLI is available (may not be in PATH for editable installs)
if command -v ignacio >/dev/null 2>&1; then
    echo "IGNACIO CLI available: $(which ignacio)"
else
    echo "Note: IGNACIO CLI may need 'python3 -m ignacio.cli' to run"
fi

echo "IGNACIO installation complete"
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--help',  # Test CLI with --help
        'verify_install': {
            'python_import': 'ignacio',
            'check_type': 'python_import'
        },
        'order': 15,  # Install after other fire models
        'optional': True,  # Not installed by default with --install
    }
