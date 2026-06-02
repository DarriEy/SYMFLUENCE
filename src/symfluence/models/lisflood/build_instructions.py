# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD build instructions for SYMFLUENCE."""

from symfluence.core.registries import R


@R.build_instructions.add("lisflood")
def get_lisflood_build_instructions():
    """Get LISFLOOD install instructions.

    LISFLOOD-OS requires PCRaster (C++ spatial modelling library). Strategy:
    1. Install lisflood-model + deps via pip into the SYMFLUENCE venv.
    2. Patch NumPy 2.x compatibility issues in installed lisflood source.
    3. Create a conda env with PCRaster matching the active Python version.
    4. The runner injects that env's site-packages via PYTHONPATH at runtime.
    """
    return {
        "description": "LISFLOOD Distributed Hydrological Model (JRC/Deltares)",
        "config_path_key": "LISFLOOD_INSTALL_PATH",
        "config_exe_key": "LISFLOOD_EXE",
        "default_path_suffix": "installs/lisflood/bin",
        "default_exe": "lisflood",
        "repository": None,
        "branch": None,
        "install_dir": "lisflood",
        "build_commands": [
            r"""
# LISFLOOD Install Script for SYMFLUENCE
set -e
echo "=== LISFLOOD Installation Starting ==="

INSTALL_DIR="$(pwd)"
mkdir -p bin

# --- Install lisflood-model into the active Python environment ---
echo "Installing lisflood-model and dependencies..."
pip install lisflood-model --no-deps
pip install lisflood-utilities --no-deps 2>/dev/null || true
pip install future nine 2>/dev/null || true
echo "lisflood-model installed"

# --- Patch NumPy 2.x compatibility ---
# lisflood-model 4.x uses deprecated NumPy APIs removed in NumPy 2.0:
#   np.bool8 (removed, use np.bool_)
#   ndarray.tostring() (removed, use .tobytes())
# These patches are safe on NumPy 1.x too (the replacements exist there).
LISFLOOD_PKG=$(python -c "import lisflood; import os; print(os.path.dirname(lisflood.__file__))" 2>/dev/null || echo "")
if [ -n "$LISFLOOD_PKG" ]; then
    echo "Patching lisflood for NumPy 2.x compatibility..."
    find "$LISFLOOD_PKG" -name '*.py' -not -path '*__pycache__*' \
        -exec sed -i.bak 's/np\.bool8/np.bool_/g' {} + 2>/dev/null || true
    find "$LISFLOOD_PKG" -name '*.py' -not -path '*__pycache__*' \
        -exec sed -i.bak 's/\.tostring()/.tobytes()/g' {} + 2>/dev/null || true
    # pandas 2.x: squeeze() on single-column DataFrame returns scalar
    KINWAVE="$LISFLOOD_PKG/hydrological_modules/kinematic_wave_parallel.py"
    if [ -f "$KINWAVE" ]; then
        sed -i.bak 's/\.set_index("order")\.squeeze()/.set_index("order")["pixels"]/g' "$KINWAVE" 2>/dev/null || true
    fi
    find "$LISFLOOD_PKG" -name '*.bak' -delete 2>/dev/null || true
    echo "Patches applied"
else
    echo "WARNING: Could not locate lisflood package for patching"
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

# --- Create marker script ---
cat > bin/lisflood << 'WRAPEOF'
#!/bin/bash
# LISFLOOD is invoked directly by the SYMFLUENCE runner via Python.
# This file exists as an installation marker.
echo "LISFLOOD is installed. Use 'symfluence workflow' to run it."
WRAPEOF
chmod +x bin/lisflood

echo "=== LISFLOOD Installation Complete ==="
            """.strip()
        ],
        "dependencies": [],
        "test_command": None,
        "verify_install": {
            "file_paths": ["bin/lisflood"],
            "check_type": "exists",
        },
        "order": 21,
        "optional": True,
    }
