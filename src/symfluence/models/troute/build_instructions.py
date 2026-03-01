# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
t-route build instructions for SYMFLUENCE.

This module defines how to install t-route, including:
- Repository information
- Installation commands (Python package)
- Installation verification criteria

t-route is NOAA's Next Generation river routing model implemented as
a Python package. It consists of 4 sub-packages that must be installed
in order:
  1. troute-config (pure Python)
  2. troute-network (Cython MC extensions + optional Fortran reservoir extensions)
  3. troute-routing (Cython MC/DA extensions + optional Fortran diffusive/reach extensions)
  4. troute-nwm (pure Python CLI entry point)
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('troute')
def get_troute_build_instructions():
    """
    Get t-route build instructions.

    Installs all 4 t-route sub-packages in dependency order.
    The Cython MC extensions (reach.pyx, mc_reach.pyx) require numpy + Cython>=3.
    Reservoir Fortran extensions are optional (require gfortran + netcdf-fortran).

    Build is fatal — errors are reported clearly instead of silently swallowed.

    Returns:
        Dictionary with complete build configuration for t-route.
    """
    return {
        'description': "NOAA's Next Generation river routing model",
        'config_path_key': 'TROUTE_INSTALL_PATH',
        'config_exe_key': 'TROUTE_PKG_PATH',
        'default_path_suffix': 'installs/t-route/src/troute-nwm',
        'default_exe': 'src/nwm_routing/__main__.py',
        'repository': 'https://github.com/NOAA-OWP/t-route.git',
        'branch': None,
        'install_dir': 't-route',
        'build_commands': [
            r'''
set -e

echo "=== Installing t-route (all 4 sub-packages) ==="

# --- Detect Python ---
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/python.exe" ]; then
    PYTHON_BIN="$CONDA_PREFIX/python.exe"
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
    PYTHON_BIN="${SYMFLUENCE_PYTHON:-python}"
else
    PYTHON_BIN="${SYMFLUENCE_PYTHON:-python3}"
fi
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# --- Install build dependencies if missing ---
"$PYTHON_BIN" -m pip install 'Cython>=3' numpy -q 2>&1

"$PYTHON_BIN" -c "
import Cython
major = int(Cython.__version__.split('.')[0])
assert major >= 3, f'Cython {Cython.__version__} < 3.0'
print(f'Cython {Cython.__version__} OK')
" || {
    echo "ERROR: Cython >= 3.0 install failed"
    exit 1
}

"$PYTHON_BIN" -c "import numpy; print(f'NumPy {numpy.__version__} OK')" || {
    echo "ERROR: NumPy install failed"
    exit 1
}

# --- Check for gfortran (optional — needed for reservoir extensions only) ---
# Note: macOS has /usr/bin/fc (bash history builtin), NOT a Fortran compiler.
# We must check specifically for gfortran, not fc.
HAS_FORTRAN=false
if command -v gfortran >/dev/null 2>&1; then
    # Verify it actually works (not just exists)
    if gfortran --version >/dev/null 2>&1; then
        HAS_FORTRAN=true
        export FC=gfortran
        export F90=gfortran
        echo "gfortran found — reservoir extensions will be built"
    fi
fi
if [ "$HAS_FORTRAN" = false ]; then
    echo "gfortran not found — building MC-only (no reservoir routing)"
fi

# --- Verify source tree ---
for pkg in troute-config troute-network troute-routing troute-nwm; do
    if [ ! -d "src/$pkg" ]; then
        echo "ERROR: src/$pkg not found in t-route repository"
        exit 1
    fi
done

# --- 1. Install troute-config (pure Python, always succeeds) ---
echo ""
echo "--- [1/4] Installing troute-config ---"
cd src/troute-config
"$PYTHON_BIN" -m pip install . --no-deps -q
cd ../..
echo "troute-config installed OK"

# --- 2. Install troute-network (Cython — the critical package) ---
# troute-network's setup.py has an unconditional Fortran compiler check
# that runs during metadata generation. For MC-only builds we must patch
# setup.py to remove the Fortran detection and reservoir extensions.
echo ""
echo "--- [2/4] Installing troute-network (Cython extensions) ---"
cd src/troute-network

# Save original setup.py for potential restore
cp setup.py setup.py.orig

_patch_setup_mc_only() {
    # Patch setup.py: replace Fortran detection block with a no-op and
    # remove reservoir extensions from ext_modules.
    # Write patch script to a temp file to avoid shell quoting issues.
    cat > /tmp/_sf_patch_setup.py << 'PYEOF'
import re

with open('setup.py', 'r') as f:
    src = f.read()

# 1. Replace Fortran detection block (fcompopt through build_ext_subclass)
#    with a trivial no-op version
pattern = r'fcompopt = \{.*?class build_ext_subclass\( build_ext \):.*?build_ext\.build_extensions\(self\)'
replacement = ('fcompiler_type = None\n\n'
               'class build_ext_subclass( build_ext ):\n'
               '    def build_extensions(self):\n'
               '        build_ext.build_extensions(self)')
src = re.sub(pattern, replacement, src, flags=re.DOTALL)

# 2. Remove reservoir extensions from ext_modules list
src = src.replace(
    'ext_modules = [reach, levelpool_reservoirs, rfc_reservoirs, musk]',
    'ext_modules = [reach, musk]'
)

with open('setup.py', 'w') as f:
    f.write(src)

print('setup.py patched for MC-only build')
PYEOF
    "$PYTHON_BIN" /tmp/_sf_patch_setup.py
    rm -f /tmp/_sf_patch_setup.py
}

# Pre-generate .c files from .pyx if they don't exist.
# The repo ships with .pyx (Cython) but not pre-generated .c files.
# setup.py defaults to USE_CYTHON=False which expects .c files.
_cythonize_sources() {
    echo "Pre-generating .c files from Cython .pyx sources..."
    "$PYTHON_BIN" -m cython troute/network/reach.pyx 2>&1
    "$PYTHON_BIN" -m cython troute/network/musking/mc_reach.pyx 2>&1
    if [ "$1" = "full" ]; then
        # Also cythonize reservoir extensions for full build
        "$PYTHON_BIN" -m cython troute/network/reservoirs/levelpool/levelpool.pyx 2>&1 || true
        "$PYTHON_BIN" -m cython troute/network/reservoirs/rfc/rfc.pyx 2>&1 || true
    fi
}

BUILD_OK=false
if [ "$HAS_FORTRAN" = true ]; then
    # Try full build with FC=gfortran (MC + reservoir extensions)
    echo "Attempting full build (MC + reservoirs)..."
    _cythonize_sources full
    if "$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps -q 2>&1; then
        BUILD_OK=true
    else
        echo "Full build failed, retrying MC-only (without reservoir extensions)..."
        cp setup.py.orig setup.py
        _patch_setup_mc_only
        _cythonize_sources mc
        if "$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps -q 2>&1; then
            BUILD_OK=true
        fi
    fi
else
    # No Fortran — patch setup.py for MC-only build
    _patch_setup_mc_only
    _cythonize_sources mc
    if "$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps -q 2>&1; then
        BUILD_OK=true
    fi
fi

# Restore original setup.py
cp setup.py.orig setup.py
rm -f setup.py.orig

if [ "$BUILD_OK" = false ]; then
    echo "ERROR: troute-network Cython build failed"
    exit 1
fi
cd ../..
echo "troute-network installed OK"

# --- 3. Install troute-routing (Python modules only, skip Cython extensions) ---
# troute-routing's Cython mc_reach.pyx cimports troute.network.reservoirs.levelpool
# which requires pre-compiled Fortran libraries (binding_lp.a, netcdf-fortran).
# Since SYMFLUENCE uses its own builtin MC routing (pure Python/NumPy), we only
# need troute-routing's Python modules (compute.py, diffusive_utils.py, etc.).
# We patch setup.py to remove all ext_modules and Fortran detection.
echo ""
echo "--- [3/4] Installing troute-routing (Python modules only) ---"
cd src/troute-routing

cp setup.py setup.py.orig

cat > /tmp/_sf_patch_routing_pyonly.py << 'PYEOF'
import re

with open('setup.py', 'r') as f:
    src = f.read()

# 1. Replace Fortran detection block with no-op
pattern = r'fcompopt = \{.*?class build_ext_subclass\( build_ext \):.*?build_ext\.build_extensions\(self\)'
replacement = ('fcompiler_type = None\n\n'
               'class build_ext_subclass( build_ext ):\n'
               '    def build_extensions(self):\n'
               '        build_ext.build_extensions(self)')
src = re.sub(pattern, replacement, src, flags=re.DOTALL)

# 2. Remove ALL ext_modules (Cython extensions need reservoir Fortran libs)
src = src.replace(
    'ext_modules = [reach, mc_reach, diffusive, simple_da, chxsec_lookuptable]',
    'ext_modules = []'
)

# 3. Remove package_data referencing .pxd files
src = re.sub(r'package_data = \{.*?\}', 'package_data = {}', src, flags=re.DOTALL)

with open('setup.py', 'w') as f:
    f.write(src)

print('troute-routing setup.py patched for Python-only build')
PYEOF
"$PYTHON_BIN" /tmp/_sf_patch_routing_pyonly.py
rm -f /tmp/_sf_patch_routing_pyonly.py

if "$PYTHON_BIN" -m pip install . --no-build-isolation --no-deps -q 2>&1; then
    echo "troute-routing Python modules installed OK"
else
    echo "ERROR: troute-routing Python-only install failed"
    cp setup.py.orig setup.py
    rm -f setup.py.orig
    exit 1
fi

cp setup.py.orig setup.py
rm -f setup.py.orig
cd ../..
echo "troute-routing installed OK"

# --- 4. Install troute-nwm (CLI entry point, pure Python) ---
echo ""
echo "--- [4/4] Installing troute-nwm ---"
cd src/troute-nwm
"$PYTHON_BIN" -m pip install . --no-deps -q
cd ../..
echo "troute-nwm installed OK"

# --- Install missing runtime dependencies ---
# troute sub-packages were installed with --no-deps to avoid build isolation issues.
# Most deps (numpy, pandas, xarray, netcdf4, etc.) are already in the SYMFLUENCE venv.
# Install the few lightweight deps that are likely missing.
echo ""
echo "--- Installing missing runtime dependencies ---"
"$PYTHON_BIN" -m pip install deprecated toolz joblib pyarrow -q 2>&1 || true

# --- Verify installation ---
# troute-config requires pydantic v1, which conflicts with SYMFLUENCE's pydantic v2.
# We verify the network Cython extensions (MC structures) and Python routing modules.
echo ""
echo "=== Verifying t-route installation ==="
"$PYTHON_BIN" -c "
import troute.network.reach
print('troute.network.reach Cython extension OK')
import troute.network.musking.mc_reach
print('troute.network.musking.mc_reach Cython extension OK')
print('All t-route Cython extensions verified')
" || {
    echo "ERROR: t-route Cython extension verification failed"
    exit 1
}

echo ""
echo "=== t-route installation complete ==="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'python_import': 'troute.network.musking.mc_reach',
            'check_type': 'python_import'
        },
        'order': 4
    }
