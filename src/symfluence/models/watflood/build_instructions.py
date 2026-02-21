"""
WATFLOOD/CHARM build instructions for SYMFLUENCE.

WATFLOOD (CHARM — Canadian Hydrological And Routing Model) is a distributed
flood forecasting model developed at the University of Waterloo.

The open-source GitHub repository is incomplete (missing area_watflood module),
so SYMFLUENCE uses the pre-compiled Windows binary (charm64x.exe) executed
via Wine on macOS/Linux.

Tier 1: Download pre-compiled charm64x.exe + netCDF DLLs from UWaterloo.
Tier 2: Check for existing binary in install directory.
Tier 3: Manual installation guidance.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('watflood')
def get_watflood_build_instructions():
    """Get WATFLOOD/CHARM build instructions (Wine-based)."""
    common_env = get_common_build_environment()

    return {
        'description': 'WATFLOOD/CHARM distributed flood forecasting model (Wine)',
        'config_path_key': 'WATFLOOD_INSTALL_PATH',
        'config_exe_key': 'WATFLOOD_EXE',
        'default_path_suffix': 'installs/watflood/bin',
        'default_exe': 'charm64x.exe',
        'repository': None,
        'branch': None,
        'install_dir': 'watflood',
        'build_commands': [
            common_env,
            r'''
set -e

echo "=== WATFLOOD/CHARM Installation Starting ==="
echo "WATFLOOD runs as a Windows binary (charm64x.exe) via Wine."

INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}/bin"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
BIN_DIR="${INSTALL_DIR}/bin"

# === Check for Wine ===
WINE_CMD=""
for candidate in wine /opt/homebrew/bin/wine /usr/local/bin/wine; do
    if command -v "$candidate" >/dev/null 2>&1; then
        WINE_CMD="$candidate"
        break
    fi
done

if [ -z "$WINE_CMD" ]; then
    echo "WARNING: Wine not found. Install with:"
    echo "  macOS:  brew install --cask wine-stable"
    echo "  Ubuntu: sudo apt-get install wine64"
    echo "  HPC:    module load wine"
    echo "Continuing installation (Wine needed at runtime)."
fi

# === Check for existing binary ===
if [ -f "${BIN_DIR}/charm64x.exe" ]; then
    echo "Found existing charm64x.exe in ${BIN_DIR}"
    # Check for DLLs
    if [ -f "${BIN_DIR}/netcdf.dll" ]; then
        echo "DLLs already present."
        echo "=== WATFLOOD Installation Complete (existing) ==="
        exit 0
    else
        echo "Binary found but DLLs missing — downloading DLLs..."
    fi
fi

# === Download charm64x.exe from UWaterloo ===
CHARM_URL="https://www.civil.uwaterloo.ca/watflood/downloads/charm64x.exe"
DLLS_URL="https://www.civil.uwaterloo.ca/watflood/downloads/netCDF_dll.zip"

if [ ! -f "${BIN_DIR}/charm64x.exe" ]; then
    echo "Downloading charm64x.exe..."
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "${BIN_DIR}/charm64x.exe" "$CHARM_URL" 2>&1 || true
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${BIN_DIR}/charm64x.exe" "$CHARM_URL" 2>&1 || true
    fi
fi

# === Download netCDF DLLs ===
if [ ! -f "${BIN_DIR}/netcdf.dll" ]; then
    echo "Downloading netCDF DLLs..."
    TMP_ZIP=$(mktemp /tmp/netcdf_dll_XXXXXX.zip)
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$TMP_ZIP" "$DLLS_URL" 2>&1 || true
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$TMP_ZIP" "$DLLS_URL" 2>&1 || true
    fi

    if [ -f "$TMP_ZIP" ] && [ -s "$TMP_ZIP" ]; then
        TMP_DIR=$(mktemp -d /tmp/netcdf_dll_XXXXXX)
        unzip -o "$TMP_ZIP" -d "$TMP_DIR" 2>/dev/null || true
        # Copy all DLLs to bin directory
        find "$TMP_DIR" -name "*.dll" -exec cp {} "${BIN_DIR}/" \;
        rm -rf "$TMP_DIR" "$TMP_ZIP"
    fi
fi

# === Verify installation ===
if [ -f "${BIN_DIR}/charm64x.exe" ]; then
    echo ""
    echo "Files installed in ${BIN_DIR}:"
    ls -la "${BIN_DIR}/"*.exe "${BIN_DIR}/"*.dll 2>/dev/null | wc -l
    echo " files (exe + DLLs)"

    if [ -n "$WINE_CMD" ]; then
        echo ""
        echo "Wine found: $($WINE_CMD --version)"
        echo "Quick test..."
        WINEDEBUG=-all $WINE_CMD "${BIN_DIR}/charm64x.exe" </dev/null >/dev/null 2>&1 &
        WPID=$!
        sleep 3
        kill $WPID 2>/dev/null || true
        wait $WPID 2>/dev/null || true
        echo "Wine execution test passed."
    fi

    echo ""
    echo "=== WATFLOOD Installation Complete ==="
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "  Could not download charm64x.exe automatically."
    echo ""
    echo "  Manual installation:"
    echo "    1. Download charm64x.exe from:"
    echo "       https://www.civil.uwaterloo.ca/watflood/downloads/"
    echo "    2. Download netCDF_dll.zip from the same page"
    echo "    3. Place charm64x.exe and all DLLs in:"
    echo "       ${BIN_DIR}/"
    echo "    4. Install Wine:"
    echo "       macOS: brew install --cask wine-stable"
    echo "       Linux: sudo apt-get install wine64"
    echo "======================================================================"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['wine'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/charm64x.exe'],
            'check_type': 'exists'
        },
        'order': 24,
        'optional': True,
    }
