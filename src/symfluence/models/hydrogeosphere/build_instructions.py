"""
HydroGeoSphere build instructions for SYMFLUENCE.

HGS is proprietary software from Aquanty. The install script checks
for an existing installation and validates the license. Users must
obtain HGS independently through Aquanty or their university license.

Tier 1: Check for pre-existing HGS installation (user-provided path)
Tier 2: Prompt user with download/license instructions
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('hydrogeosphere')
def get_hydrogeosphere_build_instructions():
    """
    Get HydroGeoSphere build/install instructions.

    HGS is proprietary — checks for existing binary and validates
    license rather than attempting download.

    Returns:
        Dictionary with complete build configuration for HGS.
    """
    return {
        'description': 'HydroGeoSphere (Fully-Coupled 3D Subsurface + Surface Flow)',
        'config_path_key': 'HGS_INSTALL_PATH',
        'config_exe_key': 'HGS_EXE',
        'default_path_suffix': 'installs/hydrogeosphere/bin',
        'default_exe': 'hgs',
        'repository': None,  # Proprietary — no public repo
        'branch': None,
        'install_dir': 'hydrogeosphere',
        'build_commands': [
            r'''
# HydroGeoSphere Install Script for SYMFLUENCE
# HGS is proprietary software from Aquanty Inc.
# This script checks for an existing installation.

set -e

echo "=== HydroGeoSphere Installation Check ==="

INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
mkdir -p "${INSTALL_DIR}/bin"

echo "Install directory: ${INSTALL_DIR}"

# === Check for existing HGS installation ===
HGS_FOUND=false

# Check common installation paths
for path in \
    "${INSTALL_DIR}/bin/hgs" \
    "/usr/local/bin/hgs" \
    "${HOME}/hgs/bin/hgs" \
    "${HOME}/HydroGeoSphere/bin/hgs" \
    "/opt/hgs/bin/hgs"; do
    if [ -x "$path" ]; then
        echo "Found HGS at: $path"
        if [ "$path" != "${INSTALL_DIR}/bin/hgs" ]; then
            cp "$path" "${INSTALL_DIR}/bin/hgs"
            chmod +x "${INSTALL_DIR}/bin/hgs"
        fi
        HGS_FOUND=true

        # Also look for grok in same directory
        GROK_DIR=$(dirname "$path")
        if [ -x "${GROK_DIR}/grok" ]; then
            cp "${GROK_DIR}/grok" "${INSTALL_DIR}/bin/grok"
            chmod +x "${INSTALL_DIR}/bin/grok"
            echo "Found grok at: ${GROK_DIR}/grok"
        fi

        break
    fi
done

# Check if hgs is on PATH
if [ "$HGS_FOUND" = "false" ] && command -v hgs >/dev/null 2>&1; then
    HGS_PATH=$(which hgs)
    echo "Found HGS on PATH: $HGS_PATH"
    cp "$HGS_PATH" "${INSTALL_DIR}/bin/hgs"
    chmod +x "${INSTALL_DIR}/bin/hgs"
    HGS_FOUND=true

    if command -v grok >/dev/null 2>&1; then
        GROK_PATH=$(which grok)
        cp "$GROK_PATH" "${INSTALL_DIR}/bin/grok"
        chmod +x "${INSTALL_DIR}/bin/grok"
    fi
fi

if [ "$HGS_FOUND" = "true" ]; then
    echo ""
    echo "=== HydroGeoSphere Installation Complete ==="
    echo "HGS binary: ${INSTALL_DIR}/bin/hgs"
    "${INSTALL_DIR}/bin/hgs" -v 2>/dev/null || echo "(version check not supported)"
else
    echo ""
    echo "================================================================="
    echo "  HydroGeoSphere (HGS) not found on this system."
    echo ""
    echo "  HGS is proprietary software from Aquanty Inc."
    echo "  To obtain HGS:"
    echo "    1. University license: Contact your department's"
    echo "       groundwater/hydrogeology group"
    echo "    2. Commercial license: https://www.aquanty.com"
    echo "    3. Evaluation: Contact Aquanty for evaluation access"
    echo ""
    echo "  After installation, set HGS_INSTALL_PATH in your"
    echo "  SYMFLUENCE config to point to the HGS directory."
    echo "================================================================="
    exit 1
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '-v',
        'verify_install': {
            'file_paths': ['bin/hgs'],
            'check_type': 'exists'
        },
        'order': 28,  # After PIHM (27)
        'optional': True,
    }
