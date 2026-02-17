"""
PIHM (MM-PIHM) build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from MM-PIHM GitHub releases.
Tier 2 (fallback): Build from source using CMake with SUNDIALS dependency.

MM-PIHM is the actively maintained multi-module variant of the Penn State
Integrated Hydrologic Model. Pre-compiled binaries may be available for
Linux and macOS.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('pihm')
def get_pihm_build_instructions():
    """
    Get PIHM (MM-PIHM) build/install instructions.

    Downloads pre-compiled binary from GitHub releases or builds from
    source with CMake + SUNDIALS.

    Returns:
        Dictionary with complete build configuration for PIHM.
    """
    return {
        'description': 'PIHM (Penn State Integrated Hydrologic Model / MM-PIHM)',
        'config_path_key': 'PIHM_INSTALL_PATH',
        'config_exe_key': 'PIHM_EXE',
        'default_path_suffix': 'installs/pihm/bin',
        'default_exe': 'pihm',
        'repository': 'https://github.com/PSUmodeling/MM-PIHM.git',
        'branch': 'master',
        'install_dir': 'pihm',
        'build_commands': [
            r'''
# MM-PIHM Install Script for SYMFLUENCE
# Tier 1: Download pre-compiled binary from GitHub releases
# Tier 2: Build from source with CMake + SUNDIALS (fallback)

set -e

echo "=== MM-PIHM Installation Starting ==="

# Resolve INSTALL_DIR to absolute path
INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
mkdir -p "${INSTALL_DIR}/bin"

echo "Install directory: ${INSTALL_DIR}"

# Platform detection
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

case "$UNAME_S" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        if [ "$UNAME_M" = "arm64" ]; then
            PLATFORM="darwin-arm64"
        else
            PLATFORM="darwin-x86_64"
        fi
        ;;
    *)
        echo "WARNING: Unknown platform $UNAME_S, will try source build"
        PLATFORM="unknown"
        ;;
esac

echo "Detected platform: $PLATFORM ($UNAME_S $UNAME_M)"

# === Tier 1: Download pre-compiled binary ===
DOWNLOAD_SUCCESS=false

if [ "$PLATFORM" != "unknown" ]; then
    echo "Attempting binary download from MM-PIHM GitHub releases..."

    _api_url="https://api.github.com/repos/PSUmodeling/MM-PIHM/releases/latest"
    _api_json=""
    if command -v curl >/dev/null 2>&1; then
        _api_json=$(curl -fsSL -H "Accept: application/vnd.github+json" "$_api_url" 2>/dev/null) || true
    fi
    if [ -z "$_api_json" ] && command -v wget >/dev/null 2>&1; then
        _api_json=$(wget -qO- --header="Accept: application/vnd.github+json" "$_api_url" 2>/dev/null) || true
    fi
    LATEST_TAG=""
    if [ -n "$_api_json" ]; then
        LATEST_TAG=$(echo "$_api_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tag_name', ''))" 2>/dev/null) || true
    fi

    if [ -z "$LATEST_TAG" ]; then
        echo "WARNING: Could not determine latest release tag"
        LATEST_TAG="v2.0"
    fi

    echo "Latest release: $LATEST_TAG"

    DOWNLOAD_URL="https://github.com/PSUmodeling/MM-PIHM/releases/download/${LATEST_TAG}/mm-pihm-${LATEST_TAG}-${PLATFORM}.tar.gz"
    echo "Download URL: $DOWNLOAD_URL"

    TMPTAR=$(mktemp /tmp/pihm_XXXXXX.tar.gz)
    TMPEXTRACT=$(mktemp -d /tmp/pihm_extract_XXXXXX)

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ ! -s "$TMPTAR" ] && command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ -s "$TMPTAR" ]; then
        if file "$TMPTAR" | grep -q "gzip\|tar"; then
            tar xzf "$TMPTAR" -C "$TMPEXTRACT" 2>/dev/null || true

            PIHM_BIN=$(find "$TMPEXTRACT" -name "pihm" -o -name "mm-pihm" | head -1)

            if [ -n "$PIHM_BIN" ]; then
                cp "$PIHM_BIN" "${INSTALL_DIR}/bin/pihm"
                chmod +x "${INSTALL_DIR}/bin/pihm"

                if "${INSTALL_DIR}/bin/pihm" --version >/dev/null 2>&1 || \
                   "${INSTALL_DIR}/bin/pihm" -v >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                else
                    echo "WARNING: Downloaded pihm binary cannot run on this system"
                    rm -f "${INSTALL_DIR}/bin/pihm"
                fi
            fi
        fi
    fi

    rm -f "$TMPTAR"
    rm -rf "$TMPEXTRACT"
fi

# === Tier 2: Build from source with CMake ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Binary download failed, attempting source build with CMake..."

    for tool in cmake make; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "ERROR: $tool not found. Install with: brew install cmake (or apt install cmake)"
            exit 1
        fi
    done

    BUILD_TMPDIR=$(mktemp -d /tmp/pihm_build_XXXXXX)
    echo "Build directory: ${BUILD_TMPDIR}"

    echo "Cloning MM-PIHM source..."
    git clone --depth 1 https://github.com/PSUmodeling/MM-PIHM.git "${BUILD_TMPDIR}/mm-pihm"

    mkdir -p "${BUILD_TMPDIR}/mm-pihm/build"

    echo "Configuring CMake build..."
    cmake -S "${BUILD_TMPDIR}/mm-pihm" \
          -B "${BUILD_TMPDIR}/mm-pihm/build" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

    echo "Compiling..."
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "${BUILD_TMPDIR}/mm-pihm/build" -j "${NCPU}"

    echo "Installing to ${INSTALL_DIR}..."
    cmake --install "${BUILD_TMPDIR}/mm-pihm/build" 2>/dev/null || \
        cp "${BUILD_TMPDIR}/mm-pihm/build/pihm" "${INSTALL_DIR}/bin/pihm" 2>/dev/null || \
        cp "${BUILD_TMPDIR}/mm-pihm/build/mm-pihm" "${INSTALL_DIR}/bin/pihm" 2>/dev/null

    chmod +x "${INSTALL_DIR}/bin/pihm" 2>/dev/null || true

    rm -rf "${BUILD_TMPDIR}"
    echo "Build artifacts cleaned up"
fi

# === Verify installation ===
if [ -f "${INSTALL_DIR}/bin/pihm" ]; then
    echo ""
    echo "=== MM-PIHM Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/pihm"
    "${INSTALL_DIR}/bin/pihm" -v 2>/dev/null || echo "(version check not supported)"
else
    echo "ERROR: pihm not found at ${INSTALL_DIR}/bin/pihm"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'make'],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/pihm'],
            'check_type': 'exists'
        },
        'order': 27,  # After ParFlow (26)
        'optional': True,
    }
