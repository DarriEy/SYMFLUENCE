"""
MODFLOW 6 build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from USGS GitHub releases.
Tier 2 (fallback): Build from source using meson.

MODFLOW 6 is the USGS modular groundwater flow model. Pre-compiled
binaries are available for Linux, macOS (Intel + ARM), and Windows.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('modflow')
def get_modflow_build_instructions():
    """
    Get MODFLOW 6 build/install instructions.

    Downloads pre-compiled mf6 binary from USGS GitHub releases.
    Falls back to meson source build if download fails.

    Returns:
        Dictionary with complete build configuration for MODFLOW 6.
    """
    return {
        'description': 'MODFLOW 6 (USGS Modular Groundwater Flow Model)',
        'config_path_key': 'MODFLOW_INSTALL_PATH',
        'config_exe_key': 'MODFLOW_EXE',
        'default_path_suffix': 'installs/modflow/bin',
        'default_exe': 'mf6',
        'repository': 'https://github.com/MODFLOW-ORG/modflow6.git',
        'branch': 'master',
        'install_dir': 'modflow',
        'build_commands': [
            r'''
# MODFLOW 6 Install Script for SYMFLUENCE
# Tier 1: Download pre-compiled binary from USGS GitHub releases
# Tier 2: Build from source with meson (fallback)

echo "=== MODFLOW 6 Installation Starting ==="

# Determine install directory (meson requires an absolute prefix)
INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}/bin"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Platform detection
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

case "$UNAME_S" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        if [ "$UNAME_M" = "arm64" ]; then
            PLATFORM="macarm"
        else
            PLATFORM="mac"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="win64"
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
    echo "Attempting binary download from USGS GitHub releases..."

    # Discover latest release tag via GitHub API (with retry for rate limits)
    # Try curl first, fall back to wget for HPC systems with incompatible curl
    LATEST_TAG=""
    _api_url="https://api.github.com/repos/MODFLOW-ORG/modflow6/releases/latest"
    for attempt in 1 2 3; do
        _api_json=""
        if command -v curl >/dev/null 2>&1; then
            _api_json=$(curl -fsSL --retry 2 \
                -H "Accept: application/vnd.github+json" \
                "$_api_url" 2>/dev/null) || true
        fi
        if [ -z "$_api_json" ] && command -v wget >/dev/null 2>&1; then
            _api_json=$(wget -qO- --header="Accept: application/vnd.github+json" \
                "$_api_url" 2>/dev/null) || true
        fi
        if [ -n "$_api_json" ]; then
            LATEST_TAG=$(echo "$_api_json" \
                | python3 -c "import sys, json; print(json.load(sys.stdin).get('tag_name', ''))" 2>/dev/null)
        fi
        [ -n "$LATEST_TAG" ] && break
        sleep 2
    done

    if [ -z "$LATEST_TAG" ]; then
        echo "WARNING: Could not determine latest release tag, trying default"
        LATEST_TAG="6.5.0"
    fi

    echo "Latest release: $LATEST_TAG"

    # Construct download URL - try multiple naming conventions
    # MODFLOW uses patterns like mf6.5.0_linux.zip or mf6.6.0_linux.zip
    DOWNLOAD_URLS=(
        "https://github.com/MODFLOW-ORG/modflow6/releases/download/${LATEST_TAG}/mf${LATEST_TAG}_${PLATFORM}.zip"
        "https://github.com/MODFLOW-ORG/modflow6/releases/download/${LATEST_TAG}/modflow6_${PLATFORM}.zip"
    )

    TMPZIP=$(mktemp /tmp/modflow6_XXXXXX.zip)
    TMPDIR=$(mktemp -d /tmp/modflow6_extract_XXXXXX)

    DOWNLOAD_OK=false
    for DOWNLOAD_URL in "${DOWNLOAD_URLS[@]}"; do
        echo "Trying: $DOWNLOAD_URL"
        # Try curl first, fall back to wget for HPC compatibility
        if command -v curl >/dev/null 2>&1; then
            curl -fSL --retry 3 --retry-delay 5 -o "$TMPZIP" "$DOWNLOAD_URL" 2>/dev/null || true
        fi
        if [ ! -s "$TMPZIP" ] && command -v wget >/dev/null 2>&1; then
            wget -q -O "$TMPZIP" "$DOWNLOAD_URL" 2>/dev/null || true
        fi
        if [ -s "$TMPZIP" ]; then
            DOWNLOAD_OK=true
            break
        fi
    done

    if [ "$DOWNLOAD_OK" = "true" ]; then
        # Verify it's actually a zip file
        if file "$TMPZIP" | grep -q "Zip"; then
            unzip -qo "$TMPZIP" -d "$TMPDIR"

            # Find mf6 binary in extracted files
            MF6_BIN=$(find "$TMPDIR" -name "mf6" -o -name "mf6.exe" | head -1)

            if [ -n "$MF6_BIN" ]; then
                cp "$MF6_BIN" "${INSTALL_DIR}/bin/mf6"
                chmod +x "${INSTALL_DIR}/bin/mf6"

                # Verify the binary actually runs (catches glibc mismatch
                # on HPC where pre-compiled binaries target newer glibc)
                if "${INSTALL_DIR}/bin/mf6" --version >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                    "${INSTALL_DIR}/bin/mf6" --version 2>/dev/null || true

                    # Also copy zbud6 if available
                    ZBUD_BIN=$(find "$TMPDIR" -name "zbud6" -o -name "zbud6.exe" | head -1)
                    if [ -n "$ZBUD_BIN" ]; then
                        cp "$ZBUD_BIN" "${INSTALL_DIR}/bin/zbud6"
                        chmod +x "${INSTALL_DIR}/bin/zbud6"
                    fi
                else
                    echo "WARNING: Downloaded mf6 binary cannot run on this system (glibc mismatch?)"
                    echo "  Will fall back to source build"
                    rm -f "${INSTALL_DIR}/bin/mf6"
                fi
            else
                echo "WARNING: mf6 binary not found in downloaded archive"
            fi
        else
            echo "WARNING: Downloaded file is not a valid zip archive"
        fi
    else
        echo "WARNING: Download failed or empty file"
    fi

    # Cleanup
    rm -f "$TMPZIP"
    rm -rf "$TMPDIR"
fi

# === Tier 2: Build from source with meson ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Binary download failed, attempting source build with meson..."

    # Check for gfortran
    if ! command -v gfortran >/dev/null 2>&1; then
        echo "ERROR: gfortran not found. Install with: module load gcc (HPC) or brew install gcc (macOS)"
        exit 1
    fi

    # Auto-install meson and ninja via pip if in a venv/conda env
    for tool in meson ninja; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "$tool not found, attempting pip install..."
            if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
                pip install --quiet $tool 2>/dev/null || pip3 install --quiet $tool 2>/dev/null
            fi
            if ! command -v $tool >/dev/null 2>&1; then
                echo "ERROR: $tool not found and could not be installed."
                echo "  Install with: pip install meson ninja"
                exit 1
            fi
        fi
    done

    # Clone if needed (CIME-style: repo is cloned into install_dir)
    if [ -f "meson.build" ]; then
        echo "Already in MODFLOW 6 source directory"
        SRC_DIR="."
    elif [ -d "modflow6" ]; then
        SRC_DIR="modflow6"
    else
        echo "Cloning MODFLOW 6 source..."
        git clone --depth 1 https://github.com/MODFLOW-ORG/modflow6.git modflow6_src
        SRC_DIR="modflow6_src"
    fi

    cd "$SRC_DIR"

    echo "Configuring meson build..."
    meson setup builddir -Ddebug=false --prefix="${INSTALL_DIR}"

    echo "Compiling..."
    meson compile -C builddir

    if [ $? -ne 0 ]; then
        echo "ERROR: Meson build failed"
        exit 1
    fi

    # Copy binary
    MF6_BUILT=$(find builddir -name "mf6" -type f | head -1)
    if [ -n "$MF6_BUILT" ]; then
        cp "$MF6_BUILT" "${INSTALL_DIR}/bin/mf6"
        chmod +x "${INSTALL_DIR}/bin/mf6"
    else
        echo "ERROR: mf6 binary not found after build"
        exit 1
    fi

    cd ..
fi

# === Verify installation ===
if [ -f "${INSTALL_DIR}/bin/mf6" ]; then
    echo ""
    echo "=== MODFLOW 6 Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/mf6"
    "${INSTALL_DIR}/bin/mf6" --version 2>/dev/null || echo "(version check not supported on this build)"
else
    echo "ERROR: mf6 not found at ${INSTALL_DIR}/bin/mf6"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/mf6'],
            'check_type': 'exists'
        },
        'order': 25  # After CLM (20)
    }
