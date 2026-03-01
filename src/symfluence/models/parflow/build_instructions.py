# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from ParFlow GitHub releases.
Tier 2 (fallback): Build from source using CMake.

ParFlow is a parallel integrated hydrologic model. Pre-compiled
binaries are available for Linux and macOS.
"""

from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('parflow')
def get_parflow_build_instructions():
    """
    Get ParFlow build/install instructions.

    Downloads pre-compiled ParFlow binary from GitHub releases.
    Falls back to CMake source build if download fails.

    Returns:
        Dictionary with complete build configuration for ParFlow.
    """
    return {
        'description': 'ParFlow (Parallel Integrated Hydrologic Model)',
        'config_path_key': 'PARFLOW_INSTALL_PATH',
        'config_exe_key': 'PARFLOW_EXE',
        'default_path_suffix': 'installs/parflow/bin',
        'default_exe': 'parflow',
        'repository': 'https://github.com/parflow/parflow.git',
        'branch': 'master',
        'install_dir': 'parflow',
        'build_commands': [
            r'''
# ParFlow Install Script for SYMFLUENCE
# Tier 1: Download pre-compiled binary from GitHub releases
# Tier 2: Build from source with CMake (fallback)

set -e

echo "=== ParFlow Installation Starting ==="

# Resolve INSTALL_DIR to an absolute path BEFORE any cd commands.
# This is critical — CMake's CMAKE_INSTALL_PREFIX must be absolute,
# otherwise `make install` targets the build dir, not the real prefix.
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
    echo "Attempting binary download from ParFlow GitHub releases..."

    # Discover latest release tag via GitHub API (try curl then wget)
    _api_url="https://api.github.com/repos/parflow/parflow/releases/latest"
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
        echo "WARNING: Could not determine latest release tag, trying default"
        LATEST_TAG="v3.12.0"
    fi

    echo "Latest release: $LATEST_TAG"

    # ParFlow releases may have platform-specific archives
    DOWNLOAD_URL="https://github.com/parflow/parflow/releases/download/${LATEST_TAG}/parflow-${LATEST_TAG}-${PLATFORM}.tar.gz"
    echo "Download URL: $DOWNLOAD_URL"

    TMPTAR=$(mktemp /tmp/parflow_XXXXXX.tar.gz)
    TMPEXTRACT=$(mktemp -d /tmp/parflow_extract_XXXXXX)

    # Try curl first, fall back to wget
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ ! -s "$TMPTAR" ] && command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ -s "$TMPTAR" ]; then
        # Verify it's actually a tar.gz file
        if file "$TMPTAR" | grep -q "gzip\|tar"; then
            tar xzf "$TMPTAR" -C "$TMPEXTRACT" 2>/dev/null || true

            # Find parflow binary
            PF_BIN=$(find "$TMPEXTRACT" -name "parflow" -type f 2>/dev/null | head -1)

            if [ -n "$PF_BIN" ]; then
                cp "$PF_BIN" "${INSTALL_DIR}/bin/parflow"
                chmod +x "${INSTALL_DIR}/bin/parflow"

                # Verify the binary actually runs (catches glibc mismatch
                # on HPC where pre-compiled binaries target newer glibc)
                if "${INSTALL_DIR}/bin/parflow" --version >/dev/null 2>&1 || \
                   "${INSTALL_DIR}/bin/parflow" -v >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                else
                    echo "WARNING: Downloaded parflow binary cannot run on this system (glibc mismatch?)"
                    echo "  Will fall back to source build"
                    rm -f "${INSTALL_DIR}/bin/parflow"
                fi
            else
                echo "WARNING: parflow binary not found in downloaded archive"
            fi
        else
            echo "WARNING: Downloaded file is not a valid archive"
        fi
    else
        echo "WARNING: Download failed or empty file"
    fi

    # Cleanup
    rm -f "$TMPTAR"
    rm -rf "$TMPEXTRACT"
fi

# === Tier 2: Build from source with CMake ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Binary download failed, attempting source build with CMake..."

    # Check dependencies
    for tool in cmake make; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "ERROR: $tool not found. Install with: brew install cmake (or apt install cmake)"
            exit 1
        fi
    done

    # Clone source into a temp directory (not inside INSTALL_DIR)
    BUILD_TMPDIR=$(mktemp -d /tmp/parflow_build_XXXXXX)
    echo "Build directory: ${BUILD_TMPDIR}"

    echo "Cloning ParFlow source..."
    git clone --depth 1 https://github.com/parflow/parflow.git "${BUILD_TMPDIR}/parflow_src"

    mkdir -p "${BUILD_TMPDIR}/parflow_src/build"

    echo "Configuring CMake build..."
    # ParFlow source has several C99/C11 compliance issues that modern
    # Clang (16+) treats as hard errors:
    #   - int-to-pointer conversions in AMPS seq layer tests (test6.c)
    #   - implicit function declarations (seepage.c missing <string.h>)
    # We suppress these as warnings to allow the build to succeed.
    cmake -S "${BUILD_TMPDIR}/parflow_src" \
          -B "${BUILD_TMPDIR}/parflow_src/build" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DPARFLOW_AMPS_LAYER=seq \
        -DPARFLOW_ENABLE_TIMING=FALSE \
        -DPARFLOW_HAVE_CLM=OFF \
        -DCMAKE_C_FLAGS="-Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"

    echo "Compiling..."
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "${BUILD_TMPDIR}/parflow_src/build" -j "${NCPU}"

    echo "Installing to ${INSTALL_DIR}..."
    cmake --install "${BUILD_TMPDIR}/parflow_src/build"

    # Cleanup build artifacts
    rm -rf "${BUILD_TMPDIR}"
    echo "Build artifacts cleaned up"
fi

# === Install Python dependencies ===
echo ""
echo "Installing Python dependencies (pftools, parflowio)..."
pip install pftools parflowio 2>/dev/null || \
    echo "WARNING: Could not install pftools/parflowio (optional, manual reader available)"

# === Verify installation ===
if [ -f "${INSTALL_DIR}/bin/parflow" ]; then
    echo ""
    echo "=== ParFlow Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/parflow"
    "${INSTALL_DIR}/bin/parflow" -v 2>/dev/null || echo "(version check not supported)"
else
    echo "ERROR: parflow not found at ${INSTALL_DIR}/bin/parflow"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'make'],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/parflow'],
            'check_type': 'exists'
        },
        'order': 26,  # After MODFLOW (25)
        'optional': True,  # Not installed by default with --install
    }
