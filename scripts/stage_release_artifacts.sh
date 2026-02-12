#!/bin/bash
#
# Stage SYMFLUENCE release artifacts for npm distribution
#
# Creates a standardized directory structure:
#   symfluence-tools/
#     bin/          - Executables (summa, mizuroute, fuse, ngen, taudem tools)
#     share/        - Shared data files (if any)
#     LICENSES/     - License files from all tools
#     toolchain.json - Build metadata
#
# Usage:
#   ./scripts/stage_release_artifacts.sh <platform> <installs_dir> <output_dir>
#
# Example:
#   ./scripts/stage_release_artifacts.sh \
#     linux-x86_64 \
#     $SYMFLUENCE_DATA/installs \
#     ./release

set -e

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC='';
fi

print_error()   { echo -e "${RED}Error:${NC} $1" >&2; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

# Parse arguments
PLATFORM="${1:-}"
INSTALLS_DIR="${2:-}"
OUTPUT_DIR="${3:-}"

if [ -z "$PLATFORM" ] || [ -z "$INSTALLS_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    cat >&2 <<EOF
Usage: $0 <platform> <installs_dir> <output_dir>

Arguments:
  platform       Platform identifier (e.g., linux-x86_64, macos-arm64)
  installs_dir   Directory containing tool installations
  output_dir     Directory to stage artifacts (will contain symfluence-tools/)

Example:
  $0 linux-x86_64 \$SYMFLUENCE_DATA/installs ./release
EOF
    exit 1
fi

# Resolve absolute paths
INSTALLS_DIR="$(realpath "$INSTALLS_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

if [ ! -d "$INSTALLS_DIR" ]; then
    print_error "Installation directory not found: $INSTALLS_DIR"
    exit 1
fi

print_info "Staging SYMFLUENCE tools for $PLATFORM"
print_info "Source: $INSTALLS_DIR"
print_info "Output: $OUTPUT_DIR"

# Create staging directory
STAGE_DIR="$OUTPUT_DIR/symfluence-tools"
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"

# Create standard structure
mkdir -p bin share LICENSES

print_info "Created staging structure:"
tree -L 1 . 2>/dev/null || ls -la

# Counter for staged files
STAGED_COUNT=0

# Helper function to stage a binary
stage_binary() {
    local src="$1"
    local dest_name="$2"
    local tool_name="$3"

    if [ -f "$src" ]; then
        cp "$src" "bin/$dest_name"
        chmod +x "bin/$dest_name"
        print_success "Staged $tool_name → bin/$dest_name"
        STAGED_COUNT=$((STAGED_COUNT + 1))
        return 0
    else
        print_warning "Not found: $src"
        return 1
    fi
}

# Helper function to stage a license
stage_license() {
    local src_dir="$1"
    local tool_name="$2"

    for license_file in LICENSE LICENSE.txt LICENSE.md COPYING; do
        if [ -f "$src_dir/$license_file" ]; then
            cp "$src_dir/$license_file" "LICENSES/LICENSE-$tool_name"
            print_success "Staged license for $tool_name"
            return 0
        fi
    done

    print_warning "No license found for $tool_name in $src_dir"
    return 0  # Don't fail the build for missing licenses
}

# ============================================================================
# Stage SUMMA
# ============================================================================
print_info "Staging SUMMA..."

SUMMA_DIR="$INSTALLS_DIR/summa"
if [ -d "$SUMMA_DIR" ]; then
    # Try summa.exe first, then summa_sundials.exe
    if stage_binary "$SUMMA_DIR/bin/summa.exe" "summa" "SUMMA"; then
        :
    elif stage_binary "$SUMMA_DIR/bin/summa_sundials.exe" "summa" "SUMMA"; then
        :
    else
        print_warning "SUMMA binary not found"
    fi

    stage_license "$SUMMA_DIR" "SUMMA"
else
    print_warning "SUMMA not installed"
fi

# ============================================================================
# Stage mizuRoute
# ============================================================================
print_info "Staging mizuRoute..."

MIZU_DIR="$INSTALLS_DIR/mizuRoute"
if [ -d "$MIZU_DIR" ]; then
    # Try mizuRoute.exe first (Windows), then mizuRoute (Unix)
    if [ -f "$MIZU_DIR/route/bin/mizuRoute.exe" ]; then
        stage_binary "$MIZU_DIR/route/bin/mizuRoute.exe" "mizuroute" "mizuRoute"
    elif [ -f "$MIZU_DIR/route/bin/mizuRoute" ]; then
        stage_binary "$MIZU_DIR/route/bin/mizuRoute" "mizuroute" "mizuRoute"
    else
        print_warning "mizuRoute binary not found in $MIZU_DIR/route/bin"
    fi
    stage_license "$MIZU_DIR" "mizuRoute"
else
    print_warning "mizuRoute not installed"
fi

# ============================================================================
# Stage FUSE
# ============================================================================
print_info "Staging FUSE..."

FUSE_DIR="$INSTALLS_DIR/fuse"
if [ -d "$FUSE_DIR" ]; then
    if [ -f "$FUSE_DIR/bin/fuse.exe" ]; then
        stage_binary "$FUSE_DIR/bin/fuse.exe" "fuse" "FUSE"
    elif [ -f "$FUSE_DIR/bin/fuse" ]; then
        stage_binary "$FUSE_DIR/bin/fuse" "fuse" "FUSE"
    else
        print_warning "FUSE binary not found in $FUSE_DIR/bin"
    fi
    stage_license "$FUSE_DIR" "FUSE"
else
    print_warning "FUSE not installed"
fi

# ============================================================================
# Stage NGEN
# ============================================================================
print_info "Staging NGEN..."

NGEN_DIR="$INSTALLS_DIR/ngen"
if [ -d "$NGEN_DIR" ]; then
    if [ -f "$NGEN_DIR/cmake_build/ngen" ]; then
        stage_binary "$NGEN_DIR/cmake_build/ngen" "ngen" "NGEN"
    else
        print_warning "NGEN binary not found in $NGEN_DIR/cmake_build"
    fi
    stage_license "$NGEN_DIR" "NGEN"
else
    print_warning "NGEN not installed"
fi

# ============================================================================
# Stage HYPE
# ============================================================================
print_info "Staging HYPE..."

HYPE_DIR="$INSTALLS_DIR/hype"
if [ -d "$HYPE_DIR" ]; then
    # Try to stage HYPE binary (optional, may not be built on all platforms)
    stage_binary "$HYPE_DIR/bin/hype" "hype" "HYPE" || print_warning "HYPE binary not found (may not be built yet)"
    stage_license "$HYPE_DIR" "HYPE"
else
    print_warning "HYPE not installed"
fi

# ============================================================================
# Stage TauDEM
# ============================================================================
print_info "Staging TauDEM..."

TAUDEM_DIR="$INSTALLS_DIR/TauDEM"
if [ -d "$TAUDEM_DIR/bin" ]; then
    # TauDEM has multiple executables
    TAUDEM_COUNT=0
    for exe in "$TAUDEM_DIR/bin"/*; do
        if [ -x "$exe" ]; then
            exe_name="$(basename "$exe")"
            if stage_binary "$exe" "$exe_name" "TauDEM"; then
                TAUDEM_COUNT=$((TAUDEM_COUNT + 1))
            fi
        fi
    done

    if [ $TAUDEM_COUNT -gt 0 ]; then
        print_success "Staged $TAUDEM_COUNT TauDEM executables"
    fi

    stage_license "$TAUDEM_DIR" "TauDEM"
else
    print_warning "TauDEM not installed"
fi

# ============================================================================
# Stage MESH
# ============================================================================
print_info "Staging MESH..."

MESH_DIR="$INSTALLS_DIR/mesh"
if [ -d "$MESH_DIR" ]; then
    if [ -f "$MESH_DIR/bin/mesh.exe" ]; then
        stage_binary "$MESH_DIR/bin/mesh.exe" "mesh" "MESH"
    elif [ -f "$MESH_DIR/bin/mesh" ]; then
        stage_binary "$MESH_DIR/bin/mesh" "mesh" "MESH"
    else
        print_warning "MESH binary not found (may not be built yet)"
    fi
    stage_license "$MESH_DIR" "MESH"
else
    print_warning "MESH not installed"
fi

# ============================================================================
# Stage WMFire
# ============================================================================
print_info "Staging WMFire..."

WMFIRE_DIR="$INSTALLS_DIR/wmfire"
if [ -d "$WMFIRE_DIR" ]; then
    # WMFire is a shared library, but we'll stage it for completeness
    if [ "$(uname)" = "Darwin" ] && [ -f "$WMFIRE_DIR/lib/libwmfire.dylib" ]; then
        stage_binary "$WMFIRE_DIR/lib/libwmfire.dylib" "libwmfire.dylib" "WMFire"
    elif [ -f "$WMFIRE_DIR/lib/libwmfire.so" ]; then
        stage_binary "$WMFIRE_DIR/lib/libwmfire.so" "libwmfire.so" "WMFire"
    else
        print_warning "WMFire binary not found (may not be built yet)"
    fi
    stage_license "$WMFIRE_DIR" "WMFire"
else
    print_warning "WMFire not installed"
fi

# ============================================================================
# Stage RHESSys
# ============================================================================
print_info "Staging RHESSys..."

RHESSYS_DIR="$INSTALLS_DIR/rhessys"
if [ -d "$RHESSYS_DIR" ]; then
    if [ -f "$RHESSYS_DIR/bin/rhessys" ]; then
        stage_binary "$RHESSYS_DIR/bin/rhessys" "rhessys" "RHESSys"
    else
        print_warning "RHESSys binary not found (may not be built yet)"
    fi
    stage_license "$RHESSYS_DIR" "RHESSys"
else
    print_warning "RHESSys not installed"
fi

# ============================================================================
# Stage additional tools (if needed in future)
# ============================================================================

# GIStool (script only)
GISTOOL_DIR="$INSTALLS_DIR/gistool"
if [ -d "$GISTOOL_DIR" ] && [ -f "$GISTOOL_DIR/extract-gis.sh" ]; then
    print_info "Staging GIStool..."
    stage_binary "$GISTOOL_DIR/extract-gis.sh" "extract-gis.sh" "GIStool"
    stage_license "$GISTOOL_DIR" "GIStool"
fi

# Datatool (script only)
DATATOOL_DIR="$INSTALLS_DIR/datatool"
if [ -d "$DATATOOL_DIR" ] && [ -f "$DATATOOL_DIR/extract-dataset.sh" ]; then
    print_info "Staging Datatool..."
    stage_binary "$DATATOOL_DIR/extract-dataset.sh" "extract-dataset.sh" "Datatool"
    stage_license "$DATATOOL_DIR" "Datatool"
fi

# ============================================================================
# Copy toolchain metadata
# ============================================================================
print_info "Copying toolchain metadata..."

TOOLCHAIN_SRC="$INSTALLS_DIR/toolchain.json"
if [ -f "$TOOLCHAIN_SRC" ]; then
    cp "$TOOLCHAIN_SRC" .
    print_success "Copied toolchain.json"
else
    print_error "toolchain.json not found at $TOOLCHAIN_SRC"
    print_error "Run scripts/generate_toolchain.sh first"
    exit 1
fi

# ============================================================================
# Create README
# ============================================================================
print_info "Creating README..."

cat > README.md <<'EOF'
# SYMFLUENCE Tools

Pre-built hydrological modeling tools for SYMFLUENCE.

## Contents

This archive contains compiled binaries for:
- **SUMMA**: Structure for Unifying Multiple Modeling Alternatives
- **mizuRoute**: River network routing model
- **FUSE**: Framework for Understanding Structural Errors
- **NGEN**: NextGen National Water Model Framework
- **HYPE**: Hydrological Predictions for the Environment
- **TauDEM**: Terrain Analysis Using Digital Elevation Models

## Installation

### Manual Installation

```bash
# Extract archive
tar -xzf symfluence-tools-*.tar.gz

# Add to PATH
export PATH="$PWD/symfluence-tools/bin:$PATH"

# Verify
summa --version
```

### npm Installation (Recommended)

```bash
npm install -g symfluence
```

## System Requirements

See `toolchain.json` for build details.

### Linux
- glibc ≥ 2.39
- NetCDF ≥ 4.8
- HDF5 ≥ 1.10

### macOS
- macOS 12+ (Monterey)
- Homebrew: `netcdf netcdf-fortran hdf5`

## Toolchain

Build metadata is available in `toolchain.json`:
- Tool versions (commit hashes)
- Compiler versions
- Library versions
- Build timestamp

## Licenses

See `LICENSES/` directory for individual tool licenses.

## Support

- Issues: https://github.com/DarriEy/SYMFLUENCE/issues
- Documentation: https://github.com/DarriEy/SYMFLUENCE

---

🤖 Generated with SYMFLUENCE release automation
EOF

print_success "Created README.md"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Staging Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "Platform: $PLATFORM"
print_success "Staged binaries: $STAGED_COUNT"
print_success "Output directory: $STAGE_DIR"
echo ""
print_info "Staged binaries:"
ls -lh bin/ | tail -n +2 | awk '{printf "  %s (%s)\n", $9, $5}'
echo ""
print_info "Licenses:"
ls -1 LICENSES/ | sed 's/^/  /'
echo ""
print_info "Directory structure:"
tree -L 2 . 2>/dev/null || find . -maxdepth 2 -type f -o -type d
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $STAGED_COUNT -eq 0 ]; then
    print_error "No binaries were staged! Check installation directory."
    exit 1
fi

print_success "Staging complete!"
