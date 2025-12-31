#!/bin/bash
#
# Create release tarball for SYMFLUENCE tools
#
# Creates:
#   - symfluence-tools-<version>-<platform>.tar.gz
#   - symfluence-tools-<version>-<platform>.tar.gz.sha256
#
# Usage:
#   ./scripts/create_release_tarball.sh <version> <platform> <staged_dir> <output_dir>
#
# Example:
#   ./scripts/create_release_tarball.sh \
#     v0.7.0 \
#     linux-x86_64 \
#     ./release/symfluence-tools \
#     ./release

set -e

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN=''; BLUE=''; NC='';
fi

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }

# Parse arguments
VERSION="${1:-}"
PLATFORM="${2:-}"
STAGED_DIR="${3:-}"
OUTPUT_DIR="${4:-}"

if [ -z "$VERSION" ] || [ -z "$PLATFORM" ] || [ -z "$STAGED_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    cat >&2 <<EOF
Usage: $0 <version> <platform> <staged_dir> <output_dir>

Arguments:
  version        Version tag (e.g., v0.7.0)
  platform       Platform identifier (e.g., linux-x86_64)
  staged_dir     Directory containing staged symfluence-tools/
  output_dir     Directory to write tarball

Example:
  $0 v0.7.0 linux-x86_64 ./release/symfluence-tools ./release
EOF
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Tarball name
TARBALL_NAME="symfluence-tools-${VERSION}-${PLATFORM}.tar.gz"
TARBALL_PATH="$OUTPUT_DIR/$TARBALL_NAME"

print_info "Creating release tarball..."
print_info "Version: $VERSION"
print_info "Platform: $PLATFORM"
print_info "Source: $STAGED_DIR"
print_info "Output: $TARBALL_PATH"

# Create tarball
cd "$(dirname "$STAGED_DIR")"
STAGED_BASENAME="$(basename "$STAGED_DIR")"

print_info "Compressing..."
tar -czf "$TARBALL_PATH" "$STAGED_BASENAME"

TARBALL_SIZE="$(du -h "$TARBALL_PATH" | cut -f1)"
print_success "Created tarball: $TARBALL_NAME ($TARBALL_SIZE)"

# Generate checksum
print_info "Generating SHA256 checksum..."

cd "$OUTPUT_DIR"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$TARBALL_NAME" > "${TARBALL_NAME}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$TARBALL_NAME" > "${TARBALL_NAME}.sha256"
else
    echo "Warning: No SHA256 tool found (sha256sum or shasum)" >&2
    exit 1
fi

CHECKSUM="$(cat "${TARBALL_NAME}.sha256")"
print_success "Generated checksum: ${TARBALL_NAME}.sha256"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Release Artifact Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "Version: $VERSION"
print_success "Platform: $PLATFORM"
print_success "Tarball: $TARBALL_NAME ($TARBALL_SIZE)"
print_success "Checksum: $CHECKSUM"
echo ""
print_info "Files created:"
ls -lh "$OUTPUT_DIR/$TARBALL_NAME"*
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
