#!/bin/bash
# Version synchronization validation script
# Ensures src/symfluence/symfluence_version.py and tools/npm/package.json versions match

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract versions (strip comments and whitespace)
PYTHON_VERSION=$(grep '^__version__' "$REPO_ROOT/src/symfluence/symfluence_version.py" | sed 's/.*"\([0-9.]*\)".*/\1/')
NPM_VERSION=$(grep '"version":' "$REPO_ROOT/tools/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')

echo "Checking version synchronization..."
echo "  symfluence_version.py: $PYTHON_VERSION"
echo "  tools/npm/package.json: $NPM_VERSION"
echo ""

if [ "$PYTHON_VERSION" != "$NPM_VERSION" ]; then
    echo "❌ VERSION MISMATCH DETECTED!"
    echo ""
echo "The versions in symfluence_version.py and tools/npm/package.json must match."
    echo "Please update both files to the same version before releasing."
    echo ""
    echo "To fix:"
    echo "  1. Update src/symfluence/symfluence_version.py to __version__ = \"$NPM_VERSION\""
    echo "  OR"
echo "  2. Update tools/npm/package.json (line 3) to \"version\": \"$PYTHON_VERSION\""
    echo ""
    exit 1
else
    echo "✓ Versions are synchronized: $PYTHON_VERSION"
    exit 0
fi
