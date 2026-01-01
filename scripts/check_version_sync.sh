#!/bin/bash
# Version synchronization validation script
# Ensures pyproject.toml and npm/package.json versions match

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract versions (strip comments and whitespace)
PYPROJECT_VERSION=$(grep '^version = ' "$REPO_ROOT/pyproject.toml" | sed 's/version = "\([0-9.]*\)".*/\1/')
NPM_VERSION=$(grep '"version":' "$REPO_ROOT/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')

echo "Checking version synchronization..."
echo "  pyproject.toml: $PYPROJECT_VERSION"
echo "  npm/package.json: $NPM_VERSION"
echo ""

if [ "$PYPROJECT_VERSION" != "$NPM_VERSION" ]; then
    echo "❌ VERSION MISMATCH DETECTED!"
    echo ""
    echo "The versions in pyproject.toml and npm/package.json must match."
    echo "Please update both files to the same version before releasing."
    echo ""
    echo "To fix:"
    echo "  1. Update pyproject.toml (line 10) to version = \"$NPM_VERSION\""
    echo "  OR"
    echo "  2. Update npm/package.json (line 3) to \"version\": \"$PYPROJECT_VERSION\""
    echo ""
    exit 1
else
    echo "✓ Versions are synchronized: $PYPROJECT_VERSION"
    exit 0
fi
