#!/bin/bash
# Version synchronization validation script
# Single source of truth: src/symfluence/symfluence_version.py
# All other version references must match.

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract version from single source of truth
PYTHON_VERSION=$(grep '^__version__' "$REPO_ROOT/src/symfluence/symfluence_version.py" | sed 's/.*"\([0-9.]*\)".*/\1/')

# Extract versions from other locations
TOOLS_NPM_VERSION=$(grep '"version":' "$REPO_ROOT/tools/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')
NPM_VERSION=$(grep '"version":' "$REPO_ROOT/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')

echo "Checking version synchronization..."
echo "  Source of truth:"
echo "    symfluence_version.py: $PYTHON_VERSION"
echo "  Must match:"
echo "    tools/npm/package.json: $TOOLS_NPM_VERSION"
echo "    npm/package.json:       $NPM_VERSION"
echo ""

ERRORS=0

if [ "$PYTHON_VERSION" != "$TOOLS_NPM_VERSION" ]; then
    echo "❌ tools/npm/package.json ($TOOLS_NPM_VERSION) does not match ($PYTHON_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$PYTHON_VERSION" != "$NPM_VERSION" ]; then
    echo "❌ npm/package.json ($NPM_VERSION) does not match ($PYTHON_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "❌ VERSION MISMATCH DETECTED!"
    echo ""
    echo "The single source of truth is: src/symfluence/symfluence_version.py"
    echo "Update all version references to match: $PYTHON_VERSION"
    echo ""
    exit 1
else
    echo "✓ All versions synchronized: $PYTHON_VERSION"
    exit 0
fi
