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

# Extract the version from the publish-side manifest (the package.json that
# is actually used to npm-publish the binaries — see the publish-npm job in
# .github/workflows/release-binaries.yml, which runs with working-directory
# npm/). Do not add a consumer-side lockfile to this check: pinning the
# about-to-be-published version in a lockfile creates a chicken-and-egg
# failure mode where the release workflow cannot publish version N until the
# lockfile resolves N, but the lockfile cannot resolve N until N has been
# published. (This bit Release Binaries CI for three weeks in April 2026.)
NPM_VERSION=$(grep '"version":' "$REPO_ROOT/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')

echo "Checking version synchronization..."
echo "  Source of truth:"
echo "    symfluence_version.py: $PYTHON_VERSION"
echo "  Must match:"
echo "    npm/package.json:      $NPM_VERSION"
echo ""

ERRORS=0

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
