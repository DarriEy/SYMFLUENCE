#!/bin/bash
# Atomically bump the SYMFLUENCE version across all tracked files.
#
# Single source of truth: src/symfluence/symfluence_version.py
# Files kept in sync: npm/package.json
#
# Usage: scripts/bump_version.sh <new_version>
# Example: scripts/bump_version.sh 0.8.3

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.8.3"
    exit 1
fi

NEW_VERSION="$1"

if ! echo "$NEW_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: version must be X.Y.Z format (got: $NEW_VERSION)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OLD_VERSION=$(grep '^__version__' "$REPO_ROOT/src/symfluence/symfluence_version.py" | sed 's/.*"\([0-9.]*\)".*/\1/')

if [ "$OLD_VERSION" = "$NEW_VERSION" ]; then
    echo "Version is already $NEW_VERSION — nothing to do."
    exit 0
fi

echo "Bumping version: $OLD_VERSION -> $NEW_VERSION"

# Replace only the first `"version":` line in the JSON file. We use awk for
# portability — BSD sed on macOS does not support the GNU `0,/pattern/`
# address range.

replace_first_version() {
    local file="$1"
    local new_ver="$2"
    awk -v v="$new_ver" '
        !done && /"version":/ {
            sub(/"version": "[^"]*"/, "\"version\": \"" v "\"")
            done = 1
        }
        { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# 1. src/symfluence/symfluence_version.py (source of truth)
#    Portable in-place sed: -i.bak then delete .bak (works on BSD and GNU sed).
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" \
    "$REPO_ROOT/src/symfluence/symfluence_version.py"
rm -f "$REPO_ROOT/src/symfluence/symfluence_version.py.bak"

# 2. npm/package.json
replace_first_version "$REPO_ROOT/npm/package.json" "$NEW_VERSION"

echo ""
# Verify both files are in sync
"$SCRIPT_DIR/check_version_sync.sh"

echo ""
echo "Next steps:"
echo "  git add -u && git commit -m 'chore: bump version to $NEW_VERSION'"
