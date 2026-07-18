#!/bin/bash
# Package curated Paper 3 provenance records for GitHub Releases and Zenodo.
# Usage: create_paper_provenance_bundle.sh SOURCE_DIR vX.Y.Z [OUTPUT_DIR]

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 SOURCE_DIR vX.Y.Z [OUTPUT_DIR]" >&2
    exit 2
fi

SOURCE_DIR="$1"
RELEASE_TAG="$2"
OUTPUT_DIR="${3:-dist/paper-provenance}"

if ! echo "$RELEASE_TAG" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: release tag must have the form vX.Y.Z" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$(cd "$SOURCE_DIR" 2>/dev/null && pwd)" || {
    echo "Error: source directory does not exist: $1" >&2
    exit 1
}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

PACKAGE_VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' \
    "$REPO_ROOT/src/symfluence/symfluence_version.py")"
if [ "$RELEASE_TAG" != "v$PACKAGE_VERSION" ]; then
    echo "Error: tag $RELEASE_TAG does not match package version v$PACKAGE_VERSION" >&2
    exit 1
fi

if [ ! -f "$SOURCE_DIR/COVERAGE.md" ]; then
    echo "Error: source directory must contain COVERAGE.md" >&2
    exit 1
fi

ARCHIVE_BASENAME="symfluence-paper3-provenance-$RELEASE_TAG"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_BASENAME.tar.gz"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"
if [ -e "$ARCHIVE_PATH" ] || [ -e "$CHECKSUM_PATH" ]; then
    echo "Error: output already exists for $RELEASE_TAG" >&2
    exit 1
fi

STAGE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/symfluence-provenance.XXXXXX")"
trap 'rm -rf "$STAGE_PARENT"' EXIT
STAGE_DIR="$STAGE_PARENT/$ARCHIVE_BASENAME"
mkdir -p "$STAGE_DIR"
cp -R "$SOURCE_DIR"/. "$STAGE_DIR"/
rm -f "$STAGE_DIR/MANIFEST.json"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
export STAGE_DIR RELEASE_TAG PACKAGE_VERSION GIT_COMMIT
python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["STAGE_DIR"])
files = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
    })

manifest = {
    "schema_version": 1,
    "artifact": "SYMFLUENCE Paper 3 provenance",
    "release_tag": os.environ["RELEASE_TAG"],
    "symfluence_version": os.environ["PACKAGE_VERSION"],
    "git_commit": os.environ["GIT_COMMIT"],
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "files": files,
}
(root / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

tar -C "$STAGE_PARENT" -czf "$ARCHIVE_PATH" "$ARCHIVE_BASENAME"
cd "$OUTPUT_DIR"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$(basename "$ARCHIVE_PATH")" > "$(basename "$CHECKSUM_PATH")"
else
    shasum -a 256 "$(basename "$ARCHIVE_PATH")" > "$(basename "$CHECKSUM_PATH")"
fi

echo "Created $ARCHIVE_PATH"
echo "Created $CHECKSUM_PATH"
