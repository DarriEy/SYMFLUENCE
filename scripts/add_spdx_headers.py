#!/usr/bin/env python3
"""Add SPDX license headers to all Python source files.

Idempotent script that adds a 2-line SPDX header + blank separator to all
.py files under src/symfluence/. Handles shebang lines, existing docstrings,
and import statements correctly.

Usage:
    python scripts/add_spdx_headers.py              # Apply headers
    python scripts/add_spdx_headers.py --dry-run    # Preview changes
    python scripts/add_spdx_headers.py --check      # Exit 1 if any file missing header
"""

import argparse
import sys
from pathlib import Path

HEADER_LINES = [
    "# SPDX-License-Identifier: GPL-3.0-or-later\n",
    "# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>\n",
]

SPDX_MARKER = "SPDX-License-Identifier"


def needs_header(content: str) -> bool:
    """Check whether file content already contains an SPDX header."""
    return SPDX_MARKER not in content


def add_header(content: str) -> str:
    """Insert SPDX header into file content, respecting shebang lines."""
    lines = content.splitlines(keepends=True)

    # If file starts with a shebang, insert after it
    if lines and lines[0].startswith("#!"):
        insert_pos = 1
        # Preserve blank line after shebang if present
        if len(lines) > 1 and lines[1].strip() == "":
            insert_pos = 2
    else:
        insert_pos = 0

    # Build header block: header lines + blank separator
    header_block = HEADER_LINES + ["\n"]

    new_lines = lines[:insert_pos] + header_block + lines[insert_pos:]
    return "".join(new_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add SPDX license headers to Python files")
    parser.add_argument("--dry-run", action="store_true", help="Preview files that would be modified")
    parser.add_argument("--check", action="store_true", help="Check mode: exit 1 if any file is missing a header")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "src" / "symfluence",
                        help="Root directory to scan (default: src/symfluence/)")
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 1

    py_files = sorted(root.rglob("*.py"))
    missing = [f for f in py_files if needs_header(f.read_text(encoding="utf-8"))]

    if args.check:
        if missing:
            print(f"SPDX header missing from {len(missing)} file(s):")
            for f in missing:
                print(f"  {f}")
            return 1
        print(f"All {len(py_files)} files have SPDX headers.")
        return 0

    if not missing:
        print(f"All {len(py_files)} files already have SPDX headers. Nothing to do.")
        return 0

    for f in missing:
        content = f.read_text(encoding="utf-8")
        new_content = add_header(content)
        if args.dry_run:
            print(f"Would add header: {f}")
        else:
            f.write_text(new_content, encoding="utf-8")
            print(f"Added header: {f}")

    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {len(missing)} of {len(py_files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
