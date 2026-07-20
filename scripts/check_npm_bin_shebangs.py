#!/usr/bin/env python3
"""Guard: every npm/bin wrapper must start with a valid shebang.

npm relies on the shebang when it symlinks global bins on POSIX systems. A
mangled first line — like the escaped ``#\\!`` that shipped in 0.9.2 and broke
16 of the 22 wrappers — makes sh execute the JavaScript, which dies on the
first ``require()``. Windows is unaffected (npm .cmd shims invoke node
directly), which is why this survived on the raider but not on macOS/Linux.
"""
from __future__ import annotations

import sys
from pathlib import Path

BIN_DIR = Path(__file__).resolve().parent.parent / "npm" / "bin"


def main() -> int:
    bad: list[tuple[Path, bytes]] = []
    for f in sorted(BIN_DIR.iterdir()):
        if not f.is_file():
            continue
        first_line = f.read_bytes().split(b"\n", 1)[0]
        if not first_line.startswith(b"#!"):
            bad.append((f, first_line))
    for f, line in bad:
        rel = f.relative_to(BIN_DIR.parent.parent)
        print(f"{rel}: first line {line[:40]!r} is not a shebang")
    if bad:
        print("npm bin wrappers must start with '#!' (e.g. '#!/usr/bin/env node').")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
