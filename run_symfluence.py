#!/usr/bin/env python3
"""
Thin wrapper to run the SYMFLUENCE CLI directly from the repo without install.
"""

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    from symfluence.cli import main as cli_main

    return int(cli_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
