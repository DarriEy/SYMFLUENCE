#!/usr/bin/env python3
"""Preview or remove known generated SYMFLUENCE workspace artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT_LOG_GLOBS = ("*hyss*.log",)
RECURSIVE_FILE_NAMES = (".DS_Store",)
RECURSIVE_DIR_NAMES = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
GENERATED_DIRS = (
    Path("examples/paper_case_studies/plotting/08_parallel_scaling/analysis"),
    Path("examples/paper_case_studies/reproduction_logs"),
)
EXCLUDED_ROOT_DIRS = (".git", ".pixi", ".venv", "venv")


def _inside(root: Path, candidate: Path) -> bool:
    """Return whether candidate resolves below root, excluding root itself."""
    try:
        return candidate.resolve().is_relative_to(root.resolve()) and candidate.resolve() != root.resolve()
    except (OSError, RuntimeError):
        return False


def collect(root: Path) -> list[Path]:
    """Collect only explicitly recognized generated paths."""
    paths: set[Path] = set()
    for pattern in ROOT_LOG_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        generated_directories = set(directories) & set(RECURSIVE_DIR_NAMES)
        paths.update(current_path / name for name in generated_directories)
        directories[:] = [
            name for name in directories
            if name not in EXCLUDED_ROOT_DIRS and name not in RECURSIVE_DIR_NAMES
        ]
        paths.update(current_path / name for name in files if name in RECURSIVE_FILE_NAMES)
    paths.update(path for relative in GENERATED_DIRS if (path := root / relative).exists())
    return sorted((path for path in paths if _inside(root, path)), key=lambda path: (len(path.parts), str(path)), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete listed artifacts; default is a dry run")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = args.root.resolve()
    paths = collect(root)
    if not paths:
        print("Workspace is already clean.")
        return 0

    verb = "Removing" if args.apply else "Would remove"
    for path in paths:
        print(f"{verb}: {path.relative_to(root)}")
        if not args.apply:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    if not args.apply:
        print("Dry run only; pass --apply to remove these generated artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
