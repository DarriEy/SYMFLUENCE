#!/usr/bin/env python3
"""
Generate requirements.txt from pyproject.toml project dependencies.

This is a compatibility export for tools that cannot consume ``pyproject.toml``
or ``uv.lock``. It contains the base runtime dependencies by default; optional
feature groups must be requested explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Optional-dependency groups folded into requirements.txt by default.
DEFAULT_EXTRAS: tuple[str, ...] = ()


def load_pyproject(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        nargs="*",
        default=list(DEFAULT_EXTRAS),
        metavar="GROUP",
        help=(
            "optional-dependency groups to include "
            f"(default: {', '.join(DEFAULT_EXTRAS)}; pass --extras with no value for none)"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"

    data = load_pyproject(pyproject_path)
    project = data.get("project", {})
    dependencies: List[str] = list(project.get("dependencies", []))
    if not dependencies:
        raise SystemExit("No project.dependencies found in pyproject.toml")

    optional = project.get("optional-dependencies", {})
    lines = [
        "# This file is generated from pyproject.toml; do not edit by hand.",
        "# Run: python scripts/sync_requirements_from_pyproject.py",
        "",
        *dependencies,
    ]

    for group in args.extras:
        extra_deps = optional.get(group)
        if extra_deps is None:
            raise SystemExit(f"pyproject.toml has no optional-dependency group '{group}'")
        lines.extend(["", f"# [{group}] extra", *extra_deps])

    lines.append("")

    requirements_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {requirements_path} (extras: {', '.join(args.extras) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
