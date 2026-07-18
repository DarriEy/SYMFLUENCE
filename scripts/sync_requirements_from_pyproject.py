#!/usr/bin/env python3
"""
Generate requirements.txt from pyproject.toml project dependencies.

The `jax` extra is included by default: the JAX-native models (jHBV, jSACSMA,
jXAJ, jHECHMS, jTOPMODEL, jSnow17) are first-class registry members, and an
environment without them registers five fewer models with no error — the
install looks complete and silently is not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Optional-dependency groups folded into requirements.txt by default.
DEFAULT_EXTRAS = ("jax", "ml")


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
