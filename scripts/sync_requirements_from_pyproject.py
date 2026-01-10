#!/usr/bin/env python3
"""
Generate requirements.txt from pyproject.toml project dependencies.
"""

from __future__ import annotations

from pathlib import Path


def load_pyproject(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
        import tomli as tomllib  # type: ignore

    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"

    data = load_pyproject(pyproject_path)
    dependencies = data.get("project", {}).get("dependencies", [])
    if not dependencies:
        raise SystemExit("No project.dependencies found in pyproject.toml")

    lines = [
        "# This file is generated from pyproject.toml; do not edit by hand.",
        "# Run: python scripts/sync_requirements_from_pyproject.py",
        "",
    ]
    lines.extend(dependencies)
    lines.append("")

    requirements_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {requirements_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
