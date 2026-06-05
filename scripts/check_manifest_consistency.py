#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Fail CI if the three install manifests disagree on a core dependency.

Review item 16 (Tier 2): the seven install paths must stay at release quality.
``pyproject.toml`` (pip/uv/pipx), ``pixi.toml`` (pixi/conda), and
``environment.yml`` (conda) each declare the dependency stack independently, so
they drift — a package gets a new upper bound in one and not the others, or is
dropped from one entirely. That drift is exactly how an install method silently
starts resolving a different (sometimes broken) version than the others.

This guard checks, for a curated set of CORE packages that must stay consistent:

  1. **Presence** — each core package is declared in all three manifests
     (accounting for PyPI vs conda naming: netCDF4/netcdf4, torch/pytorch,
     pvlib/pvlib-python, pint_xarray/pint-xarray, SALib/salib).
  2. **Version bounds** — where both pyproject.toml and pixi.toml pin bounds,
     the lower (>=) and upper (<) bounds agree once normalised (2.0 == 2.0.0).
     environment.yml is intentionally unpinned and is checked for presence only.

Packages that are deliberately ecosystem-specific (gdal as an optional/system
extra, conda-only build tools, dev-only Jupyter) are simply absent from the
core list. Run with ``--list`` to print what is checked.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI = REPO_ROOT / "pixi.toml"
ENVIRONMENT = REPO_ROOT / "environment.yml"

# Core packages that MUST stay consistent across all three install paths.
# `names` lists every spelling the package goes by across PyPI and conda so the
# presence check matches regardless of ecosystem naming. `check_version` is
# False for packages pixi/pyproject intentionally leave unpinned ("*").
CORE_PACKAGES: List[Dict] = [
    {"canonical": "numpy", "names": {"numpy"}},
    {"canonical": "pandas", "names": {"pandas"}},
    {"canonical": "scipy", "names": {"scipy"}},
    {"canonical": "xarray", "names": {"xarray"}},
    {"canonical": "cftime", "names": {"cftime"}},
    {"canonical": "numexpr", "names": {"numexpr"}},
    {"canonical": "bottleneck", "names": {"bottleneck"}},
    {"canonical": "networkx", "names": {"networkx"}},
    {"canonical": "scikit-learn", "names": {"scikit-learn"}},
    {"canonical": "netcdf4", "names": {"netcdf4"}},
    {"canonical": "h5netcdf", "names": {"h5netcdf"}},
    {"canonical": "geopandas", "names": {"geopandas"}},
    {"canonical": "rasterio", "names": {"rasterio"}},
    {"canonical": "pyproj", "names": {"pyproj"}},
    {"canonical": "shapely", "names": {"shapely"}},
    {"canonical": "fiona", "names": {"fiona"}},
    {"canonical": "torch", "names": {"torch", "pytorch"}},
    {"canonical": "matplotlib", "names": {"matplotlib"}},
    {"canonical": "seaborn", "names": {"seaborn"}},
    {"canonical": "plotly", "names": {"plotly"}},
    {"canonical": "pyyaml", "names": {"pyyaml"}},
    {"canonical": "requests", "names": {"requests"}},
    {"canonical": "psutil", "names": {"psutil"}},
    {"canonical": "tqdm", "names": {"tqdm"}},
    {"canonical": "rich", "names": {"rich"}},
    {"canonical": "pydantic", "names": {"pydantic"}},
    {"canonical": "salib", "names": {"salib"}},
    {"canonical": "pvlib", "names": {"pvlib", "pvlib-python"}},
    {"canonical": "cdsapi", "names": {"cdsapi"}},
    {"canonical": "distributed", "names": {"distributed"}},
    {"canonical": "pint-xarray", "names": {"pint-xarray"}},
    {"canonical": "contextily", "names": {"contextily"}},
    {"canonical": "rasterstats", "names": {"rasterstats"}},
]


def normalize(name: str) -> str:
    """Normalise a package name for cross-ecosystem comparison."""
    # Drop extras/markers and channel prefixes (e.g. ``pytorch::pytorch``).
    name = name.strip().split("::")[-1]
    name = re.split(r"[\s;\[<>=!~]", name, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def parse_bounds(spec: str) -> Dict[str, Tuple[int, ...]]:
    """Parse ``>=X,<Y`` style constraints into ``{op: version-tuple}``."""
    bounds: Dict[str, Tuple[int, ...]] = {}
    for op, ver in re.findall(r"(>=|<=|==|>|<)\s*([0-9][0-9.]*)", spec):
        parts = ver.rstrip(".").split(".")
        bounds[op] = tuple(int(p) for p in parts if p.isdigit())
    return bounds


def _norm_version(v: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip trailing zeros so 2.0.0 == 2.0 == 2."""
    out = list(v)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def bounds_disagree(a: Dict[str, Tuple[int, ...]], b: Dict[str, Tuple[int, ...]]) -> List[str]:
    """Return the operators on which two bound sets disagree (both must pin it)."""
    diffs = []
    for op in (">=", ">", "<", "<="):
        if op in a and op in b and _norm_version(a[op]) != _norm_version(b[op]):
            diffs.append(f"{op}{'.'.join(map(str, a[op]))} vs {op}{'.'.join(map(str, b[op]))}")
    return diffs


def load_pyproject(path: Path) -> Dict[str, str]:
    """Return {normalized_name: version_spec} from [project] deps + optional deps."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: Dict[str, str] = {}
    entries = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        entries.extend(group)
    for entry in entries:
        name = normalize(entry)
        specs[name] = entry
    return specs


def load_pixi(path: Path) -> Dict[str, str]:
    """Return {normalized_name: version_spec} from pixi [dependencies] + [pypi-dependencies]."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    specs: Dict[str, str] = {}
    for section in ("dependencies", "pypi-dependencies"):
        for name, val in data.get(section, {}).items():
            if isinstance(val, dict):  # {version = "...", channel = "..."}
                val = val.get("version", "*")
            specs[normalize(name)] = str(val)
    return specs


def load_environment(path: Path) -> Set[str]:
    """Return the set of normalized package names declared in environment.yml."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names: Set[str] = set()
    for dep in data.get("dependencies", []):
        if isinstance(dep, str):
            names.add(normalize(dep))
        elif isinstance(dep, dict):  # the `pip:` sub-list
            for pip_dep in dep.get("pip", []):
                names.add(normalize(pip_dep))
    return names


def _present(names: Set[str], declared: Set[str]) -> bool:
    return bool(names & declared)


def _spec_for(names: Set[str], specs: Dict[str, str]) -> Optional[str]:
    for n in names:
        if n in specs:
            return specs[n]
    return None


def check_consistency() -> List[str]:
    """Return a list of human-readable consistency issues (empty == consistent)."""
    pyproject = load_pyproject(PYPROJECT)
    pixi = load_pixi(PIXI)
    env = load_environment(ENVIRONMENT)
    pyproject_names, pixi_names = set(pyproject), set(pixi)

    issues: List[str] = []
    for pkg in CORE_PACKAGES:
        canon, names = pkg["canonical"], pkg["names"]

        missing = [
            label
            for label, declared in (
                ("pyproject.toml", pyproject_names),
                ("pixi.toml", pixi_names),
                ("environment.yml", env),
            )
            if not _present(names, declared)
        ]
        if missing:
            issues.append(f"{canon}: missing from {', '.join(missing)}")
            continue  # can't compare bounds if a manifest lacks it

        py_spec = _spec_for(names, pyproject)
        px_spec = _spec_for(names, pixi)
        diffs = bounds_disagree(parse_bounds(py_spec or ""), parse_bounds(px_spec or ""))
        if diffs:
            issues.append(
                f"{canon}: version bounds disagree (pyproject '{py_spec}' vs pixi '{px_spec}'): "
                + "; ".join(diffs)
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the core package list and exit")
    args = parser.parse_args()

    if args.list:
        for pkg in CORE_PACKAGES:
            print(pkg["canonical"], "->", "/".join(sorted(pkg["names"])))
        return 0

    issues = check_consistency()
    if issues:
        print("Manifest consistency check FAILED — core dependencies disagree across install paths:\n")
        for issue in issues:
            print(f"  ❌ {issue}")
        print(
            "\nKeep pyproject.toml, pixi.toml, and environment.yml in sync for the listed packages.\n"
            "If a divergence is intentional, drop the package from CORE_PACKAGES in this script."
        )
        return 1

    print(f"Manifest consistency check passed ({len(CORE_PACKAGES)} core packages consistent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
