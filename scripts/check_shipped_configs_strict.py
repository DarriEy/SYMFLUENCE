#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Dogfood strict config-key validation on shipped configs (RTI Q3 / item 21).

SYMFLUENCE validates flat config keys at load time: unknown keys *warn* by
default and *raise* under ``STRICT_CONFIG`` / ``SYMFLUENCE_STRICT_CONFIG``
(see core/config/key_validation.py). Before recommending strict mode to users
we should dogfood it on our own shipped configs.

This guard runs the *real* strict validation against the configs we ship and:

* **enforces** that every config in ``STRICT_CLEAN`` validates with zero unknown
  keys (a regression here fails CI), and
* **reports** the unknown-key count for every other shipped config so the
  backlog stays visible, without failing the build.

Why not enforce all of them: an audit (2026-06) found the remaining unknowns are
almost entirely (a) keys for *non-active* models in kitchen-sink doc templates,
(b) features implemented in **external plugins** (whose keys only validate when
that plugin is installed), and (c) aspirational/roadmap keys. Of ~139 distinct
cross-cutting unknown keys across all shipped configs, **zero** had an in-tree
config consumer. Cleaning those is per-plugin and product-decision work tracked
separately; this guard locks in what is already clean and surfaces the rest.

Run: ``python scripts/check_shipped_configs_strict.py`` (``--report`` for the
full per-config table). Mirrored by tests/unit/config/test_shipped_configs_strict.py.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = "src/symfluence/resources/config_templates"

# Configs that MUST validate strictly clean. A regression here fails CI.
# Keep this list growing as plugins declare their keys and examples are cleaned.
STRICT_CLEAN: List[str] = [
    f"{TEMPLATES}/config_template.yaml",
    f"{TEMPLATES}/config_quickstart_minimal.yaml",
    # Nested-format templates: no flat keys, so they validate trivially clean.
    f"{TEMPLATES}/config_quickstart_minimal_nested.yaml",
    f"{TEMPLATES}/config_template_comprehensive_nested.yaml",
]

# Shipped configs intentionally NOT yet enforced, with the reason. These are the
# visible backlog; entries should be removed as each becomes clean.
EXEMPT: Dict[str, str] = {
    f"{TEMPLATES}/config_template_comprehensive.yaml":
        "kitchen-sink reference: documents keys for many models/features; not a single-model run config",
    f"{TEMPLATES}/camelsspat_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    f"{TEMPLATES}/fluxnet_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    f"{TEMPLATES}/norswe_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    "examples/**":
        "examples may exercise external-plugin features (DPE/SCF/FLASH/MIKESHE/multi-objective) "
        "whose keys validate only when that plugin is installed",
}


def _norm(key: str) -> str:
    from symfluence.core.config.legacy_aliases import NORMALIZATION_ALIASES
    return NORMALIZATION_ALIASES.get(key, key)


def _unknown_keys(path: Path) -> List[str]:
    """Return the flat keys in *path* that strict validation would reject."""
    import yaml

    from symfluence.core.config.key_validation import RESERVED_CONTROL_KEYS
    from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
    from symfluence.core.config.transformers import build_combined_flat_to_nested_map

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return []
    flat = {k for k in data if isinstance(k, str) and k == k.upper()}
    known = (
        set(build_combined_flat_to_nested_map(data.get("HYDROLOGICAL_MODEL")))
        | set(RECOGNIZED_FLAT_KEYS)
        | set(RESERVED_CONTROL_KEYS)
    )
    return sorted(k for k in flat if _norm(k) not in known)


def _is_exempt(rel: str) -> bool:
    for pattern in EXEMPT:
        if pattern.endswith("/**"):
            if rel.startswith(pattern[:-2]):
                return True
        elif rel == pattern:
            return True
    return False


def evaluate() -> Tuple[List[Tuple[str, List[str]]], List[Tuple[str, int]], List[str]]:
    """Return (failures, reported, unclassified)."""
    import symfluence  # noqa: F401  (runs registry bootstrap)

    failures: List[Tuple[str, List[str]]] = []
    reported: List[Tuple[str, int]] = []
    unclassified: List[str] = []

    for clean in STRICT_CLEAN:
        p = REPO_ROOT / clean
        if not p.exists():
            failures.append((clean, ["<file missing>"]))
            continue
        unknown = _unknown_keys(p)
        if unknown:
            failures.append((clean, unknown))

    shipped = sorted(
        glob.glob(str(REPO_ROOT / TEMPLATES / "*.yaml"))
        + glob.glob(str(REPO_ROOT / "examples" / "**" / "*.yml"), recursive=True)
        + glob.glob(str(REPO_ROOT / "examples" / "**" / "*.yaml"), recursive=True)
    )
    for f in shipped:
        rel = str(Path(f).relative_to(REPO_ROOT))
        if rel in STRICT_CLEAN:
            continue
        if _is_exempt(rel):
            try:
                reported.append((rel, len(_unknown_keys(Path(f)))))
            except Exception:  # noqa: BLE001 — reporting only, never fail here
                reported.append((rel, -1))
        else:
            unclassified.append(rel)
    return failures, reported, unclassified


def main(argv: List[str]) -> int:
    failures, reported, unclassified = evaluate()
    rc = 0

    if failures:
        rc = 1
        print("STRICT-CLEAN configs have unknown keys (must stay clean):\n", file=sys.stderr)
        for path, keys in failures:
            print(f"  {path}: {keys}", file=sys.stderr)

    if unclassified:
        rc = 1
        print(
            "\nNew shipped config(s) are neither STRICT_CLEAN nor EXEMPT — "
            "classify them in scripts/check_shipped_configs_strict.py:\n",
            file=sys.stderr,
        )
        for path in unclassified:
            print(f"  {path}", file=sys.stderr)

    if "--report" in argv:
        print("\nBacklog (exempt configs, unknown-key counts):")
        for path, n in sorted(reported, key=lambda x: -x[1]):
            print(f"  {n:4d}  {path}")

    if rc == 0:
        print(
            f"strict-config dogfood OK: {len(STRICT_CLEAN)} config(s) strict-clean; "
            f"{len(reported)} exempt (tracked backlog)."
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
