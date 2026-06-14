#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Dogfood strict config-key validation on shipped configs (RTI Q3 / item 21).

SYMFLUENCE validates flat config keys at load time: unknown keys *warn* by
default and *raise* under ``STRICT_CONFIG`` / ``SYMFLUENCE_STRICT_CONFIG``
(see core/config/key_validation.py). Before recommending strict mode to users
we should dogfood it on our own shipped configs.

This guard runs the *real* strict validation against the configs we ship and,
**enforce-by-default**:

* **enforces** that every shipped config validates with zero unknown keys
  (a regression fails CI), UNLESS the config is EXEMPT, and
* **reports** the unknown-key count for the exempt configs so the residual
  backlog stays visible.

Exempt = kitchen-sink doc templates (which document keys for many models and so
cannot be single-model clean) and the ``examples/paper_case_studies`` research
archive. Everything else — config_template.yaml, the quickstart/nested
templates, and the maintained tutorial examples — is enforced.

History: an audit (2026-06, issue #191) found the cross-cutting unknown keys had
**zero** in-tree config consumers; they were features that were removed from the
codebase (DPE), external plugins (jFUSE keys live in the jFUSE repo), or
roadmap/dead keys. Those were stripped from the maintained configs (DPE/ML
surrogate/MIKESHE removed; legacy spellings aliased), which is why the maintained
surface is now enforceable.

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

# Policy: every shipped config must validate strict-clean UNLESS it is exempt.
# A regression (a non-exempt config gaining an unknown key) fails CI.
#
# EXEMPT covers only (1) kitchen-sink doc templates that intentionally document
# keys for many models/features and so cannot be single-model clean, and (2) the
# paper_case_studies research archive, a historical record of original paper
# configs that may reference external-plugin features. The maintained tutorial
# examples (examples/01..05, examples/iceland_national_model, etc.) are enforced.
EXEMPT: Dict[str, str] = {
    f"{TEMPLATES}/config_template_comprehensive.yaml":
        "kitchen-sink reference: documents keys for many models/features; not a single-model run config",
    f"{TEMPLATES}/camelsspat_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    f"{TEMPLATES}/fluxnet_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    f"{TEMPLATES}/norswe_template.yaml":
        "dataset template: documents optional feature keys (gap-filling/emulation) not always wired",
    "examples/paper_case_studies/**":
        "research archive: historical record of original paper configs; may reference "
        "external-plugin (e.g. jFUSE) or non-active-model keys",
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
    """Return (failures, reported, enforced_paths).

    Enforce-by-default: every shipped config must be strict-clean unless it is
    EXEMPT. ``failures`` are non-exempt configs with unknown keys; ``reported``
    are exempt configs (with their unknown-key count, for the backlog view);
    ``enforced_paths`` is the list of configs that were enforced.
    """
    import symfluence  # noqa: F401  (runs registry bootstrap)

    failures: List[Tuple[str, List[str]]] = []
    reported: List[Tuple[str, int]] = []
    enforced: List[str] = []

    shipped = sorted(
        glob.glob(str(REPO_ROOT / TEMPLATES / "*.yaml"))
        + glob.glob(str(REPO_ROOT / "examples" / "**" / "*.yml"), recursive=True)
        + glob.glob(str(REPO_ROOT / "examples" / "**" / "*.yaml"), recursive=True)
    )
    for f in shipped:
        # Use POSIX separators so EXEMPT's forward-slash patterns match on Windows too.
        rel = Path(f).relative_to(REPO_ROOT).as_posix()
        if _is_exempt(rel):
            try:
                reported.append((rel, len(_unknown_keys(Path(f)))))
            except Exception:  # noqa: BLE001 — reporting only, never fail here
                reported.append((rel, -1))
            continue
        enforced.append(rel)
        unknown = _unknown_keys(Path(f))
        if unknown:
            failures.append((rel, unknown))
    return failures, reported, enforced


def main(argv: List[str]) -> int:
    failures, reported, enforced = evaluate()
    rc = 0

    if failures:
        rc = 1
        print(
            "Shipped configs have unknown keys under strict validation "
            "(fix the key, add a normalization alias, or — if it's a doc/archive "
            "config — exempt it in scripts/check_shipped_configs_strict.py):\n",
            file=sys.stderr,
        )
        for path, keys in failures:
            print(f"  {path}: {keys}", file=sys.stderr)

    if "--report" in argv:
        print("\nBacklog (exempt configs, unknown-key counts):")
        for path, n in sorted(reported, key=lambda x: -x[1]):
            print(f"  {n:4d}  {path}")

    if rc == 0:
        print(
            f"strict-config dogfood OK: {len(enforced)} config(s) enforced strict-clean; "
            f"{len(reported)} exempt (doc templates + research archive)."
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
