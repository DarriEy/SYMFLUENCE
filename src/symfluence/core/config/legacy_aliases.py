# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Legacy configuration aliases and canonical key preferences.

This module isolates compatibility shims so core transformer logic can treat
canonical mappings separately from deprecated/legacy aliases.

Three categories of aliases are defined here:

1. **Normalization aliases** (``NORMALIZATION_ALIASES``): spelling, product-name,
   and shorthand normalization applied during config loading (e.g.
   ``CONFLUENCE_DATA_DIR`` → ``SYMFLUENCE_DATA_DIR``).
2. **Deprecated keys** (``DEPRECATED_KEYS``): flat keys replaced by
   canonical successors but still accepted with a deprecation warning.
3. **Legacy flat-to-nested aliases** (``LEGACY_FLAT_TO_NESTED_ALIASES``):
   flat keys that map to nested config paths for backward compatibility.
4. **Recognized flat keys** (``RECOGNIZED_FLAT_KEYS``): real keys read in flat
   form (kept in the config ``_extra`` passthrough, not a nested field) that the
   unrecognized-key validator must treat as known. Recognition only, never
   transformation.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

# Normalization aliases: spelling, product-name, and shorthand corrections
# applied during config loading by config_loader._normalize_key().
NORMALIZATION_ALIASES: Dict[str, str] = {
    "GR_SPATIAL": "GR_SPATIAL_MODE",
    "OPTIMISATION_METHODS": "OPTIMIZATION_METHODS",
    "OPTIMISATION_TARGET": "OPTIMIZATION_TARGET",
    "OPTIMIZATION_ALGORITHM": "ITERATIVE_OPTIMIZATION_ALGORITHM",
    # Legacy CONFLUENCE naming (backwards compatibility)
    "CONFLUENCE_DATA_DIR": "SYMFLUENCE_DATA_DIR",
    "CONFLUENCE_CODE_DIR": "SYMFLUENCE_CODE_DIR",
    # The canonical flat keys for these two paths already carry the SYMFLUENCE_
    # prefix, so _load_env_overrides (which strips one leading SYMFLUENCE_) turns
    # a natural `SYMFLUENCE_DATA_DIR` env var into a bare DATA_DIR. Alias it back
    # so the natural env spelling resolves; the doubled form still works too.
    "DATA_DIR": "SYMFLUENCE_DATA_DIR",
    "CODE_DIR": "SYMFLUENCE_CODE_DIR",
    # Legacy domain discretization naming
    "DOMAIN_DISCRETIZATION": "SUB_GRID_DISCRETIZATION",
    # Legacy optimization-metric spelling -> canonical OPTIMIZATION_METRIC
    # (maps to optimization.metric). Renaming here makes old configs both
    # recognized and functional, rather than silently inert.
    "TARGET_METRIC": "OPTIMIZATION_METRIC",
    # Legacy SUMMA decisions spelling -> canonical SUMMA_DECISION_OPTIONS.
    "SUMMA_DECISIONS": "SUMMA_DECISION_OPTIONS",
    # Plain DECISION_OPTIONS historically carried SUMMA decisions (the CLI
    # preset path wrote it); nothing reads the plain spelling, so normalize
    # it to the canonical SUMMA key instead of letting it sit inert.
    "DECISION_OPTIONS": "SUMMA_DECISION_OPTIONS",
}

# Maps deprecated flat keys to their preferred replacements.
DEPRECATED_KEYS: Dict[str, str] = {
    # System legacy naming
    "MPI_PROCESSES": "NUM_PROCESSES",
    # MizuRoute legacy naming (inverted: INSTALL_PATH_MIZUROUTE -> MIZUROUTE_INSTALL_PATH)
    "INSTALL_PATH_MIZUROUTE": "MIZUROUTE_INSTALL_PATH",
    "EXE_NAME_MIZUROUTE": "MIZUROUTE_EXE",
    # NSGA-II secondary objective legacy naming
    "OPTIMIZATION_TARGET2": "NSGA2_SECONDARY_TARGET",
    "OPTIMIZATION_METRIC2": "NSGA2_SECONDARY_METRIC",
}

# Canonical + legacy key pairs used by model adapters for fallback validation.
MIZUROUTE_CANONICAL_LEGACY_KEY_PAIRS: Tuple[Tuple[str, str], ...] = (
    (DEPRECATED_KEYS["INSTALL_PATH_MIZUROUTE"], "INSTALL_PATH_MIZUROUTE"),
    (DEPRECATED_KEYS["EXE_NAME_MIZUROUTE"], "EXE_NAME_MIZUROUTE"),
)

# Canonical keys for nested paths with multiple aliases.
# When flattening nested config back to flat format, prefer these names.
CANONICAL_KEYS: Dict[Tuple[str, ...], str] = {
    ("system", "num_processes"): "NUM_PROCESSES",  # Prefer over MPI_PROCESSES
    ("optimization", "iterations"): "NUMBER_OF_ITERATIONS",  # Prefer over OPTIMIZATION_MAX_ITERATIONS
    ("optimization", "nsga2", "secondary_target"): "NSGA2_SECONDARY_TARGET",
    ("optimization", "nsga2", "secondary_metric"): "NSGA2_SECONDARY_METRIC",
    ("model", "mizuroute", "install_path"): "MIZUROUTE_INSTALL_PATH",
    ("model", "mizuroute", "exe"): "MIZUROUTE_EXE",
}

# Flat keys that remain supported for backward compatibility but are not
# canonical. Framework-level entries only: per-model legacy keys are declared
# by each model's config schema (a ``LEGACY_FLAT_ALIASES`` class attribute
# mapping legacy flat key -> schema field name) and collected from
# ``R.config_schemas`` by :func:`iter_legacy_flat_to_nested_aliases` — the
# same declaration path an external plugin uses (e.g. dRoute).
LEGACY_FLAT_TO_NESTED_ALIASES: Dict[str, Tuple[str, ...]] = {
    "MPI_PROCESSES": ("system", "num_processes"),
    "OPTIMIZATION_MAX_ITERATIONS": ("optimization", "iterations"),
    "OPTIMIZATION_TARGET2": ("optimization", "nsga2", "secondary_target"),
    "OPTIMIZATION_METRIC2": ("optimization", "nsga2", "secondary_metric"),
}


def iter_legacy_flat_to_nested_aliases() -> Dict[str, Tuple[str, ...]]:
    """All legacy flat->nested aliases: framework entries + schema-declared.

    Every schema registered in ``R.config_schemas`` may declare a
    ``LEGACY_FLAT_ALIASES`` class attribute mapping a legacy flat key to one of
    its field names; each expands to ``('model', <model key lower>, <field>)``.
    Central entries win on collision.
    """
    merged: Dict[str, Tuple[str, ...]] = {}
    try:
        from symfluence.core.registries import R

        for model_key, schema in list(R.config_schemas.items()):
            aliases = getattr(schema, "LEGACY_FLAT_ALIASES", None)
            if not isinstance(aliases, dict):
                continue
            prefix = ("model", str(model_key).lower())
            for flat_key, field_name in aliases.items():
                merged[flat_key] = (*prefix, field_name)
    except Exception:  # noqa: BLE001 — registry unavailable in stripped contexts
        pass
    merged.update(LEGACY_FLAT_TO_NESTED_ALIASES)
    return merged

# Recognized flat keys that are intentionally read in flat form — via
# ``config.get('KEY')`` or ``_get_config_value(..., dict_key='KEY')`` — and
# therefore flow through to the config object's ``_extra`` passthrough rather
# than a nested Pydantic field. They are real, consumed-in-code keys that the
# unrecognized-key validator (``key_validation``) must treat as KNOWN so it does
# not false-warn on them. They are deliberately NOT in LEGACY_FLAT_TO_NESTED_ALIASES:
# giving them a nested path would relocate them out of ``_extra`` and break the
# flat readers. Adding a key here changes recognition only, never transformation.
# (Resolves the bulk of RTI review open-question Q3 / Tier 3 item 21 noise; see
# docs/adr/0006-config-unknown-keys-warn-by-default.md. Conceptual-model and
# unbacked feature families are handled separately — see that ADR's follow-on.)
# The flat-key audit (docs/adr/config_flat_key_audit.md) emptied this set down
# to deprecated keys: every formerly-recognized key was either promoted to a
# typed Pydantic field (state/DA, IGNACIO/GNN/LSTM, multi-gauge,
# optimizer/evaluator odds, HYPE/NGEN/FUSE/GR/mizuRoute/paths, per-model
# *_PARAM_BOUNDS, and the data-handler families CanSWE/GLEAM/ESA-CCI/
# SNOTEL/SMAP/GRACE/GW/CARRA/HydroSHEDS/TDX), turned into an alias
# (OPTIMIZATION_MAX_ITERATIONS, DECISION_OPTIONS), or deleted as dead
# (EM_EARTH, MODIS_SNOW, USGS_GW, LSTM — template artifacts, no readers).
# Policy: this set is CLOSED. New config keys get Pydantic fields (or
# plugin-declared schemas, ADR-0002); only deprecated keys consumed by
# compatibility validators may live here, and they leave at the next major.
RECOGNIZED_FLAT_KEYS: frozenset[str] = frozenset({
    # Deprecated NGEN module toggles — consumed (and migrated to
    # NGEN_MODULES_SELECTED) by NGENConfig's before-validator; recognized
    # until removal at 2.0
    "ENABLE_NOAH", "ENABLE_PET", "ENABLE_SLOTH",
})


def find_missing_canonical_keys(
    config: Mapping[str, Any],
    canonical_legacy_pairs: Iterable[Tuple[str, str]],
) -> List[str]:
    """Return canonical keys missing from config after legacy fallback checks."""
    missing: List[str] = []
    for canonical_key, legacy_key in canonical_legacy_pairs:
        value = config.get(canonical_key) or config.get(legacy_key)
        if value in (None, "", "None"):
            missing.append(canonical_key)
    return missing
