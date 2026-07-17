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

# Flat keys that remain supported for backward compatibility but are not canonical.
LEGACY_FLAT_TO_NESTED_ALIASES: Dict[str, Tuple[str, ...]] = {
    "MPI_PROCESSES": ("system", "num_processes"),
    "OPTIMIZATION_MAX_ITERATIONS": ("optimization", "iterations"),
    "INSTALL_PATH_MIZUROUTE": ("model", "mizuroute", "install_path"),
    "EXE_NAME_MIZUROUTE": ("model", "mizuroute", "exe"),
    "OPTIMIZATION_TARGET2": ("optimization", "nsga2", "secondary_target"),
    "OPTIMIZATION_METRIC2": ("optimization", "nsga2", "secondary_metric"),
    # Legacy mizuRoute settings keys
    "SETTINGS_MIZU_PATH": ("model", "mizuroute", "settings_path"),
    "SETTINGS_MIZU_WITHIN_BASIN": ("model", "mizuroute", "within_basin"),
    "SETTINGS_MIZU_ROUTING_DT": ("model", "mizuroute", "routing_dt"),
    "SETTINGS_MIZU_ROUTING_UNITS": ("model", "mizuroute", "routing_units"),
    "SETTINGS_MIZU_ROUTING_VAR": ("model", "mizuroute", "routing_var"),
    "SETTINGS_MIZU_OUTPUT_FREQ": ("model", "mizuroute", "output_freq"),
    "SETTINGS_MIZU_OUTPUT_VARS": ("model", "mizuroute", "output_vars"),
    "SETTINGS_MIZU_MAKE_OUTLET": ("model", "mizuroute", "make_outlet"),
    "SETTINGS_MIZU_NEEDS_REMAP": ("model", "mizuroute", "needs_remap"),
    "SETTINGS_MIZU_TOPOLOGY": ("model", "mizuroute", "topology"),
    "SETTINGS_MIZU_PARAMETERS": ("model", "mizuroute", "parameters"),
    "SETTINGS_MIZU_CONTROL_FILE": ("model", "mizuroute", "control_file"),
    "SETTINGS_MIZU_REMAP": ("model", "mizuroute", "remap"),
    "SETTINGS_MIZU_OUTPUT_VAR": ("model", "mizuroute", "output_var"),
    "SETTINGS_MIZU_PARAMETER_FILE": ("model", "mizuroute", "parameter_file"),
    "SETTINGS_MIZU_REMAP_FILE": ("model", "mizuroute", "remap_file"),
    "SETTINGS_MIZU_TOPOLOGY_FILE": ("model", "mizuroute", "topology_file"),
    # Legacy t-route settings keys
    "SETTINGS_TROUTE_PATH": ("model", "troute", "settings_path"),
    "SETTINGS_TROUTE_TOPOLOGY": ("model", "troute", "topology_file"),
    "SETTINGS_TROUTE_CONFIG_FILE": ("model", "troute", "config_file"),
    "SETTINGS_TROUTE_DT_SECONDS": ("model", "troute", "dt_seconds"),
    # NOTE: SETTINGS_DROUTE_PATH is NOT declared here — dRoute is an external plugin package
    # (like the JAX models) and declares its own config aliases via DRouteConfig +
    # model_manifest(config_schema=...). Keeping it here would duplicate the plugin's declaration.
    "SETTINGS_FUSE_PATH": ("model", "fuse", "settings_path"),
    "SETTINGS_GR_PATH": ("model", "gr", "settings_path"),
    "SETTINGS_GSFLOW_PATH": ("model", "gsflow", "settings_path"),
    "SETTINGS_HYPE_PATH": ("model", "hype", "settings_path"),
    "SETTINGS_MESH_PATH": ("model", "mesh", "settings_path"),
    "SETTINGS_RHESSYS_PATH": ("model", "rhessys", "settings_path"),
    "SETTINGS_SUMMA_PATH": ("model", "summa", "settings_path"),
    "SETTINGS_WATFLOOD_PATH": ("model", "watflood", "settings_path"),
    # Legacy SUMMA settings keys
    "SETTINGS_SUMMA_FILEMANAGER": ("model", "summa", "filemanager"),
    "SETTINGS_SUMMA_FORCING_LIST": ("model", "summa", "forcing_list"),
    "SETTINGS_SUMMA_COLDSTATE": ("model", "summa", "coldstate"),
    "SETTINGS_SUMMA_TRIALPARAMS": ("model", "summa", "trialparams"),
    "SETTINGS_SUMMA_ATTRIBUTES": ("model", "summa", "attributes"),
    "SETTINGS_SUMMA_OUTPUT": ("model", "summa", "output"),
    "SETTINGS_SUMMA_BASIN_PARAMS_FILE": ("model", "summa", "basin_params_file"),
    "SETTINGS_SUMMA_LOCAL_PARAMS_FILE": ("model", "summa", "local_params_file"),
    "SETTINGS_SUMMA_CONNECT_HRUS": ("model", "summa", "connect_hrus"),
    "SETTINGS_SUMMA_TRIALPARAM_N": ("model", "summa", "trialparam_n"),
    "SETTINGS_SUMMA_TRIALPARAM_1": ("model", "summa", "trialparam_1"),
    "SETTINGS_SUMMA_USE_PARALLEL_SUMMA": ("model", "summa", "use_parallel"),
    "SETTINGS_SUMMA_PARALLEL_BACKEND": ("model", "summa", "parallel_backend"),
    "SETTINGS_SUMMA_CPUS_PER_TASK": ("model", "summa", "cpus_per_task"),
    "SETTINGS_SUMMA_TIME_LIMIT": ("model", "summa", "time_limit"),
    "SETTINGS_SUMMA_MEM": ("model", "summa", "mem"),
    "SETTINGS_SUMMA_GRU_COUNT": ("model", "summa", "gru_count"),
    "SETTINGS_SUMMA_GRU_PER_JOB": ("model", "summa", "gru_per_job"),
    "SETTINGS_SUMMA_PARALLEL_PATH": ("model", "summa", "parallel_path"),
    "SETTINGS_SUMMA_PARALLEL_EXE": ("model", "summa", "parallel_exe"),
    "SETTINGS_SUMMA_GLACIER_MODE": ("model", "summa", "glacier_mode"),
    "SETTINGS_SUMMA_GLACIER_ATTRIBUTES": ("model", "summa", "glacier_attributes"),
    "SETTINGS_SUMMA_GLACIER_COLDSTATE": ("model", "summa", "glacier_coldstate"),
    "SETTINGS_SUMMA_SOILPROFILE": ("model", "summa", "soilprofile"),
    "SETTINGS_FUSE_FILEMANAGER": ("model", "fuse", "filemanager"),
    "SETTINGS_FUSE_PARAMS_TO_CALIBRATE": ("model", "fuse", "params_to_calibrate"),
    "SETTINGS_GR_CONTROL": ("model", "gr", "control"),
    "SETTINGS_HYPE_INFO": ("model", "hype", "info_file"),
    "SETTINGS_MESH_INPUT": ("model", "mesh", "input_file"),
}

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
